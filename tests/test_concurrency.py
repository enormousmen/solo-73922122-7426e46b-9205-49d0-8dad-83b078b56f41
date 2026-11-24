"""并发与关键交错自测。

覆盖本次修复的语义：

1. 一次预测的列名、补零记录与分数都来自“开算时已生效”的同一模型快照；
   扩列/训练在其间完成也不会混搭新列名与旧权重。
2. 预测不参与写锁：慢训练不会拖住新预测；新请求在训练结束前仍用原模型。
3. 在途预测不阻塞训练、扩列、重载、保存。
4. 更新成功立即对新预测生效（无需先 /save）；更新失败继续用旧模型。
5. 训练/扩列/保存/重载仍由同一把锁串行，不丢失 step。

其中服务层用例用 threading.Event 精确卡住交错点；
HTTP 用例启动真实 uvicorn（随机端口），验证事件循环不被写操作阻塞。
"""
from __future__ import annotations

import os
import threading

import httpx
import pytest
import torch
import uvicorn

import app.main as main
import app.service as service_module
from app.checkpoint import load_checkpoint
from app.service import ModelService

torch.set_num_threads(1)
SRC = os.path.join("checkpoints", "model.pt")
OLD_FEATURES = ["temp", "pressure", "rpm", "current"]
ROWS = [
    {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
    {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9},
    {"temp": 0.2, "pressure": 0.0, "rpm": 0.7, "current": -0.5},
]
LABELS = [0, 1, 2]


def _new_service() -> ModelService:
    return ModelService(load_checkpoint(SRC))


def _probs(pred: dict) -> list[list[float]]:
    return [p["probabilities"] for p in pred["predictions"]]


def _max_err(a: list[list[float]], b: list[list[float]]) -> float:
    ta, tb = torch.tensor(a), torch.tensor(b)
    return (ta - tb).abs().max().item() if ta.numel() else 0.0


# --------------------------------------------------------------------------
# 服务层：确定性交错
# --------------------------------------------------------------------------
def test_predict_uses_single_snapshot_across_expand(monkeypatch):
    """预测卡在补零之后时完成扩列：该预测仍整体使用旧模型与旧列名。"""
    svc = _new_service()
    baseline = svc.predict(ROWS)

    entered = threading.Event()
    release = threading.Event()
    real_align = service_module.align_rows
    state = {"paused": False}

    def blocking_align(rows, feature_names, dtype):
        x, missing = real_align(rows, feature_names, dtype)
        if not state["paused"]:
            state["paused"] = True
            entered.set()
            assert release.wait(timeout=10), "测试等待超时"
        return x, missing

    monkeypatch.setattr(service_module, "align_rows", blocking_align)

    result: dict = {}

    def do_predict():
        result.update(svc.predict(ROWS))

    t = threading.Thread(target=do_predict)
    t.start()
    assert entered.wait(timeout=10), "预测未进入交错点"

    # 在途预测尚未返回时完成扩列（写操作不受预测影响）
    exp = svc.expand(["vibration"])
    assert exp["feature_names"] == OLD_FEATURES + ["vibration"]
    assert svc.bundle.feature_names == OLD_FEATURES + ["vibration"]

    release.set()
    t.join(timeout=10)
    assert not t.is_alive()

    # 在途预测：列名、补零、分数全部对应扩列前的模型
    assert result["feature_names"] == OLD_FEATURES
    assert [p["zero_filled_columns"] for p in result["predictions"]] == [[]] * len(ROWS)
    assert _max_err(_probs(result), _probs(baseline)) <= 1e-7

    # 扩列后新发起的预测才使用新列名，并正确回显补 0 列
    fresh = svc.predict(ROWS)
    assert fresh["feature_names"] == OLD_FEATURES + ["vibration"]
    assert all(p["zero_filled_columns"] == ["vibration"] for p in fresh["predictions"])


def test_inflight_prediction_sees_old_weights_while_train_updates(monkeypatch):
    """训练已推进（优化器 step 已执行）但尚未发布时，在途预测仍用旧权重。"""
    svc = _new_service()
    baseline = svc.predict(ROWS)

    entered = threading.Event()
    release = threading.Event()
    real_step = torch.optim.Adam.step

    def paused_step(self, *args, **kwargs):
        out = real_step(self, *args, **kwargs)
        entered.set()
        assert release.wait(timeout=10), "测试等待超时"
        return out

    monkeypatch.setattr(torch.optim.Adam, "step", paused_step)

    # 训练运行在一个线程里，卡在 Adam step 之后、发布之前
    train_result: dict = {}

    def do_train():
        train_result.update(svc.train_step(ROWS, LABELS))

    t = threading.Thread(target=do_train)
    t.start()
    assert entered.wait(timeout=10), "训练未进入交错点"

    # 此刻训练中的候选模型已更新，但尚未发布：预测必须仍是旧分数
    during = svc.predict(ROWS)
    assert _max_err(_probs(during), _probs(baseline)) <= 1e-7
    assert during["feature_names"] == OLD_FEATURES

    release.set()
    t.join(timeout=10)
    assert not t.is_alive()

    # 训练成功返回后，新预测立即使用新权重（无需保存）
    after = svc.predict(ROWS)
    assert _max_err(_probs(after), _probs(baseline)) > 0.0
    assert train_result["step"] == 201


def test_train_failure_keeps_previous_model(monkeypatch):
    """更新过程抛错：不发布，新预测继续使用此前模型。"""
    svc = _new_service()
    before_bundle = svc.bundle
    baseline = svc.predict(ROWS)

    def boom(self, *args, **kwargs):
        raise RuntimeError("模拟更新失败")

    monkeypatch.setattr(torch.optim.Adam, "step", boom)
    with pytest.raises(RuntimeError):
        svc.train_step(ROWS, LABELS)

    assert svc.bundle is before_bundle
    assert _max_err(_probs(svc.predict(ROWS)), _probs(baseline)) == 0.0


def test_reload_and_expand_failure_keep_previous_model():
    svc = _new_service()
    before_bundle = svc.bundle
    baseline = svc.predict(ROWS)

    with pytest.raises(FileNotFoundError):
        svc.reload(os.path.join("checkpoints", "does_not_exist.pt"))
    with pytest.raises(ValueError):
        svc.expand(["temp"])  # 重名列

    assert svc.bundle is before_bundle
    assert _max_err(_probs(svc.predict(ROWS)), _probs(baseline)) == 0.0


def test_train_takes_effect_without_saving(monkeypatch):
    """训练成功后立即对新预测生效，磁盘检查点不被隐式改写。"""
    svc = _new_service()
    with open(SRC, "rb") as fh:
        original_bytes = fh.read()
    baseline = svc.predict(ROWS)

    svc.train_step(ROWS, LABELS)
    after = svc.predict(ROWS)

    assert _max_err(_probs(after), _probs(baseline)) > 0.0
    with open(SRC, "rb") as fh:
        assert fh.read() == original_bytes, "训练不应隐式保存检查点"


# --------------------------------------------------------------------------
# HTTP 层：真实 uvicorn + 并发请求
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def server():
    config = uvicorn.Config(
        main.app, host="127.0.0.1", port=0, log_level="warning", lifespan="on"
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        threading.Event().wait(0.05)
    assert srv.started, "uvicorn 未能启动"
    port = srv.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture()
def http(server, monkeypatch):
    # 每个用例重置为初始检查点状态
    monkeypatch.setattr(main, "service", _new_service())
    with httpx.Client(base_url=server, timeout=30.0) as client:
        yield client


def _in_thread(fn, *args):
    box: dict = {}

    def run():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - 测试内收集
            box["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    return t, box


def test_slow_train_does_not_block_new_predictions(http, monkeypatch):
    """训练在途（未发布）时，新预测立刻用原模型返回。"""
    baseline = http.post("/predict", json={"instances": ROWS}).json()

    entered = threading.Event()
    release = threading.Event()
    original = ModelService.train_step

    def slow_train(self, instances, labels):
        entered.set()
        assert release.wait(timeout=15), "测试等待超时"
        return original(self, instances, labels)

    monkeypatch.setattr(ModelService, "train_step", slow_train)

    t, box = _in_thread(
        lambda: http.post("/train", json={"instances": ROWS, "labels": LABELS})
    )
    assert entered.wait(timeout=10), "训练未进入交错点"

    # 事件循环未被阻塞：预测立即返回旧模型结果
    resp = http.post("/predict", json={"instances": ROWS})
    assert resp.status_code == 200
    assert resp.json()["feature_names"] == OLD_FEATURES
    assert _max_err(_probs(resp.json()), _probs(baseline)) <= 1e-7
    assert http.get("/model").json()["optimizer_step"] == 200

    release.set()
    t.join(timeout=15)
    assert "error" not in box
    assert box["value"].json()["step"] == 201

    # 训练结束后新预测使用新权重
    after = http.post("/predict", json={"instances": ROWS}).json()
    assert _max_err(_probs(after), _probs(baseline)) > 0.0


def test_inflight_predict_does_not_block_writes(http, monkeypatch, tmp_path):
    """在途预测（卡在补零后）期间，训练/扩列/保存/重载都能先完成。"""
    entered = threading.Event()
    release = threading.Event()
    real_align = service_module.align_rows
    state = {"paused": False}

    def blocking_align(rows, feature_names, dtype):
        x, missing = real_align(rows, feature_names, dtype)
        if not state["paused"]:
            state["paused"] = True
            entered.set()
            assert release.wait(timeout=15), "测试等待超时"
        return x, missing

    monkeypatch.setattr(service_module, "align_rows", blocking_align)

    t, box = _in_thread(
        lambda: http.post("/predict", json={"instances": ROWS})
    )
    assert entered.wait(timeout=10), "预测未进入交错点"

    # 预测还挂着，写操作逐个完成（它们只彼此串行，不等预测）
    assert http.post("/train", json={"instances": ROWS, "labels": LABELS}).json()["step"] == 201
    assert http.post("/expand", json={"new_feature_names": ["vibration"]}).json()["input_dim"] == 5
    save_path = str(tmp_path / "model_snapshot.pt")
    assert http.post("/save", json={"path": save_path}).json()["ok"] is True
    reloaded = http.post("/reload", params={"path": save_path})
    assert reloaded.status_code == 200
    assert reloaded.json()["input_dim"] == 5

    release.set()
    t.join(timeout=15)
    assert "error" not in box
    # 在途预测仍整体对应它开算时的 4 列模型
    body = box["value"].json()
    assert body["feature_names"] == OLD_FEATURES
    assert all(p["zero_filled_columns"] == [] for p in body["predictions"])


def test_writes_are_serialized_without_lost_steps(http, monkeypatch):
    """并发训练请求串行执行：step 连续递增、无重复。"""
    original = ModelService.train_step

    def slow_train(self, instances, labels):
        threading.Event().wait(0.05)
        return original(self, instances, labels)

    monkeypatch.setattr(ModelService, "train_step", slow_train)

    results: list[dict] = []
    lock = threading.Lock()

    def one_train():
        resp = http.post("/train", json={"instances": ROWS, "labels": LABELS})
        assert resp.status_code == 200, resp.text
        with lock:
            results.append(resp.json())

    threads = [threading.Thread(target=one_train) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    steps = sorted(r["step"] for r in results)
    assert steps == [201, 202, 203, 204, 205]
    assert http.get("/model").json()["optimizer_step"] == 205


def test_failed_writes_keep_serving_old_model(http, tmp_path):
    """HTTP 层：扩列/重载/保存失败后继续用原模型预测。"""
    baseline = http.post("/predict", json={"instances": ROWS}).json()

    assert http.post("/expand", json={"new_feature_names": ["temp"]}).status_code == 422
    assert http.post("/reload", params={"path": str(tmp_path / "missing.pt")}).status_code == 500
    blocker = tmp_path / "blocked"
    blocker.write_text("x")
    assert http.post("/save", json={"path": str(blocker / "x.pt")}).status_code == 500

    after = http.post("/predict", json={"instances": ROWS}).json()
    assert after["feature_names"] == OLD_FEATURES
    assert _max_err(_probs(after), _probs(baseline)) == 0.0
    assert http.get("/model").json()["optimizer_step"] == 200
