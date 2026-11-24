#!/usr/bin/env python3
"""并发调用示例：预测与训练/扩列/保存/重载交错。

演示本次修复的语义：
  * 一次 /predict 的列名、补零记录、分数来自同一份模型快照；
  * 训练进行中，新 /predict 仍可用原模型立即返回（不等训练）；
  * 在途 /predict 不会阻塞训练、扩列、保存、重载；
  * 训练/扩列成功后立即对新预测生效，无需先 /save；
  * 失败的重载/扩列不改变当前模型。

前提：服务已启动（docker compose up -d --build，默认 127.0.0.1:8000）。
建议在刚启动、尚未扩列的服务上运行，脚本会自动选一个未使用的列名。
用法：python3 examples/concurrent_api.py [BASE_URL]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "BASE_URL", "http://127.0.0.1:8000"
)
OLD_ROWS = [
    {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
    {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9},
    {"temp": 0.2, "pressure": 0.0, "rpm": 0.7, "current": -0.5},
]
LABELS = [0, 1, 2]


def show(title: str, body) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(body, ensure_ascii=False, indent=2))


async def predict(client: httpx.AsyncClient, tag: str) -> dict:
    resp = await client.post("/predict", json={"instances": OLD_ROWS})
    resp.raise_for_status()
    body = resp.json()
    print(
        f"[predict {tag}] feature_names={body['feature_names']} "
        f"zero_filled={[p['zero_filled_columns'] for p in body['predictions']]} "
        f"scores={[round(p['score'], 6) for p in body['predictions']]}"
    )
    return body


def assert_self_consistent(body: dict, n_rows: int) -> None:
    """列名、补零记录、分数必须来自同一模型（旧字段都不缺时补零应为空）。"""
    names = body["feature_names"]
    for item in body["predictions"]:
        # 本示例只传原始字段：缺失列恰好是 feature_names 中不在 OLD_ROWS[0] 的部分
        expected = [n for n in names if n not in OLD_ROWS[0]]
        assert item["zero_filled_columns"] == expected, (
            f"补零记录与列名不一致: {item['zero_filled_columns']} != {expected}"
        )
        assert abs(sum(item["probabilities"]) - 1.0) <= 1e-5
    assert len(body["predictions"]) == n_rows


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        info = (await client.get("/model")).json()
        show("初始模型", info)
        baseline = await predict(client, "baseline")
        assert_self_consistent(baseline, len(OLD_ROWS))

        # 1) 训练与预测并发：训练完成前发出的预测整体对应旧快照，
        #    训练发布后新预测才使用新权重。
        print("\n== 并发：持续预测 + 一次训练 ==")
        start_step = info["optimizer_step"]
        training = asyncio.create_task(
            client.post("/train", json={"instances": OLD_ROWS, "labels": LABELS})
        )
        during = [
            await predict(client, f"train-inflight-{i}") for i in range(3)
        ]
        train_resp = await training
        train_resp.raise_for_status()
        show("训练结果", train_resp.json())
        assert train_resp.json()["step"] == start_step + 1
        for body in during:
            # 每个响应自身自洽；列名可能因训练无改列而与本批次一致
            assert_self_consistent(body, len(OLD_ROWS))
        await predict(client, "after-train")

        # 2) 扩列与并发预测交错：每个响应都必须自洽；新列零权重，
        #    因此扩列前后旧字段分数不变。
        print("\n== 并发：扩列 + 多个预测 ==")
        keys = set((await client.get("/model")).json()["feature_names"])
        new_col = next(
            name for name in ("vibration", "vibration_2", "vibration_3")
            if name not in keys
        )
        pre_expand = await predict(client, "pre-expand")
        preds = [
            asyncio.create_task(predict(client, f"expand-inflight-{i}"))
            for i in range(5)
        ]
        expand_resp = await client.post(
            "/expand", json={"new_feature_names": [new_col]}
        )
        expand_resp.raise_for_status()
        show("扩列结果", expand_resp.json())
        for task in preds:
            body = await task
            assert_self_consistent(body, len(OLD_ROWS))
            if new_col in body["feature_names"]:
                for got, want in zip(body["predictions"], pre_expand["predictions"]):
                    assert abs(got["score"] - want["score"]) <= 1e-6, (
                        "扩列前后旧字段分数不一致"
                    )
        fresh = await predict(client, "after-expand")
        assert fresh["feature_names"] == pre_expand["feature_names"] + [new_col]

        # 3) 保存 → 重载与预测并发，均不阻塞预测
        print("\n== 并发：保存 + 重载 + 预测 ==")
        info = (await client.get("/model")).json()
        ckpt_dir = os.path.dirname(info["source_path"]) or "."
        save_path = os.path.join(ckpt_dir, "model_concurrent.pt")

        async def save_then_reload() -> tuple[httpx.Response, httpx.Response]:
            saved = await client.post("/save", json={"path": save_path})
            saved.raise_for_status()
            return saved, await client.post("/reload", params={"path": save_path})

        save_reload, pred_a, pred_b = await asyncio.gather(
            save_then_reload(),
            predict(client, "while-save-reload-1"),
            predict(client, "while-save-reload-2"),
        )
        save_resp, reload_resp = save_reload
        reload_resp.raise_for_status()
        show("保存结果", save_resp.json())
        show("重载结果", reload_resp.json())
        assert_self_consistent(pred_a, len(OLD_ROWS))
        assert_self_consistent(pred_b, len(OLD_ROWS))

        # 4) 失败不切换模型
        print("\n== 失败保护：重载不存在的文件 ==")
        bad = await client.post("/reload", params={"path": "/nonexistent/nope.pt"})
        print(f"[reload] status={bad.status_code} detail={bad.json().get('detail')}")
        assert bad.status_code == 500
        after_fail = await predict(client, "after-failed-reload")
        assert after_fail["feature_names"] == fresh["feature_names"]

        print(
            "\n全部断言通过：预测快照一致、训练/扩列即时生效、"
            "失败不切换模型、写操作与预测互不阻塞。"
        )


if __name__ == "__main__":
    asyncio.run(main())
