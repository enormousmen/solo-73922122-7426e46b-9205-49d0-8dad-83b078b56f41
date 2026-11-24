"""模型服务：单作业内存状态、训练/预测/扩列/保存/重载。

并发模型
--------
- **发布即不可变**：`self._bundle` 指向的模型/优化器在发布后不再被原地修改。
  训练在私有深拷贝上进行，成功后才用一次引用赋值（CPython 下原子）发布新 bundle；
  扩列、重载同样先构建完整的新 bundle，再整体替换。
- **预测无锁**：预测在开始时读取一次 `self._bundle` 快照，随后列名、补零、
  前向计算全部取自同一快照，因此不会出现“旧模型算分、新列名回显”的混搭。
  在途预测持有旧快照的引用，不会被后续训练/扩列/重载影响。
- **写操作串行**：训练、扩列、保存、重载由调用方通过同一把 `asyncio.Lock` 串行化；
  预测不参与该锁，因此慢训练不会拖住新预测，在途预测也不会阻塞写操作。
- **失败不发布**：任何一步更新抛错时都不会执行引用替换，新预测继续使用此前的模型。
- **无需先保存**：训练/扩列成功后内存中的新 bundle 立即对新预测生效，
  是否落盘由 `/save` 单独决定。

CPU 密集的写操作应由 API 层放到工作线程执行（`asyncio.to_thread`），
避免阻塞事件循环；本模块的函数保持同步、可直接测试。
"""
from __future__ import annotations

import asyncio
import copy
import os

import torch

from .checkpoint import (
    CheckpointBundle,
    expand_bundle,
    load_checkpoint,
    save_checkpoint_atomic,
)
from .data import align_labels, align_rows


class ModelService:
    def __init__(self, bundle: CheckpointBundle):
        self._bundle = _in_eval_mode(bundle)
        self._lock = asyncio.Lock()

    # ---- 基础信息 -------------------------------------------------
    @property
    def bundle(self) -> CheckpointBundle:
        """当前已发布（对后续预测生效）的 bundle 快照。"""
        return self._bundle

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def info(self) -> dict:
        b = self._bundle
        return {
            "feature_names": list(b.feature_names),
            "input_dim": len(b.feature_names),
            "hidden_dim": b.hidden_dim,
            "num_classes": b.num_classes,
            "dtype": str(b.dtype).replace("torch.", ""),
            "optimizer_step": _read_step(b.optimizer),
            "source_path": b.source_path,
        }

    # ---- 预测 -----------------------------------------------------
    @torch.no_grad()
    def predict(self, instances: list[dict]) -> dict:
        """用“开算时已生效”的模型整批完成预测。

        只在入口读取一次 bundle：列名、dtype、模型都来自同一快照。
        返回体中的 feature_names 与 predictions 因而严格对应。
        """
        b = self._bundle  # 快照；发布后不会被原地修改
        feature_names = b.feature_names
        x, missing = align_rows(instances, feature_names, b.dtype)
        logits = b.model(x)
        probs = torch.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        predictions = [
            {
                "label": int(labels[i]),
                "score": float(probs[i, labels[i]]),
                "probabilities": [float(p) for p in probs[i]],
                "zero_filled_columns": missing[i],
            }
            for i in range(len(instances))
        ]
        return {"feature_names": list(feature_names), "predictions": predictions}

    # ---- 训练一步 -------------------------------------------------
    def train_step(self, instances: list[dict], labels: list[int]) -> dict:
        """在私有副本上训练一步，成功后原子发布；失败则保留原模型。"""
        if len(instances) != len(labels):
            raise ValueError(
                f"instances 数量({len(instances)})与 labels 数量({len(labels)})不一致"
            )

        current = self._bundle
        # 先做输入校验/对齐（只读），避免为非法请求复制模型
        x, _ = align_rows(instances, current.feature_names, current.dtype)
        y = align_labels(labels, current.num_classes)

        # 深拷贝模型与 Adam 状态：训练只改副本，已发布模型对在途预测始终不变。
        # 本作业模型很小（4→8→3），拷贝开销可忽略。
        candidate = copy.deepcopy(current)

        criterion = torch.nn.CrossEntropyLoss()
        candidate.model.train()

        with torch.no_grad():
            loss_before = float(criterion(candidate.model(x), y))

        candidate.optimizer.zero_grad()
        logits = candidate.model(x)
        loss = criterion(logits, y)
        loss.backward()
        candidate.optimizer.step()

        with torch.no_grad():
            loss_after = float(criterion(candidate.model(x), y))

        result = {
            "step": _read_step(candidate.optimizer),
            "loss_before": loss_before,
            "loss_after": loss_after,
            "num_samples": len(instances),
            "feature_names": list(candidate.feature_names),
        }

        # 发布前恢复推理模式（本模型无 Dropout/BatchNorm，数值不受影响；
        # 避免未来加入这类层时预测结果依赖训练态）。
        candidate.model.eval()

        # 更新成功：一次引用赋值发布，之后开始的预测立即使用新权重（无需先保存）。
        self._bundle = candidate
        return result

    # ---- 扩列 -----------------------------------------------------
    def expand(self, new_feature_names: list[str]) -> dict:
        names = list(dict.fromkeys(new_feature_names))  # 去重但保序
        if len(names) != len(new_feature_names):
            raise ValueError("new_feature_names 内部存在重复列名")
        # expand_bundle 只读旧 bundle、构建全新模型/优化器；失败时抛出且不发布
        new_bundle = _in_eval_mode(expand_bundle(self._bundle, names))
        self._bundle = new_bundle  # 原子发布；在途预测仍持有旧快照
        return {
            "feature_names": list(new_bundle.feature_names),
            "input_dim": len(new_bundle.feature_names),
            "added": names,
        }

    # ---- 保存（原子） ---------------------------------------------
    def save(self, path: str | None) -> str:
        b = self._bundle  # 已发布 bundle 不可变，序列化期间不会被训练改写
        target = os.path.abspath(path) if path else b.source_path
        save_checkpoint_atomic(b, target)
        return target

    # ---- 从磁盘重载（重启/回退用） --------------------------------
    def reload(self, path: str | None = None) -> dict:
        b = self._bundle
        target = os.path.abspath(path) if path else b.source_path
        # 先在本地完整加载，成功后再发布；失败时继续使用此前的模型
        loaded = _in_eval_mode(load_checkpoint(target, dtype=b.dtype))
        self._bundle = loaded
        return {
            "path": loaded.source_path,
            "feature_names": list(loaded.feature_names),
            "input_dim": len(loaded.feature_names),
            "optimizer_step": _read_step(loaded.optimizer),
        }


def _in_eval_mode(bundle: CheckpointBundle) -> CheckpointBundle:
    """把 bundle 置为推理模式后返回（不改权重与优化器状态）。"""
    bundle.model.eval()
    return bundle


def _read_step(optimizer: torch.optim.Optimizer) -> int:
    """兼容不同 PyTorch 版本读取 Adam 的 step。"""
    state = optimizer.state
    if not state:
        return 0
    step = next(iter(state.values())).get("step", 0)
    if torch.is_tensor(step):
        return int(step.item())
    return int(step)
