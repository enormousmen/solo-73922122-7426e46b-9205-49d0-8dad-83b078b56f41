# 设备振动测点扩列续训 API

产线加装振动测点后，分类模型可在**不停机、不重训**的前提下追加输入列并继续训练；
未改造设备仍用原有字段预测，缺失的新列自动补 0。基于 **FastAPI + CPU 版 PyTorch**
实现的单作业（single-job）API，附带已训练的两层分类模型及**含 Adam 状态**的检查点。

## 1. 目录结构

```
.
├── app/
│   ├── model.py          # 两层分类器与扩列（零权重追加列）
│   ├── checkpoint.py     # 检查点加载/扩列/原子保存（含 Adam 状态迁移）
│   ├── data.py           # 按列名对齐、缺列补 0、多余列忽略
│   ├── schemas.py        # 请求/响应模型
│   ├── service.py        # 单作业状态、训练一步、预测、保存、重载
│   └── main.py           # FastAPI 路由与启动加载
├── scripts/
│   └── train_initial.py  # 生成初始检查点（合成数据，200 步 Adam）
├── tests/
│   ├── test_core.py      # 数值对照 / 扩列 / Adam 状态 / 保存失败等 9 项
│   ├── test_api.py       # HTTP 端到端 7 个用例
│   └── test_concurrency.py # 并发/关键交错 9 个用例（服务层 + 真实 uvicorn）
├── checkpoints/
│   └── model.pt          # 已训练检查点（4 输入 → 8 隐藏 → 3 类，含 Adam 状态）
├── examples/
│   ├── expand_api.sh     # 扩列 API 示例
│   ├── retrain_api.sh    # 续训 API 示例
│   └── concurrent_api.py # 预测与训练/扩列/保存/重载并发调用示例
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 2. 模型与检查点

- 模型：`Linear(4→8) → ReLU → Linear(8→3)`，原始字段
  `temp, pressure, rpm, current`，3 个分类。
- 初始检查点由 `scripts/train_initial.py` 在合成三分类数据上训练 200 步得到
  （训练集准确率约 99%），文件内同时保存：
  - 模型 `state_dict`
  - **Adam 优化器完整状态**（`exp_avg`、`exp_avg_sq`、`step`、超参）
  - 特征列名、隐藏维度、类别数、dtype、格式版本
- 检查点格式版本字段 `version=1`，加载入口 `app.checkpoint.load_checkpoint`。

## 3. 扩列与续训的一致性设计

| 需求 | 实现方式 |
| --- | --- |
| 按列名对齐 | `app/data.py` 以模型 `feature_names` 为准建立列索引，请求中的列名映射到固定位置 |
| 缺新列补 0 | 对齐张量先以 0 初始化，缺失列保持 0；响应中通过 `zero_filled_columns` 回显 |
| 多余列忽略 | 不在 `feature_names` 中的键直接丢弃 |
| 扩列后旧分数不变 | 仅在 `fc1.weight` **尾部追加零初始化列**，偏置与 `fc2` 不变；零列对补 0 的旧样本前向无贡献 |
| 续训一步后原参数/Adam 一致 | 旧参数及其 `exp_avg`/`exp_avg_sq`/`step` 原样迁移；新列动量同样以 0 追加，参数顺序不变 |
| 新读数更新新权重 | 带 `vibration` 的样本使新列产生非零梯度，Adam 一步即可更新新增权重 |
| 确定性 | 单线程 CPU（`TORCH_NUM_THREADS=1`）+ `torch.use_deterministic_algorithms(True)` |

## 4. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/model` | 当前列名、维度、dtype、Adam step |
| POST | `/predict` | 预测；缺列补 0，返回标签、各类概率、补 0 列 |
| POST | `/train` | 用一批样本**训练一步**并返回训练前后 loss |
| POST | `/expand` | 追加一个或多个测点列（零权重 + Adam 零状态） |
| POST | `/save` | 原子保存检查点（先写临时文件再 `os.replace`） |
| POST | `/reload` | 从磁盘重载（模拟重启/回退），支持 query 参数 `path` |

训练、扩列、保存、重载由同一把异步锁串行化；预测为只读，不参与该锁。

### 并发修复需求对照

| 需求 | 实现 | 验证 |
| --- | --- | --- |
| 一次预测的列名、补零记录、分数对应开算时已生效的模型 | `predict()` 入口取一次 bundle 快照，响应直接由该快照生成 | `test_predict_uses_single_snapshot_across_expand` |
| 训练未结束时，新请求仍可用原模型完成预测 | 训练在深拷贝上执行，发布前 `_bundle` 不变；预测不加锁 | `test_inflight_prediction_sees_old_weights_while_train_updates`、`test_slow_train_does_not_block_new_predictions` |
| 已有预测在跑时，训练/扩列/重载可先完成 | 预测只持旧快照引用，写操作不等待预测 | `test_inflight_predict_does_not_block_writes` |
| 更新成功即对新预测生效，无需先保存 | 成功后一次引用赋值发布新 bundle | `test_train_takes_effect_without_saving` |
| 更新失败继续用此前模型 | 失败路径不执行引用替换（扩列/重载先在本地构建） | `test_train_failure_keeps_previous_model`、`test_reload_and_expand_failure_keep_previous_model`、`test_failed_writes_keep_serving_old_model` |
| 写操作串行、API 格式不变 | 共用 `svc.lock`；路由与请求/响应模型未改动 | `test_writes_are_serialized_without_lost_steps`、原有 16 项测试 |

### 并发与一致性（预测快照）

- **发布即不可变**：`ModelService` 持有一个 `CheckpointBundle` 快照。训练在**私有深拷贝**
  上完成，成功后仅用一次引用赋值发布新快照；扩列、重载同样先整体构建新 bundle 再替换。
  已发布对象在发布后不再被原地修改。
- **预测整体一致**：`/predict` 在入口只取一次快照，列名、补零记录、dtype 与前向计算
  全部来自该快照，响应中的 `feature_names` 与 `predictions` 严格对应同一模型。
  因此不会出现“用旧模型算分，却报出新列名”。
- **训练不阻塞新预测**：训练尚未结束时，新 `/predict` 仍用原模型立即返回；训练成功
  返回后，新预测立即使用新权重（无需先 `/save`）。
- **在途预测不阻塞写操作**：预测不获取写锁，持有旧快照的预测不会阻挡训练、扩列、
  保存或重载；CPU 密集的写操作通过 `asyncio.to_thread` 执行，事件循环不被占用。
- **失败不发布**：训练/扩列/重载抛错时不执行引用替换，后续预测继续使用此前的模型；
  保存失败（原子写）也不影响内存模型与已有检查点。
- **写操作仍串行**：所有写请求共用同一把锁，Adam step 连续递增，不会出现并发更新
  覆盖或丢步。

代价是每次训练/扩列会复制模型与 Adam 状态；本作业模型很小（4→8→3），开销可忽略。

### 保存与回退语义

- 保存走“临时文件 + 原子 rename”，**保存失败时原检查点与内存模型均不受影响**，服务继续可用。
- 保存新版本（如 `model_v2.pt`）后，旧文件 `model.pt` 保持不变，仍可通过 `/reload` 切回。
- 重启容器后从 `CHECKPOINT_PATH` 加载；把环境变量指向新版本即恢复扩列后的模型，
  旧字段预测与按新列续训均可用。

## 5. 本地运行（不使用 Docker）

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/train_initial.py          # （可选）重新生成 checkpoints/model.pt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. Docker Compose 启动

```bash
docker compose up -d --build
docker compose logs -f classifier-api
curl http://127.0.0.1:8000/model
```

`./checkpoints` 挂载进容器，容器内保存的新版本在宿主机同目录可见。
镜像以单 uvicorn worker 运行，保证进程内只有一个活动模型。

### 扩列示例

```bash
bash examples/expand_api.sh
```

核心请求：

```bash
curl -X POST http://127.0.0.1:8000/expand \
  -H "Content-Type: application/json" \
  -d '{"new_feature_names": ["vibration"]}'

# 未改造设备：只传原字段，vibration 自动补 0
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1}]}'
```

### 续训示例

```bash
bash examples/retrain_api.sh
```

核心请求（带新测点读数的样本）：

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1, "vibration": 0.25},
      {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9, "vibration": 0.51}
    ],
    "labels": [0, 1]
  }'
```

### 并发调用示例

```bash
python3 examples/concurrent_api.py                 # 默认 http://127.0.0.1:8000
python3 examples/concurrent_api.py http://host:8000
```

脚本用 `httpx.AsyncClient` 并发发起预测与写操作，并断言：

- 每个 `/predict` 响应的列名、补零记录与分数自洽（来自同一模型快照）；
- 训练进行中发出的预测仍对应旧权重，训练返回后新预测使用新权重；
- 扩列前后旧字段分数不变（新列零权重），在途预测不会报出未生效的新列名；
- 在途预测期间保存/重载可完成；
- 重载不存在的文件返回 500，当前模型不变。

也可手动并发调用：

```bash
# 一个终端持续预测
while true; do
  curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
    -d '{"instances": [{"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1}]}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["feature_names"], [p["zero_filled_columns"] for p in d["predictions"]])'
done

# 另一个终端训练 / 扩列
curl -s -X POST http://127.0.0.1:8000/expand -H "Content-Type: application/json" \
  -d '{"new_feature_names": ["vibration"]}'
```

保存与重启恢复：

```bash
curl -X POST http://127.0.0.1:8000/save \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/checkpoints/model_v2.pt"}'

# 重启后将 CHECKPOINT_PATH 指向 model_v2.pt，或显式 reload
curl -X POST "http://127.0.0.1:8000/reload?path=/app/checkpoints/model_v2.pt"
```

## 7. 自测与实测结果

```bash
python -m pytest tests/ -v
```

**实测环境**：Python 3.11 / torch 2.15.0+cpu / FastAPI 0.165，float32，单线程。
本机以 `uvicorn --workers 1` 等价方式运行（容器内命令相同）；本环境未安装 Docker，
见第 8 节。

需求中的两处硬性数值对照（12 条随机旧样本，全量 **25** 个测试通过）：

| 对照项 | 限值 | 实测最大绝对误差 |
| --- | --- | --- |
| 扩列后、续训前旧样本分类分数（softmax 概率） | 1e-6 | **0.000e+00** |
| 同批旧样本各训一步后的原参数（4 个张量） | 1e-6 | **0.000e+00** |
| 同上，对应 Adam 的 `exp_avg`/`exp_avg_sq`/`step` | 1e-6 | **0.000e+00** |

其他实测结论：

- 新列零初始化；仅喂旧字段训练一步后新列权重仍为 0；带 `vibration` 读数训练后新列权重变为非零。
- 保存到非法路径返回 500，原 `model.pt` 可重新加载、服务预测不受影响。
- 保存 → 重载（等价重启）→ 旧字段预测正常（补 0 列正确回显）→ 按新列续训，Adam step 正确递增（201→202→203）。
- 重名列、空扩列、越界标签、样本/标签数量不一致均返回 422。

### 并发/交错实测（`tests/test_concurrency.py`，9 项）

服务层用 `threading.Event` 精确卡住交错点，HTTP 层启动真实 uvicorn 并并发请求：

| 交错场景 | 断言 | 结果 |
| --- | --- | --- |
| 预测卡在“补零之后”时完成扩列 | 在途预测列名仍是旧 4 列、补零为空、分数与扩列前逐位一致；扩列后新预测才用新列名并回显 `vibration` | 通过 |
| 训练已执行 Adam step 但尚未发布时发起预测 | 在途预测分数与旧模型一致；训练返回后新预测分数改变；step=201 | 通过 |
| 训练/扩列/重载失败 | `service.bundle` 为原对象，预测分数不变；HTTP 返回 422/500 后继续服务 | 通过 |
| 训练成功后不调用 `/save` | 新预测立即使用新权重；磁盘 `model.pt` 字节不变 | 通过 |
| 训练在途 15s 时并发 `/predict` 与 `/model` | 立即返回旧模型结果，`optimizer_step` 仍为 200；训练完成后为 201 | 通过 |
| 预测在途时依次完成训练/扩列/保存/重载 | 四个写操作均完成；在途预测仍整体对应 4 列旧模型 | 通过 |
| 5 个并发训练请求 | step 恰为 201..205，无重复/丢步 | 通过 |
| 并发预测 + 训练/扩列 | 每个响应的列名与 `zero_filled_columns` 自洽；扩列前后旧字段分数 ≤1e-6 | 通过 |
| 12 路并发预测 + 4 次扩列 × 3 轮续训/保存/重载（压测） | 1054 次预测，0 个错误；每次响应自洽 | 通过 |

`examples/concurrent_api.py` 在真实 uvicorn（`--workers 1`）上跑通上述并发流程。
另外把 `predict`/`train_step` 临时改回“读实时 bundle、原地训练”的旧实现时，
这些交错用例会失败，说明测试能捕获原缺陷。

## 8. 已实现功能与未完成项

**已实现**

- 单作业 FastAPI 服务，CPU PyTorch，启动即加载含 Adam 状态的检查点
- 两层分类模型初始训练脚本与随附检查点
- 按列名对齐、缺列补 0、多余列忽略，预测响应回显补 0 列
- 追加测点列（支持一次多列、重名拒绝），零权重 + Adam 零状态迁移
- 扩列前后旧样本分数、续训一步后原参数与 Adam 状态逐位一致（误差 0 ≤ 1e-6）
- 带新读数样本可更新新增权重；检查点原子保存、失败保护、重载恢复
- **并发安全**：预测使用单一不可变快照，训练/扩列/重载采用“副本构建 + 原子发布”；
  写操作串行、预测不加写锁；更新成功即时生效（无需保存），失败保持旧模型
- **不阻塞事件循环**：训练/扩列/保存/重载经 `asyncio.to_thread` 执行
- Dockerfile + docker-compose（单 worker、检查点卷挂载、健康检查）
- 扩列/续训/并发 API 示例与 25 项自动化测试

**未完成 / 边界说明**

- 当前环境未安装 Docker，**镜像未在本机实际构建运行**；Dockerfile/compose 已按 CPU wheel 与单作业要求编写，需在有 Docker 的环境执行 `docker compose up -d --build` 验证。
  本次并发验证使用本机 `uvicorn --workers 1`（与 compose 等价）；`examples/concurrent_api.py` 可直接对 compose 服务运行。
- 训练接口固定为“每请求一步”的小批量增量训练，未实现多 epoch 调度、学习率调度器与早停。
- 未接入鉴权/限流与多作业并发；服务按单作业设计，写操作由进程内锁串行化。
- 并发发布模型为**进程内单写者**语义：多 uvicorn worker / 多副本会各自持有独立模型，需外部路由或共享存储协调，当前不在范围内。
- 每次训练/扩列会深拷贝模型与 Adam 状态（当前 4→8→3 规模可忽略）；若模型显著增大，可改为参数级写时复制或原地更新 + 读锁。
- 合成数据仅用于演示与数值核对；接入真实产线数据时需替换 `scripts/train_initial.py` 的数据来源并重新标定类别。
