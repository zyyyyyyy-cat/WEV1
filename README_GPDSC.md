# GPD-SC：UAD 的模态子中心全局原型判别

## 改动

- 保留 UAD/ViT、CE + WRT + 0.01·ProCA + GPD 总损失、pre-BN CLS 推理，完全不变。
- 只把 GPD 的 `[C,768]` 单原型替换为 `[C,3,768]` 模态子中心：
  - K=3，对应 RGB/NIR/TIR；
  - `logits[c] = max_k cos(f, P[c,k]) / tau`，CE 不变；
  - 初始化：训练集每个 `(pid, modality)` 的类均值；
  - 更新：用 ProCA 的模态中心（同身份同模态 K=4 特征均值）更新对应子中心，不用 `g_cent`；
  - 计数 EMA：`m_eff = clamp(1/(count+1), 0.1, 0.5)`；
  - DDP all-gather 后更新。
- 不改推理：仍用 pre-BN CLS + L2 + 欧氏；不加载 bank。
- 默认 `GPD_TYPE='single'` 时，原 UAD 路径不变。

## 关键文件

| 文件 | 改动 |
|---|---|
| `config/defaults.py` | `GPD_TYPE/GPD_SUB_CENTERS/GPD_TAU/GPD_COUNT_EMA/GPD_EMA_MIN/MAX/GPD_WARMSTART` |
| `configs/SCGPD_UAD.yml` | 唯一训练配置：120 epoch、K=3、warm start、count EMA |
| `processor/processor.py` | `PCA` 返回模态中心；`SubCenterPrototypeBank`；`warm_start_subcenters`；`GPD_SC`；`do_train` 分支 |
| `datasets/make_dataloader.py` | 返回三模态确定性 `train_normal_loaders`（只用于暖启动） |
| `train.py` / `test.py` | 适配新的 loader 返回 |

## DDP 训练（4 卡，首选）

```bash
cd /path/to/WEV1/WHU

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/SCGPD_UAD.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/SCGPD_1000_DDP
```

## 单卡

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config_file configs/SCGPD_UAD.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_PATH /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008 \
  OUTPUT_DIR logs/SCGPD_1000_single
```

## 测试

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/SCGPD_UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.ROOT_DIR /project/code/data \
  TEST.WEIGHT logs/SCGPD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/SCGPD_1000_DDP/test
```

1000GD：同一权重加 `DATASETS.FILTER_GD True`；2337：`DATASETS.DATASET_DIR WHU-MARS-2337`，独立训练。

## 训练时看什么

1. `GPD-SC bank: active per modality [...]`：三个子中心的活跃数是否都接近 500（1000）/1000（2337）；若某一模态明显少，说明该模态子中心饿死。
2. `GPD` 损失量级：与 baseline 的 GPD loss 对比；若明显更小/更大，先看 `max_k` 抬高 logit 的影响，再考虑 tau。
3. 每 10 epoch 的 mAP/R1/R5/R10；尾 30 epoch 的绝对斜率。
4. 九组合：TIR→TIR/TIR→RGB/RGB→TIR 是否改善；同模态 R1 是否掉。
5. 与 H014 对比：mAP/R1 至少 +0.3、R5/R10 至少 +0.2，且 mINP 不降，才进入 2–3 seed。

## 注意

- 第一轮只跑 GPD-SC，不开 EMA/TTA/CA-WRT/LLRD/margin。
- `GPD_SUB_CENTERS` 必须等于模态数（本数据为 3）；K=1 基线用 `GPD_TYPE='single'`；K=2 暖启动未实现。代码会在 bank 创建时断言，避免 K≠模态数时静默污染 bank。
- 对照基线用 **H014 DDP 4 卡 batch 256/LR 0.032**；若用单卡跑 SCGPD，必须用同单卡 64/0.008 的 UAD 基线，不能直接对比历史单卡数字。
- Bank 是 processor buffer，不保存在 checkpoint；每次训练/恢复会用 warm start 重建。
- Warm start 前后保存/恢复 torch CPU/CUDA RNG，保证 warmstart on/off 的训练增强随机流一致。
- 若暖启动阶段显存不足，降低 `TEST.IMS_PER_BATCH`（512→256/128）或 `DATALOADER.NUM_WORKERS`；不影响训练 batch。
- 原 UAD 默认配置 `configs/UAD.yml` 仍然走 `GPD_TYPE='single'`，行为不变。
