# GPD-SC 独立代码审查（review_1）

- 审查对象：`D:\CODE\Doing\WHU_base\WEV1\WHU`
  - `processor/processor.py` md5 `8f74fb8082b9ee01c22a12b3c4dbc9e5`（2026-10-29 20:14，审查中新增了 480–488 行 per-epoch bank 日志，仅日志）
  - `config/defaults.py`、`configs/SCGPD_UAD.yml`、`datasets/make_dataloader.py`、`train.py`、`test.py`
- 原始 UAD 对照：`D:\CODE\Doing\WHU_Code\WHU`（只读）
- 设计对照：`WDV1_CC/research/phase3_analysis/04_synthesis.md` §6、§8

## 结论

**主实验（K=3、M=3、DDP、warm start、count EMA）可以定稿：P0 = 0，P1 = 2。**
两处 P1 都不阻塞主实验：P1-1 破坏设计 §8 的 K=1/K=2 消融臂；P1-2 使“暖启动 on/off”不是严格同种子单变量。原 UAD 默认路径（`GPD_TYPE='single'`、`GPD_WARMSTART=False`、`GPD_COUNT_EMA=False`、`GPD_TAU=0.03`）与原始实现逐行等价。

## P0 清单

无。

## P1 清单

### P1-1 子中心更新隐含假设 K == num_modalities；K=1/K=2 消融臂崩溃或静默污染 bank

- 位置：`processor/processor.py:139`（`idx = ids_all * K + mod_all`）、`:432-433`（`mod_ids = torch.arange(num_modalities).repeat_interleave(num_ids)`）、`:192-194`（warm start `len(loaders) != bank.K` 直接抛错）。
- 问题：`mod_ids` 取值域是 `[0, M)`，而 `idx` 用 `id*K + mod`，只有 `K == M` 时两者才一一对应。代码没有任何 `K == num_modalities` 校验。
  - K=1：mod=1/2 被映射到 `id+1`/`id+2`，末位身份越界，`index_add_` 直接 RuntimeError（实测 `index 9 is out of bounds for dimension 0 with size 8`）。
  - K=2：mod=2 被静默写到 `(id+1, slot0)`，实测 `bank[2,0]` 等于 id1 的 modality-2 中心（True），无报错、bank 被污染，训练结果不可解释。
  - K≠3 且 warm start 打开时在 `warm_start_subcenters` 直接 ValueError（loaders=3）。
- 失败场景：设计 §8 明确计划的 K=1/K=2/K=3 消融（“同一代码开关”）；§6.5 的失败回退“K=1（原 GPD）”；或换用模态数≠3 的数据集。主实验 K=3 不受影响。
- 最小修复：创建 bank 时 `if cfg.MODEL.GPD_SUB_CENTERS != num_modalities: raise ValueError(...)`；K=1 基线改用 `GPD_TYPE='single'`。或在 `update` 入口断言 `K == num_modalities`（当按模态索引更新时）。

### P1-2 warm start 推进全局 torch RNG，破坏 warmstart on/off 的同种子对比

- 位置：`processor/processor.py:342-352`（rank0 在训练前迭代 `train_normal_loaders`）、`datasets/make_dataloader.py:131-141`（`num_workers=cfg.DATALOADER.NUM_WORKERS=8`）。
- 问题：每个 DataLoader iterator 创建时都会从全局 torch CPU RNG 抽 `base_seed`；实测 `iter(DataLoader)` 会推进 `torch.get_rng_state()`。训练 loader 的 worker 种子在其之后才创建，因此 warmstart on/off 两次运行的随机翻转/擦除/裁剪流不同。
  - 身份采样走 np RNG（`shared_random_seed` + `np.random.seed`），不受影响；dropout/drop-path 走 CUDA RNG，warm start 在 eval（DropPath 不抽随机数），也不受影响。
  - 主实验（与独立 UAD 基线比较）不受影响；但设计 §8 的“暖启动 on/off”消融不是严格单变量，且 R1 的 seed 噪声（1.13pp）大于 gate（0.3pp），会混淆归因。
- 最小修复：warm start 前后保存/恢复 `torch.get_rng_state()`（如需保守再加 `torch.cuda.get_rng_state()`）；或给 warm-start 的三个 DataLoader 传独立 `torch.Generator`（注意 worker 种子派生）。

## 已验证项

- **默认路径不变**：diff 显示只新增分支；`GPD_TYPE='single'` 时仍用 `PrototypeBank`、原 `GPD`、`g_cent`、`tau=0.03`（现取自 `MODEL.GPD_TAU` 默认 0.03）、原 `proto_bank.update(ids, g_cent)`、原 loss 累加；`UAD.yml` 未改；`compileall` 通过。
- **PCA**：返回 4 值，唯一调用点已更新 unpack；`centers[m]` = 每 `(pid, modality)` 的 K 个 L2 特征均值再 L2，顺序与 `feat.reshape(M,B,D)`、`target.reshape(-1,K)` 一致；`g_cent`/`loss`/`ids` 与原实现逐位一致（CPU 对照测试）。
- **SubCenterPrototypeBank**：`[C,K,D]`/`counts [C,K]` float32、buffer 在 device；`initialize` shape 校验 + `copy_` + `counts=1`；`update` 的 `ids.repeat(M)` ↔ `cat(centers)`（modality-major）↔ `arange(M).repeat_interleave(P)` 一一对应；同批重复 `(id,mod)` 先聚合、counts 只加 1；count=0 直接赋值、count=1 时 `m_eff=0.5`、counts 递增；`view` 写入确实改到 buffer（均已 CPU 实测）。
- **GPD_SC**：`einsum('nd,ckd->nck')` + `max(dim=2)` 与显式参考一致；K=1 时与原 GPD loss 数值一致；`target.repeat(M)` 与 modality-major 特征对齐；全零 class 行回退 `g_cent` 复制 K 份、无 NaN；梯度只回 `feat`（bank 为 no_grad buffer）；bf16 autocast 下 loss 为 fp32 且梯度有限（CPU 代理测试）。
- **warm start**：三模态 loader 顺序 `[RGB, IR, Thermal]` → mode 1..3；`NECK_FEAT='before'` 返回 `global_feat=base(x)`（pre-BN，ViT 内无 BN）；每 `(pid,modality)` 均值/L2、缺失模态回退 identity mean、counts=1；`eval()→train()` 恢复；rank0 计算 + barrier + broadcast 所有 rank 同 cfg 到达；bank 不进 `state_dict`、每次运行重建（与设计一致）。
- **do_train/入口/配置**：`gpd_type` 读取与校验、bank 选择、warm start 条件正确；single 分支更新保留；`make_dataloader` 返回 7 元组，train/test unpack 正确；`SCGPD_UAD.yml` 的 K=3、tau=0.03、count EMA、warm start、120 epoch、batch 256、LR 0.032、wrt、PCA 0.01、base+pca+gpd 与 §8 一致；默认 `UAD.yml` 不受影响。

## 未验证 / 残余风险

- NCCL 路径（`broadcast` 的 float32 buffer、`all_gather` 的 int64/float32）本机 CPU-only 无法执行，仅静态确认 dtype/长度；`dist` 路径需在 4 卡上冒烟一次。
- DDP 每 rank 迭代数可能不齐：sampler 末尾 block 不足 `mini_batch_size` 时被 `BatchSampler(drop_last=True)` 丢弃，个别 rank 少 1 个 batch → 下一次 `all_gather`/DDP backward 挂起（概率约 total_rows mod (world*mini) ∈ {253..255}）。这是原 UAD `PrototypeBank` 路径同样存在的预先问题，不是 GPD-SC 回归；建议每 epoch 断言各 rank `len(train_loader)` 一致。
- `index_add_` 非确定性：各 rank 的 bank 可能有微小浮点分叉，且只在 warm start 同步一次（原 GPD 相同）；影响可忽略但非严格一致。
- AMP 跳步（inf/nan）时 bank 仍用该 batch 的 ProCA 中心更新；与原 GPD 一致，可接受。
- warm start 仅 rank0 做 3×|train| 次 ViT 前向，其余 rank 在 NCCL barrier 等待；2337 规模估算 1–3 分钟，远小于默认 10 分钟超时，但未实测。
- 设计 §6.5 要求监控子中心 usage/饿死；当前 per-epoch 日志的 `active per modality` 在 warm start 后恒为 C（`counts` 初始化为 1），无法反映真实 usage（仅日志问题，不影响训练）。

## 与设计不一致处

1. 设计 §8 的 K=1/K=2 消融与 §6.5 的“K=1（原 GPD）”回退在当前代码不可用（见 P1-1）；K=1 基线应走 `GPD_TYPE='single'`。
2. 设计 §8 的“更新目标：模态中心 vs 单样本”“分配：强制模态 vs argmax”消融没有代码开关；当前只实现“强制模态 + 模态中心”（主方案 A）。
3. 基线可比性：`SCGPD_UAD.yml` 是 batch 256/LR 0.032（4 卡 DDP），仓库内 `UAD.yml` 是 batch 64/LR 0.008；若直接与历史单卡 UAD 结果比较则不是单变量。需用同一 config 覆盖 `MODEL.GPD_TYPE single`、`GPD_WARMSTART False` 跑 DDP 匹配的 UAD 基线。
4. count-aware EMA 在 count≥9 后固定 `m_eff=0.1`，warm start 后约 9 次更新即进入 0.1 慢跟踪；与设计公式一致，但意味着子中心主要保留 warm-start 语义，需按 §6.5 监控。
