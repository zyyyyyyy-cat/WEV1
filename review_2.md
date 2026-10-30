# GPD-SC 修复复核（review_2）

- 复核对象：`D:\CODE\Doing\WHU_base\WEV1\WHU\processor\processor.py` md5 `273916b134a33e72f1c6ad20078d708b`（2026-10-29 20:19）
- 对照：`review_1.md`；其余文件（defaults.py、SCGPD_UAD.yml、make_dataloader.py、train.py、test.py）hash 与上一轮一致，未改动。

## 结论

两个 P1 均已按预期修复，未引入新 P0/P1；**主实验 K=3 仍可定稿（P0=0，P1=0）**。`compileall` 通过，原 CPU 冒烟套件重跑全部通过。

## P1-1 复核：K 必须等于模态数

- 位置：`processor.py:325-331`，位于 `gpd_type == 'subcenter'` 分支内、创建 `SubCenterPrototypeBank` 之前、warm start 与所有集合通信之前。
- 实现：`num_modalities_cfg = len(cfg.DATASETS.MODALITIES)`；不等即 `ValueError`，报错文案明确指向 `GPD_TYPE='single'` 作为 K=1 基线。
- 判定：**已修复**。所有 rank 的 cfg 相同，会在任何 `all_gather`/`barrier` 之前一致抛错，不会挂起；默认 `GPD_TYPE='single'` 路径不进入该分支，原 UAD 路径不受影响；K=3/M=3 主配置 3==3 通过。
- 备注：协调者提到的 README 说明未落盘（`README.md` mtime/hash 仍为原始版本，全仓库仅在报错文案中出现 `GPD_TYPE='single'`）。属文档缺失，不影响代码修复判定。

## P1-2 复核：warm start RNG 保存/恢复

- 位置：`processor.py:354-371`。save 在 warm start 之前，restore 在 DDP barrier/broadcast 或单进程 warm start 之后，位于 dist 分支之外，所有 rank 都会执行（非 rank0 本无可恢复状态，等价 no-op）。
- 验证：按实际代码模式做 CPU 模拟——warm start 前迭代 DataLoader（消耗 base_seed）→ 恢复 `torch.get_rng_state()` → 后续 RNG 流与无 warm start 基线逐位一致（True）；不恢复的对照组则不同（True）。CUDA RNG 同理；warm start 在 `eval()` 下 DropPath/dropout 不抽随机数，CUDA 恢复为无害 no-op。
- 判定：**已修复**，warmstart on/off 的训练增强随机流对齐；身份采样（np RNG）与 dropout（CUDA RNG）本就未受影响。barrier/broadcast 顺序未变，无新死锁风险。

## usage 日志

- 位置：`processor.py:499-511`。`updated = (counts > 1).sum(dim=0)`、`mean_count = counts.mean(dim=0)`，表达式实测可用；warm start 后 counts 从 1 起，`>1` 正确表示“至少更新过一次”，不再恒为 C。
- 备注（仅日志）：warm start 关闭时 counts 从 0 起，恰更新一次的 slot 显示为 0，会低估 usage；主实验 warm start 开启，不受影响。

## 新问题检查

- 新增异常发生在 bank 创建/集合通信之前，多 rank 行为一致，不会造成 DDP hang。
- 训练路径（PCA 4 值 unpack、GPD_SC/GPD 分支、`ids.repeat(M)`/`mod_ids`/`centers_all`、single 分支 `update(ids, g_cent)`、loss 累加）逐行未变；冒烟套件 T1–T10 重跑通过。
- 无新 P0/P1。

## 残余风险（沿用 review_1，非本次修复引入）

- NCCL 的 broadcast/all_gather 仍未在 GPU 上实测；DDP 各 rank 迭代数不齐的预先风险仍在（原 UAD 同样存在）。
- 建议跑前断言各 rank `len(train_loader)` 一致。
