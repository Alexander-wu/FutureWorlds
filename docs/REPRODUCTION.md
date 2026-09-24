# 训练与评测复现

## 1. 环境与资产

推荐 Linux CUDA 服务器。`./setup.sh` 创建独立 `.venv` 并安装完整依赖。已有匹配环境可设置 `PYTHON=/path/to/python` 后运行脚本。`requirements-linux.txt` 记录原实验环境版本；Dockerfile 是构建方案，尚未做干净容器构建验证。GPU 驱动须与安装的 PyTorch CUDA 版本兼容。

必须准备：

1. 已整理的权重 bundle，含 `manifest.json` 和三个数据集目录。15 份预测器及 3 个编解码器约 9.27 GB。
2. `google-t5/t5-base`，锁定 revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`。可在配置中指定该 revision 的本地快照。
3. 数据及经过审计的原始窗口 JSONL，或本工程的无损 NPZ + manifest。格式见 [DATA.md](DATA.md)。脚本不会自动另切一套随机数据来代替论文固定样本。
4. 评测 / RL 的 LPIPS-VGG 参数；`lpips` 首次初始化会通过 torchvision 下载 VGG 权重，可预先放入 PyTorch 标准缓存。

尚无已发布的 Hugging Face 仓库 ID。发布完成后可用以下命令下载并核验（占位符必须替换）：

```bash
futureworlds download --repo-id ACCOUNT/REPOSITORY --revision COMMIT_SHA --output /data/weights
```

## 2. 复现已有权重结果

复制 `configs/paths.example.json` 为 `configs/paths.local.json`，填写绝对路径。`sources` 仅在对应 `data/DATASET/evaluation/manifest.json` 不存在时使用。原始清单带旧绝对路径时，在对应数据集加 `"path_map": {"/old/data": "/new/data"}`。

```bash
./run.sh --config configs/paths.local.json --suite main-table --dry-run
./run.sh --config configs/paths.local.json --suite main-table
./run.sh --config configs/paths.local.json --suite matched-200
./run.sh --config configs/paths.local.json --suite memory
```

主表三行保留原协议：RT-1 / RoboCasa GPU FP32；Bridge CPU FP32。`gpus: 8` 会把 GPU 评测按固定样本索引分成 8 份，Bridge 主表不自动改为 GPU。匹配消融与记忆消融使用 GPU。不能为加速悄悄改变协议后要求数值逐位相同。

`limit: 1` 可先检查每数据集一个样本；结果会标记 `subset: true`，不能用于论文全量表。默认每数据集固定 128 个样本，不用新随机抽样。预测器只看两帧初始观测；未来真实帧仅用于指标。

输出为 `output/SUITE/DATASET/VARIANT/MEMORY/`，包含逐样本 `prediction.npz`、`result.json`、完整 `summary.json`，suite 根目录生成 CSV / Markdown。每份结果绑定数据、权重和协议指纹，中断后重跑会核验并跳过已完成样本，配置不同则拒绝混算。

## 3. 重新训练

复制 `configs/train-workflow.example.json` 为本地配置，指定训练 / 评测清单及初始化权重。

```bash
./run.sh --config configs/train-workflow.local.json --suite train --dry-run
./run.sh --config configs/train-workflow.local.json --suite train
```

流程是：无损数据准备 → 编解码器 / T5 缓存 → SFT（可选）→ 后训练 → 导出独立推理 bundle → 128 样本评测。

- `mode: post-only`：使用 bundle 的 `sft` 初始化，默认 200 步。
- `mode: sft+post`：先 50k SFT，须提供 `initial_checkpoint`，其哈希必须匹配配方。该初始化权重不包含在 15 份终点模型 bundle 中。
- `method` 可选 `memspo`、`grpo`、`ordinary`。
- `main_table: true` 且方法为 MemSPO 时，选主表配方：RT-1 500、Bridge E4/R1 200、RoboCasa 400。
- 三数据集训练可将 `training` 设置成对象列表，依次执行。原生全局 batch 不随本地 GPU 数自动改变，卡数须整除它。

`configs/schedules/` 保存后训练实际样本顺序；`sampling.py` 保留各数据集 SFT 抽样方式。修改超参数或样本顺序属于新实验。8 卡累积能保持全局 batch 和原 16 个逻辑槽的采样种子，但不同 GPU 数 / 内核不保证重训权重逐位一致。

检查点包含模型、适配器、optimizer、各 rank RNG 和指纹。重新执行同一 workflow 会恢复最新完整检查点；没有任何完整检查点的失败任务需新输出目录。直接 trainer 则在配置加入 `resume` 检查点路径。续训要求 world size 不变。已经导出的相同结果可重复核验复用。

跨节点可用自己的调度系统执行（两个节点分别填不同 node_rank，地址由用户环境提供）：

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=MASTER_HOST --master_port=29500 \
  -m futureworlds train --config configs/my-training.json
```

## 4. 编解码器训练

主表 / 后训练可直接使用已经训练的编解码器。需要从头重复 Bridge / RoboCasa 编解码器适配时，已保留原生 GAN worker、采样器和判别器；编辑 `configs/train/bridge_codec.json` 或 `robocasa_codec.json` 的资产路径。

```bash
# 原生 worker 的正式配方使用 16 个进程；先小步验证，再去掉 --smoke。
torchrun --standalone --nproc_per_node=16 -m futureworlds train-codec \
  --dataset bridge --config configs/train/bridge_codec.json --smoke
```

多节点使用上一节 torchrun 参数。原始 codec worker 接受其原生清单，不接受统一转换后的 manifest。RT-1 使用上游冻结编解码器，不把预训练上游整个模型宣称为本项目新增实验。

## 5. 当前已验证的范围

- 本地：核心检查、真实原生 codec + 小型随机 backbone 的 RGB 流水线、三种 RL 更新、SFT 恢复、导出及重复执行。macOS 跳过分布式检查；GitHub Actions 的 Linux CPU 环境已通过全部 19 项检查，包括双进程 DDP、恢复与导出。
- 实际服务器：三个数据集各 1 个样本，已整理真实权重；新旧入口首个预测帧的 80 个 token、浮点像素完全相同，未用 GPU。
- 尚未：新入口的完整 384 × 32 帧 GPU 回归、50k SFT / 全量 RL 重训、干净 Docker 构建、公开权重与数据下载闭环。

这些限制针对新整理入口；原论文已有实验结果不受改动。此整理过程未恢复 GPU 训练或占卡任务。
