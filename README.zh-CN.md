# FutureWorlds

机器人世界模型的完整训练与评测工程：**数据转换 → 冻结特征缓存 → SFT → MemSPO / GRPO / 普通束搜索后训练 → RGB 预测 → 指标与结果汇总**。

本目录已包含原生视觉编解码器、多卡训练、断点恢复和三个数据集的固定实验配方，不再只是核心算法摘录。先看 [一键复现指南](docs/REPRODUCTION.md) 和 [代码导航](docs/CODE_TOUR.md)。

> 当前为已公开的代码候选版：Linux CPU 19 项集成检查（含双进程 DDP）及三数据集真实权重首帧对齐已通过；新入口的全量 GPU 重训 / 384 样本 × 32 帧评测尚未运行。权重已私有整理，尚未发布到 Hugging Face。首次运行仍须提供权重、数据与本机路径，不能在尚无公开资源时声称“下载 ZIP 即自动复现整篇论文”。

## 一次配置，一条命令运行

```bash
# Extract FutureWorlds-code.zip, then enter its directory
cd FutureWorlds
./setup.sh
cp configs/paths.example.json configs/paths.local.json
# 编辑 paths.local.json：权重、数据、输出、T5 路径和 GPU 数量。
./run.sh --config configs/paths.local.json --suite main-table
```

程序核验权重、准备固定 128 个样本 / 数据集、生成 32 帧、汇总 10/20/32 帧指标，保存 `results.csv`、`results.md`、逐样本预测、token 和运行记录。中断后重跑相同命令会复用已校验完成的样本。

```bash
# SFT / GRPO / 普通束搜索 / MemSPO 的匹配消融
./run.sh --config configs/paths.local.json --suite matched-200
# 完整记忆、近期历史和最小历史，固定同一 MemSPO200 权重
./run.sh --config configs/paths.local.json --suite memory
# 从指定初始化权重重新训练，随后评测
./run.sh --config configs/train-workflow.local.json --suite train
# 只打印任务，不启动计算
./run.sh --config configs/paths.local.json --suite main-table --dry-run
```

训练配置从 `configs/train-workflow.example.json` 复制。`post-only` 从已整理的 SFT 权重进行后训练；`sft+post` 先运行 50k SFT。支持 8 卡等可整除全局 batch 的本地多卡配置。跨节点启动与编解码器训练见复现指南。

## 内容

| 部分 | 入口 |
|---|---|
| RT-1 / BridgeV2 / RoboCasa 数据、动作与划分 | `data.py`、`data_readers/`、`configs/cohorts/` |
| 原生 FSQ 编解码器及其训练 | `codec/`、`codec_training/` |
| RGB、动作、T5 到未来帧 | `pipeline.py` |
| 候选搜索、独立记忆与一致历史评分 | `search.py`、`ordinary_search.py`、`memory.py` |
| SFT / MemSPO / GRPO / 普通束搜索训练 | `training.py`、`configs/train/` |
| 缓存、恢复、导出与一键编排 | `cache.py`、`exporting.py`、`workflow.py` |
| PSNR / SSIM / LPIPS / MAE / MSE | `evaluation.py`、`reporting.py` |
| 权重校验与 Hugging Face 准备 | `checkpoints.py`、`scripts/`、`model_cards/` |

## 权重

已核验 **15 份预测器权重与 3 份编解码器，合计约 9.27 GB**，不含外部 T5。三个数据集均显式区分 `main`、`sft`、`memspo200`、`grpo200`、`ordinary200`。主表与匹配消融的模型和部分设备协议不同，见 [实验协议](docs/PROTOCOLS.md)。权重二进制不塞进源码 ZIP。

## 验证

```bash
USE_TF=0 .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python scripts/verify_sources.py
```

测试含真实原生编解码器、小模型 RGB 推理、SFT 中断恢复、三种后训练路径；Linux 双进程 CPU DDP 训练、恢复与导出测试也已通过；合成测试数据不代表论文性能。真实权重检查在三个数据集各一个样本上比较首个预测帧，80 个 token 及浮点像素均与原入口完全一致。详细边界见 `provenance/validation.json`。

更多文档：[数据格式](docs/DATA.md) · [权重加载](docs/LOADING.md) · [第三方来源](THIRD_PARTY_NOTICES.md) · [HF 发布准备](docs/HUGGINGFACE_RELEASE.md)。外部 Baseline 的独立训练仓库、WorldArena 扩展诊断和论文排版不属于本包的一键 suite；这里的一键主表命令复现 FutureWorlds 三行。
