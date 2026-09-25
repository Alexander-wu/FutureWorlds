# 代码导航

建议依次读：`workflow.py` → `pipeline.py` → `training.py` → `search.py` / `memory.py` → `evaluation.py`。

## 端到端调用

```text
run.sh → cli.run → workflow.run
  main-table / matched-200 / memory
    data.convert_manifest → cli.reproduce → evaluation.evaluate → reporting.report
  train
    data.convert_manifest → cache.cache_dataset
    training.train (SFT → MemSPO / GRPO / ordinary)
    exporting.export_model → evaluation.evaluate
```

## 核心算法

- `search.rollout_group_beam`：4 组 × 2 分支，全视频候选搜索，返回 4 条轨迹。
- `ordinary_search.rollout_ordinary_beam`：普通全视频 beam8 返回 top4，候选构造消融。
- `memory.rollout`：GRPO 采样分支。
- `memory.make_prefix` / `frame_logp`：生成与策略更新使用一致候选历史。
- `decoding.beam_decode`：评测逐帧 Beam4，每帧提交最高模型分数；不是训练全视频搜索。
- `objective`：组相对优势、裁剪策略目标和参考模型约束。
- `rewards`：原生冻结解码 + R0；显式接受 13 / 52 个动作 token。
- `motion_reward`：Bridge E4 主表 R1，保留固定归一化尺度。

`pipeline.WorldPipeline` 负责原始 RGB / 动作 / 文字编码及严格权重加载；`training` 负责数据顺序、DDP、梯度累积、优化器、检查点和 RNG。未来图像仅用于 teacher-forcing SFT 或训练奖励 / 评测，不能进入预测入口。

## 最小核心示例

```bash
USE_TF=0 PYTHONPATH=src python examples/trace_search.py
```

随机小模型展示候选及记忆过程，不是论文性能实验。实际训练用 `configs/train/`；主表 / 消融协议用 `configs/recipes/`；固定样本和顺序用 `configs/cohorts/`、`configs/schedules/`。

## 检查依据

`provenance/core_sources.json` 对应原核心文件，`portable_sources.json` 对应编解码器 / reader，`native_training_sources.json` 对应原生 codec trainer。`SOURCE_MANIFEST.json` 记录本包最终文件哈希。

真实权重首帧对齐见 `provenance/native_pipeline_validation.json`；完整验证范围见 [REPRODUCTION.md](REPRODUCTION.md)。
