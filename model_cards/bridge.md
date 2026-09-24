---
language:
- en
tags:
- futureworlds
- world-model
- robotics
- video-prediction
- reinforcement-learning
---
# FutureWorlds — BridgeV2

Local release candidate, not yet published. Model weights are verified by CPU SHA-256; one-frame RGB predictions on one fixed case per dataset match the original entry point exactly on CPU; full-horizon GPU regression remains pending.

## Checkpoint variants

| Folder | Method | Updates | Use |
|---|---|---:|---|
| `models/main` | MemSPO (R1 / E4) | 200 | Paper main table |
| `models/sft` | SFT | 50,000 | Initialization/control |
| `models/memspo200` | MemSPO (R0) | 200 | Matched ablation |
| `models/grpo200` | GRPO (R0) | 200 | Matched ablation |
| `models/ordinary200` | Ordinary global beam8, top4 (R0) | 200 | Proposal-strategy ablation |

All post-training checkpoints start from their paired SFT checkpoint. `main` and `memspo200` must not be interchanged even though both use 200 updates: Bridge main uses R1, while the matched ablation uses R0.

## Main-table evaluation

128 trajectories, 32 predicted frames, 2 initial observations, per-frame Beam4, anchor1 + recent6, no true-future refresh. Recorded split: held-out development; runtime: CPU FP32.

| PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|
| 22.0276 | 0.796801 | 0.140858 |

These are recorded paper results; staging weights does not constitute a fresh reproduction.

## Required components

- Dataset-specific `codec/` and `preprocessing/` files, paired by the manifest.
- This released Bridge branch does not use text conditioning.
- 13 action tokens per predicted frame. Use the dataset adapter and recorded action ranges, not another dataset’s statistics.
- Original FP32 safetensors; no post-hoc quantization or checkpoint merging.

## Loading and scope

Use `futureworlds.checkpoints.load_world_model(bundle_root, "bridge", variant="main")` for the predictor only. This does not decode RGB, prepare actions, or download T5 automatically. Use `futureworlds.pipeline.WorldPipeline` for the complete RGB/action/text pipeline; see `docs/LOADING.md`. Clean-environment full-horizon GPU equivalence remains pending. These models are custom action-conditioned predictors, not generic `pipeline("text-to-video")` models.

## Provenance and limitations

Core architecture and tokenizer derive from RLVR-World/iVideoGPT. Upstream license texts are included under `licenses/`; the license for original additions is not yet selected. The raw datasets and third-party baseline weights are not included. Learned-world feedback examples are qualitative generated trajectories, not independently verified physical task-success results.
