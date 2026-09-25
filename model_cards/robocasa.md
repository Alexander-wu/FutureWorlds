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
# FutureWorlds — RoboCasa

Local release candidate, not yet published. Model weights are verified by CPU SHA-256; one-frame RGB predictions on one fixed case per dataset match the original entry point exactly on CPU; full-horizon GPU regression remains pending.

## Checkpoint variants

| Folder | Method | Updates | Use |
|---|---|---:|---|
| `models/main` | MemSPO (R0) | 400 | Paper main table |
| `models/sft` | SFT | 50,000 | Initialization/control |
| `models/memspo200` | MemSPO (R0) | 200 | Matched ablation |
| `models/grpo200` | GRPO (R0) | 200 | Matched ablation |
| `models/ordinary200` | Ordinary global beam8, top4 (R0) | 200 | Proposal-strategy ablation |

All post-training checkpoints start from their paired SFT checkpoint. `main` and `memspo200` must not be interchanged.

## Main-table evaluation

128 trajectories, 32 predicted frames, 2 initial observations, per-frame Beam4, anchor1 + recent6, no true-future refresh. Recorded split: held-out test; runtime: GPU FP32.

| PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|
| 18.4719 | 0.765948 | 0.183338 |

These are recorded paper results; staging weights does not constitute a fresh reproduction.

## Required components

- Dataset-specific `codec/` and `preprocessing/` files, paired by the manifest.
- Trained text adapters in each checkpoint and frozen T5-base at revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`; T5 is referenced, not duplicated.
- 52 action tokens per predicted frame. RoboCasa uses four ordered 13-dimensional control blocks; do not flatten time in a different order.
- Original FP32 safetensors; no post-hoc quantization or checkpoint merging.

## Loading and scope

Use `futureworlds.checkpoints.load_world_model(bundle_root, "robocasa", variant="main")` for the predictor only. This does not decode RGB, prepare actions, or download T5 automatically. Use `futureworlds.pipeline.WorldPipeline` for the complete RGB/action/text pipeline; see `docs/LOADING.md`. Clean-environment full-horizon GPU equivalence remains pending. These models are custom action-conditioned predictors, not generic `pipeline("text-to-video")` models.

## Provenance and limitations

Core architecture and tokenizer derive from RLVR-World/iVideoGPT. Upstream license texts are included under `licenses/`; the license for original additions is not yet selected. The raw datasets and third-party baseline weights are not included. Learned-world feedback examples are qualitative generated trajectories, not independently verified physical task-success results.
