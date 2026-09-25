---
tags:
- futureworlds
- robotics
- world-model
- video-prediction
---
# FutureWorlds research bundle

Private staging candidate. Contains all three datasets’ paired predictors, visual codecs, text adapters, action ranges and explicit checkpoint variants. See each dataset README and `weights_manifest.json`. Weights are unmodified FP32 safetensors, with original byte-level SHA-256.

`main` identifies the paper main-table checkpoint. `memspo200`, `grpo200`, `ordinary200` and `sft` identify matched comparison variants. Bridge `main` uses R1; Bridge `memspo200` uses R0.

Install the predictor code with `pip install ./code`. Load it with `futureworlds.checkpoints.load_world_model(bundle_root, dataset, variant="main")`. This low-level loader handles the predictor and trained text adapters. The full `WorldPipeline` and `run.sh` workflows now provide RGB inference, SFT/post-training and evaluation; see the code README. Public asset download configuration and full GPU regression are still pending.

Frozen T5-base is an external pinned dependency for RT-1 and RoboCasa. Raw training data, cached dataset features, optimizer states, account credentials and other researchers’ pretrained baseline weights are excluded.

Before a public release: finish the clean-environment original-checkpoint equivalence run and complete the license/model-card metadata. No public repository URL or paper citation identifier is fabricated in this candidate.
