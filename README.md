<div align="center">

# FutureWorlds
### Learning Robotic World Models from Alternative Futures

**Construct diverse futures · Maintain individual histories · Learn from relative quality**

[GitHub repository](https://github.com/Alexander-wu/FutureWorlds) · [中文说明](README.zh-CN.md) · [Getting started](docs/GETTING_STARTED.md) · [Reproduction](docs/REPRODUCTION.md) · [Model loading](docs/LOADING.md) · [Project website](https://alexander-wu.github.io/FutureWorlds/) · [Release status](#release-status)

</div>

![FutureWorlds framework](site/assets/method.png)

FutureWorlds learns action-conditioned robotic world models from alternative predictions. Diverse beam search constructs candidate futures, bounded candidate-specific memory maintains their histories, and **MemSPO** (Memory-Conditioned Search-Guided Policy Optimization) learns from their relative trajectory rewards.

> **Release candidate.** Training, inference and evaluation workflows are implemented. Public checkpoint/data downloads and final licensing are pending. The portable pipeline passes 19 Linux CPU checks, including two-process DDP, plus real-checkpoint first-frame comparisons on three datasets. Full GPU reproduction remains to be validated.

## Highlights

- **Three robotic datasets:** RT-1, BridgeV2 and RoboCasa, with explicit action layouts and fixed evaluation cohorts.
- **Search-guided post-training:** MemSPO, GRPO and ordinary-beam controls share a unified trainer.
- **Persistent candidate memory:** generation and policy scoring use matching retained histories.
- **A reproducible workflow:** data conversion, frozen-feature caching, SFT, post-training, resume, checkpoint export, RGB prediction and metric aggregation.

## Reported results

Manuscript Table 1: **32 predicted frames, 128 trajectories per dataset**. These are the existing experiment results, not a rerun of the consolidated release pipeline.

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | LPIPS reduction vs. strongest baseline |
|:--|--:|--:|--:|--:|
| RT-1 | **22.47** | **0.8026** | **0.1551** | **14.78%** |
| BridgeV2 | **22.03** | **0.7968** | **0.1409** | **20.84%** |
| RoboCasa | **18.47** | **0.7659** | **0.1833** | **9.12%** |

[Full 13-model comparison (CSV)](site/data/results.csv) · [Machine-readable data and provenance](site/data/results.json)

![Selected qualitative comparisons with matched detail views](site/assets/qualitative.png)

Selected qualitative examples show object appearance, spatial relations and robot motion. They are not measurements of physical robot success.

## Quick start

Clone the repository, install dependencies, and configure your local assets:

```bash
git clone https://github.com/Alexander-wu/FutureWorlds.git
cd FutureWorlds
./setup.sh
cp configs/paths.example.json configs/paths.local.json
# Edit bundle, data, output, text_model and GPU count.
./run.sh --config configs/paths.local.json --suite main-table
```

**Required assets:** the paired model/codec bundle, fixed-cohort data, pinned T5-base snapshot and LPIPS-VGG weights. Downloads are not yet published; this command does not manufacture missing assets. If using a pre-existing Python environment, set `PYTHON=/path/to/python`.

The main-table command evaluates **FutureWorlds' three dataset variants**, not every external baseline. It saves 10/20/32-frame results, per-case predictions, tokens and protocol records, then writes `results.csv` and `results.md`.

| Workflow | Command suffix | Purpose |
|:--|:--|:--|
| Main-table models | `--suite main-table` | Three dataset-specific FutureWorlds variants |
| Matched comparison | `--suite matched-200` | SFT / GRPO200 / ordinary200 / MemSPO200 |
| Memory ablation | `--suite memory` | Full / recent / minimal history, fixed weights |
| Retraining | `--suite train` | Data → cache → optional SFT → post-training → evaluation |

```bash
# Inspect without starting computation.
./run.sh --config configs/paths.local.json --suite main-table --dry-run

# Matched post-training and memory ablations.
./run.sh --config configs/paths.local.json --suite matched-200
./run.sh --config configs/paths.local.json --suite memory

# Edit the training configuration before launching.
cp configs/train-workflow.example.json configs/train-workflow.local.json
./run.sh --config configs/train-workflow.local.json --suite train
```

## Checkpoints and protocols

**15 predictor variants + 3 codecs (~9.27 GB)** are prepared and checksum-verified, excluding external T5. Main variants with their codecs total approximately 3.10 GB. Weight binaries are separate from the code repository.

| Dataset | Main variant | Reward / text | Main-table evaluation |
|:--|:--|:--|:--|
| RT-1 | MemSPO 500 | R0 / text-conditioned | GPU FP32 |
| BridgeV2 | MemSPO 200, E4 | R1 / no text | CPU FP32 |
| RoboCasa | MemSPO 400 | R0 / text-conditioned | GPU FP32 |

Each dataset also has `sft`, `memspo200`, `grpo200` and `ordinary200` variants. **Main-table and matched-200 checkpoints are not interchangeable.** Training uses full-video candidate search; evaluation uses per-frame Beam4. Fixed 32-clip monitoring curves are distinct from 128-trajectory evaluation.

See [protocols](docs/PROTOCOLS.md), [model cards](model_cards/README.md), [weight inventory](weights_manifest.json) and [loading examples](docs/LOADING.md).

## Repository layout

```text
src/futureworlds/
  codec/                 Native FSQ visual codec
  codec_training/        Original Bridge / RoboCasa codec trainers
  data.py, data_readers/  Lossless data conversion and native readers
  pipeline.py            RGB + action + text → future frames
  search.py, memory.py   Diverse candidates and consistent history
  training.py            SFT / MemSPO / GRPO / ordinary-beam training
  cache.py, exporting.py Feature caching and verified model export
  evaluation.py          PSNR / SSIM / LPIPS / MAE / MSE
  workflow.py            One-command orchestration
configs/                 Cohorts, schedules, recipes and local examples
docs/                    Setup, data, protocols and release instructions
model_cards/             Checkpoint descriptions
provenance/              Source lineage and validation evidence
site/                    Static project page for GitHub Pages
tests/                   Core, pipeline, recovery and integrity checks
```

[Code tour](docs/CODE_TOUR.md) · [Reproduction guide](docs/REPRODUCTION.md) · [Data format](docs/DATA.md) · [English web guide](site/guides.html)

## Validation

[![Code checks](https://github.com/Alexander-wu/FutureWorlds/actions/workflows/checks.yml/badge.svg)](https://github.com/Alexander-wu/FutureWorlds/actions/workflows/checks.yml)

Try `python examples/trace_search.py` after installation for an asset-free CPU walkthrough. See [Getting started](docs/GETTING_STARTED.md) for troubleshooting and [Contributing](CONTRIBUTING.md) for development checks.

```bash
USE_TF=0 .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python scripts/verify_sources.py
```

- **19 Linux CPU checks pass**, covering search, RGB inference, cache equivalence, two-process SFT and recovery, post-training update paths, export and integrity.
- **Real checkpoints:** one fixed sample per dataset; all 80 generated first-frame tokens and decoded float pixels exactly match the original entry point on CPU.
- **Pending:** full 384-trajectory × 32-frame GPU evaluation, full retraining and clean Docker validation. The macOS distributed test is skipped after a rendezvous timeout.

[Linux CI run](https://github.com/Alexander-wu/FutureWorlds/actions/runs/36025779096) · [Validation record](provenance/validation.json) · [Real-checkpoint audit](provenance/native_pipeline_validation.json)

## Release status

- [x] Core algorithms and native RGB pipeline
- [x] Unified SFT / post-training / evaluation workflows
- [x] Explicit model variants, cohorts, schedules and checkpoint hashes
- [x] Model cards and GitHub Pages project site
- [x] Linux CPU distributed training / recovery regression
- [ ] Full GPU regression
- [ ] Public Hugging Face model repository and data-access instructions
- [ ] Final license for original additions, paper link and citation metadata

Paper and checkpoint links will be added when published. The external-baseline training repositories, WorldArena diagnostic scripts and manuscript source are not included in these one-command suites.

## Project website

The dependency-free site is in [`site/`](site/). Preview locally:

```bash
python3 -m http.server 8785 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8785/`. A GitHub Actions workflow deploys **only `site/`** to GitHub Pages after the repository owner enables Pages. See [deployment instructions](docs/GITHUB_PAGES.md).

## Attribution and licensing

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses/). Upstream components, pretrained assets and datasets retain their own licenses. A license for the original additions is not yet selected; preparation of this repository is not a new blanket license grant.
