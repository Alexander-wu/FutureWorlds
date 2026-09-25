# Getting started

## Clone and install

```bash
# Extract FutureWorlds-code.zip, then enter its directory
cd FutureWorlds
./setup.sh
source .venv/bin/activate
python -m futureworlds doctor
```

Linux with a compatible NVIDIA driver is the training target. The default installer creates `.venv` and installs the full Python dependencies. For an existing CUDA environment, use `python -m pip install -e '.[full]'` and set `PYTHON` to that interpreter when calling `run.sh`. Choose the PyTorch build compatible with your host before installing the package.

`requirements-linux.txt` records the audited native environment; it is not a cross-platform lockfile. The Dockerfile is a starting point and has not yet passed a clean-build test.

## Try the code without weights or a GPU

```bash
python examples/trace_search.py
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/verify_sources.py
```

The example uses a tiny random model to trace candidate search and retained histories. Integration tests create temporary synthetic data and tiny models, exercise the native codec, training updates, recovery, and export, and do not download research checkpoints. They are code checks, not paper results. The two-process distributed check runs on Linux and is skipped on macOS.

## Evaluate real checkpoints

Provide these assets first:

| Asset | Where to find the specification |
|---|---|
| Predictor, codec, action ranges, and optional text adapter | [Checkpoint inventory](../weights_manifest.json), [model cards](../model_cards/README.md) |
| RT-1, BridgeV2, and RoboCasa evaluation data | [Data preparation](DATA.md), `configs/cohorts/` |
| Pinned T5 snapshot for RT-1 / RoboCasa | [Reproduction guide](REPRODUCTION.md) |
| LPIPS / VGG metric weights | Standard LPIPS / torchvision cache |

Public FutureWorlds weight downloads are pending. Configurations use explicit local paths and will not substitute random weights or alternate samples when an asset is missing.

```bash
cp configs/paths.example.json configs/paths.local.json
# Edit the local file to point to your assets.
./run.sh --config configs/paths.local.json --suite main-table --dry-run
./run.sh --config configs/paths.local.json --suite main-table
```

Set `limit` to `1` in the local configuration for a first-sample check. Remove it for the fixed 128-sample cohort. Set `gpus` to the number of evaluation workers. Bridge main-table evaluation preserves CPU FP32; the matched-200 and memory suites use their separate device protocols.

| Suite | Output / purpose |
|---|---|
| `main-table` | FutureWorlds on three datasets at 10 / 20 / 32 frames |
| `matched-200` | SFT, GRPO200, ordinary-beam200, MemSPO200 |
| `memory` | Full, recent, minimal history using fixed weights |
| `train` | Prepare data, cache features, train, export, and evaluate |

Each completed evaluation writes predictions, metrics, and protocol fingerprints. The suite writes `results.csv` and `results.md`. Resume by rerunning the same command; a conflicting configuration is rejected. These suites do not retrain the external baseline repositories.

## Train or fine-tune

```bash
cp configs/train-workflow.example.json configs/train-workflow.local.json
# Edit paths, dataset, method, GPU count, and initial checkpoint if required.
./run.sh --config configs/train-workflow.local.json --suite train --dry-run
./run.sh --config configs/train-workflow.local.json --suite train
```

`post-only` starts from an SFT variant. `sft+post` requires a separate verified initialization checkpoint and runs SFT before post-training. Select `memspo`, `grpo`, or `ordinary`; do not mix main-table checkpoints with matched-200 variants. See [protocols](PROTOCOLS.md) and the [training guide](REPRODUCTION.md) for multi-node launch, accumulation, checkpoint recovery, and codec training.

## Common problems

- **Missing asset:** replace `/path/to/...` placeholders; check the selected variant exists in the bundle and run `python -m futureworlds doctor --bundle /your/bundle`.
- **Different metrics:** compare dataset IDs, checkpoint hashes, horizon, decoding settings, memory, device and precision against the saved protocol. Main-table and matched-200 settings are intentionally distinct.
- **CUDA out of memory:** first use a single sample to check the pipeline. Do not silently alter beam width or memory when claiming a matched paper result.
- **Resume rejected:** restore the same configuration and world size or start a new output directory.
- **Offline metric initialization fails:** prepare the LPIPS / VGG cache in advance. CI uses synthetic perceptual metrics only for training-path checks.

The homepage shows recorded manuscript results. Full GPU reproduction of the consolidated entry point, public asset downloads, and a license for original additions remain release tasks; upstream licenses and notices are preserved.
