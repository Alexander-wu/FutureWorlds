# Development and reproducibility

Start with [Getting started](docs/GETTING_STARTED.md) and the [code tour](docs/CODE_TOUR.md). Work from a source checkout with an editable installation.

Before submitting a change:

```bash
USE_TF=0 USE_FLAX=0 python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/verify_sources.py
```

GitHub Actions runs the checks on Linux CPU, including two-process SFT, recovery and checkpoint export. It does not measure paper quality or GPU performance.

Changes to search, candidate memory, causal indexing, rewards or training must preserve generation/scoring history consistency and avoid future-observation leakage. Add a focused regression test for behavior changes. Keep original-source notices in derived modules.

Keep `configs/*.local.json`, credentials, dataset files and model binaries out of Git. Example configs use placeholder paths. Publish weights separately with checksum manifests and applicable upstream terms.

`SOURCE_MANIFEST.json` is the release file inventory. After an intentional edit, update the changed entries (SHA-256 and byte size); explicitly add new release files. Verify the inventory before committing. Do not change manuscript metrics to match a synthetic test or a new protocol. Record any new evaluation's weights, cohort, settings and output separately.

For bug reports, include the commit, command, package versions (`python -m futureworlds doctor`), dataset/variant, and a short sanitized traceback. Do not attach private paths, tokens, or unpublished data.

A license for original additions is pending. See [third-party notices](THIRD_PARTY_NOTICES.md); publication does not replace upstream licensing terms.
