# Third-party provenance

This repository consolidates the FutureWorlds research code. No blanket relicensing of dependencies or original additions is implied. The author has not yet selected a release license for original additions.

- `objective.py` and `reward_core.py` preserve ByteDance / Hugging Face Apache-2.0 headers from the audited RLVR-World / verl-derived implementation. Upstream RLVR-World reference: `e1b8b6f40ca0696919ce9b8dbe965c4153bcf5f1`.
- `codec/` includes the native iVideoGPT / diffusers-derived implementation and finite scalar quantization code, retaining upstream notices. Full license texts available from the audited checkout are copied into `licenses/` (iVideoGPT, diffusers, FSQ, Paella and taming-transformers).
- LPIPS, PyTorch, Transformers, piqa and other runtime dependencies retain their respective licenses. LPIPS's license is also provided for reference.
- `licenses/Apache-2.0.txt` is a license text supporting upstream notices, not an asserted new license grant over this whole repository.
- Model weights, T5 assets and datasets are separate downloads subject to their own terms. Dataset redistribution is not assumed by including a reader.

File-level provenance is in `provenance/`. Changes include portable imports, explicit paths and configs, the unified training / inference pipeline, 52-action reward support, and an optional reward-image saving helper. Checkpoint content is not altered by code formatting or packaging.
