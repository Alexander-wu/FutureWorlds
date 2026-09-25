"""Load an explicit released world-model variant; never substitute adapters."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts or "\\" in relative:
        raise ValueError(f"Invalid relative path: {relative!r}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path escapes bundle root")
    return path


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty manifest")
    seen = set()
    for item in manifest["files"]:
        name = item["path"]
        if name in seen:
            raise ValueError(f"Duplicate file: {name}")
        seen.add(name)
        path = safe_path(root, name)
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing file or wrong size: {name}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    return manifest


def load_world_model(root, dataset, variant="main", device="cpu", verify=True):
    """Load the predictor only. Codec/actions/T5 preprocessing are separate.

    FP32 and SDPA match the audited loaders. Dataset/variant selection is
    explicit; missing text adapters are an error, never silently zero-filled.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    from .text_conditioning import TextWorldModel

    root = Path(root)
    if verify:
        verify_bundle(root)
    settings = json.loads(
        safe_path(root, f"{dataset}/futureworlds_config.json").read_text()
    )
    if settings["dataset"] != dataset or variant not in settings["variants"]:
        raise ValueError("Dataset or variant does not match bundle configuration")
    entry = settings["variants"][variant]
    folder = safe_path(root, entry["path"])
    cfg = AutoConfig.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False
    )
    core = AutoModelForCausalLM.from_config(
        cfg,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    core.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
    if settings["text_conditioning"]["enabled"]:
        tc = settings["text_conditioning"]
        model = TextWorldModel(
            core, text_width=tc["width"], layers=tc["layers"], inner=tc["inner"]
        )
        adapters = folder / "text_adapters.safetensors"
        if not adapters.is_file():
            raise FileNotFoundError(
                f"Required trained text adapters missing: {adapters}"
            )
        model.adapters.load_state_dict(load_file(str(adapters)), strict=True)
    else:
        model = core
    return model.to(device).eval()
