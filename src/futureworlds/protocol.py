"""Bind runs to executable code, runtime versions and external text assets."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from .checkpoints import file_sha256


def implementation():
    root = Path(__file__).parent
    files = {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*.py"))
    }
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "numpy",
        "piqa",
        "lpips",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return dict(
        source_sha256=hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        versions=versions,
    )


def text_identity(bundle, dataset, text_model):
    settings = json.loads(
        (Path(bundle) / dataset / "futureworlds_config.json").read_text()
    )["text_conditioning"]
    if not settings["enabled"]:
        return None
    if text_model is None:
        return dict(repo_id=settings["repo_id"], revision=settings["revision"])
    root = Path(text_model)
    files = {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and ".cache" not in p.parts
        and p.suffix in (".json", ".model", ".bin", ".safetensors")
    }
    if not files:
        raise ValueError("Local text snapshot has no model/tokenizer assets")
    return dict(files=files)


def feature_fingerprint(bundle, dataset, text_model):
    value = dict(
        bundle=file_sha256(Path(bundle) / "manifest.json"),
        text=text_identity(bundle, dataset, text_model),
        implementation=implementation(),
    )
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
