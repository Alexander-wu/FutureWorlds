"""Build a slim inference bundle from a newly trained checkpoint."""

import json, os, shutil
from pathlib import Path
from .checkpoints import verify_bundle, safe_path, file_sha256
from .data import write_json


def export_model(bundle, dataset, checkpoint, output):
    bundle = Path(bundle)
    checkpoint = Path(checkpoint)
    output = Path(output)
    verify_bundle(bundle)
    if output.exists():
        previous = verify_bundle(output)
        if previous.get("training_checkpoint_sha256") != file_sha256(
            checkpoint / "experiment.json"
        ):
            raise ValueError("Existing export belongs to another checkpoint")
        if previous.get("source_bundle_sha256") != file_sha256(
            bundle / "manifest.json"
        ):
            raise ValueError("Existing export uses another source bundle")
        return
    receipt = json.loads((checkpoint / "experiment.json").read_text())
    for name, digest in receipt["files"].items():
        if file_sha256(safe_path(checkpoint, name)) != digest:
            raise ValueError("Training checkpoint checksum mismatch")
    config = json.loads(
        safe_path(bundle, f"{dataset}/futureworlds_config.json").read_text()
    )
    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError("Predictor checkpoint missing")
    if (
        config["text_conditioning"]["enabled"]
        and not (checkpoint / "text_adapters.safetensors").exists()
    ):
        raise FileNotFoundError("Trained text adapters required")
    output.mkdir(parents=True)

    def copy(src, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Copies avoid later changes to the new run mutating released weights.
        shutil.copyfile(src, dest)

    for name in (
        "config.json",
        "model.safetensors",
        "text_adapters.safetensors",
        "generation_config.json",
    ):
        if (checkpoint / name).exists():
            copy(checkpoint / name, output / dataset / "models/main" / name)
    codec = safe_path(bundle, config["codec"])
    for f in codec.iterdir():
        if f.is_file():
            copy(f, output / dataset / "codec" / f.name)
    copy(
        safe_path(bundle, config["action_ranges"]),
        output / dataset / "preprocessing/action_ranges.pt",
    )
    trained = (
        json.loads((checkpoint / "experiment.json").read_text())
        if (checkpoint / "experiment.json").exists()
        else {}
    )
    names = (
        ["main", "sft"]
        if trained.get("config", {}).get("method") == "sft"
        else ["main"]
    )
    config.update(
        codec=f"{dataset}/codec",
        action_ranges=f"{dataset}/preprocessing/action_ranges.pt",
        status="locally_trained",
        variants={
            k: dict(path=f"{dataset}/models/main", purpose="locally_trained")
            for k in names
        },
    )
    write_json(output / dataset / "futureworlds_config.json", config)
    files = [
        dict(
            path=p.relative_to(output).as_posix(),
            bytes=p.stat().st_size,
            sha256=file_sha256(p),
        )
        for p in output.rglob("*")
        if p.is_file()
    ]
    write_json(
        output / "manifest.json",
        dict(
            schema_version=1,
            files=files,
            training_checkpoint_sha256=file_sha256(checkpoint / "experiment.json"),
            source_bundle_sha256=file_sha256(bundle / "manifest.json"),
        ),
    )
    verify_bundle(output)
