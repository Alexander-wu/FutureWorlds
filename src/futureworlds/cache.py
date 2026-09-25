"""Reusable frozen-codec and frozen-T5 features with explicit source fingerprints."""

import json, hashlib, os
from pathlib import Path
import numpy as np
import torch
from .checkpoints import file_sha256
from .protocol import feature_fingerprint
from .data import read_manifest, read_case, write_json


def cache_key(row, bundle_hash):
    return hashlib.sha256((row["sha256"] + bundle_hash).encode()).hexdigest()


def cache_dataset(bundle, dataset, manifest, output, device="cuda", text_model=None):
    from .pipeline import WorldPipeline

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if device.startswith("cuda"):
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    pipe = WorldPipeline(bundle, dataset, "sft", device, text_model)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    bundle_hash = feature_fingerprint(bundle, dataset, text_model)
    m = read_manifest(manifest, dataset)
    for row in m["cases"][rank::world]:
        key = cache_key(row, bundle_hash)
        target = root / (key + ".npz")
        if target.exists():
            load_cached(root, row, bundle_hash, device)
            continue
        images, actions, text = read_case(manifest, row, pipe.action_dim)
        c, d, a, _, condition = pipe.encode_training_window(images, actions, text)
        values = dict(
            context=c.cpu().numpy(),
            dynamics=d.cpu().numpy(),
            actions=a.cpu().numpy(),
            source_sha256=np.array(row["sha256"]),
            bundle_sha256=np.array(bundle_hash),
        )
        values.update({k: v.cpu().numpy() for k, v in condition.items()})
        tmp = target.with_suffix(f".rank{rank}.partial")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **values)
        tmp.replace(target)
        write_json(target.with_suffix(".json"), dict(sha256=file_sha256(target)))
    if world > 1:
        import torch.distributed as dist

        dist.init_process_group("gloo")
        dist.barrier()
    if rank == 0:
        write_json(
            root / "complete.json",
            dict(
                manifest_sha256=file_sha256(manifest),
                bundle_sha256=bundle_hash,
                cases=len(m["cases"]),
            ),
        )


def load_cached(root, row, bundle_hash, device):
    p = Path(root) / (cache_key(row, bundle_hash) + ".npz")
    if file_sha256(p) != json.loads(p.with_suffix(".json").read_text())["sha256"]:
        raise ValueError("Corrupt token cache")
    with np.load(p, allow_pickle=False) as z:
        if (
            z["source_sha256"].item() != row["sha256"]
            or z["bundle_sha256"].item() != bundle_hash
        ):
            raise ValueError("Stale token cache")
        condition = {
            k: torch.from_numpy(z[k].copy()).to(device)
            for k in ["text_features", "text_mask"]
            if k in z
        }
        return (
            *[
                torch.from_numpy(z[k].copy()).to(device)
                for k in ["context", "dynamics", "actions"]
            ],
            condition,
        )
