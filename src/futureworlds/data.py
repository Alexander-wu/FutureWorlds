"""Portable, lossless windows. Actions[t] controls frame t -> t+1."""

import json
from pathlib import Path
import numpy as np
from .checkpoints import safe_path, file_sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def read_manifest(path, dataset=None):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or not value.get("cases"):
        raise ValueError("Expected a nonempty version-1 window manifest")
    if dataset is not None and value["dataset"] != dataset:
        raise ValueError("Dataset mismatch")
    ids = [r["id"] for r in value["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")
    for row in value["cases"]:
        safe_path(path.parent, row["file"])
        if not row.get("sha256") or not row.get("episode_id"):
            raise ValueError("Each case requires file hash and episode_id")
    return value


def read_case(manifest_path, row, action_dim, horizon=None):
    file = safe_path(Path(manifest_path).parent, row["file"])
    if file_sha256(file) != row["sha256"]:
        raise ValueError(f"Input checksum mismatch: {row['id']}")
    with np.load(file, allow_pickle=False) as z:
        images, actions = z["images"].copy(), z["actions"].copy()
        text = str(z["instruction"].item())
    if (
        images.dtype != np.uint8
        or images.ndim != 4
        or images.shape[1:] != (256, 320, 3)
    ):
        raise ValueError("Expected uint8 RGB [T,256,320,3]; no implicit resizing")
    if actions.shape != (len(images), action_dim) or not np.isfinite(actions).all():
        raise ValueError("Invalid action shape or nonfinite controls")
    if horizon is not None:
        if horizon < 1 or len(images) < horizon + 2:
            raise ValueError("Insufficient frames for requested horizon")
        images, actions = images[: horizon + 2], actions[: horizon + 2]
    return images, actions.astype(np.float32), text


def disjoint(train, evaluation):
    if {r["episode_id"] for r in train["cases"]} & {
        r["episode_id"] for r in evaluation["cases"]
    }:
        raise ValueError("Training/evaluation episodes overlap")


def convert_manifest(dataset, source, output, ids=None, path_map=None, limit=None):
    """Convert an existing audited raw-window manifest, preserving order/splits.

    path_map maps old path prefixes to locally downloaded data roots. No random
    split regeneration, image interpolation, action reduction, or time shift.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]

    def key(r):
        return r.get("window_id", Path(r.get("output", "")).stem)

    if ids is not None:
        index = {key(r): r for r in rows}
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate requested IDs")
        rows = [index[i] for i in ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty input")
    output.mkdir(parents=True)
    cases = []
    for original in rows:
        r = dict(original)
        for k in ("path", "images", "arrays"):
            if k not in r:
                continue
            for old, new in sorted((path_map or {}).items(), key=lambda x: -len(x[0])):
                if r[k] == old or r[k].startswith(old.rstrip("/") + "/"):
                    r[k] = str(Path(new) / Path(r[k]).relative_to(old))
                    break
        if dataset == "rt1":
            from .data_readers.rt1 import read_images

            images, actions, text = read_images(r, length=r["window_frames"])
        elif dataset == "robocasa":
            from .data_readers.robocasa import read_images

            for k, hk in [("arrays", "arrays_sha256"), ("images", "image_sha256")]:
                if hk in r and file_sha256(r[k]) != r[hk]:
                    raise ValueError(f"Corrupt {k}")
            images, actions, text = read_images(r)
        elif dataset == "bridge":
            if file_sha256(r["path"]) != r["sha256"]:
                raise ValueError("Corrupt Bridge input")
            with np.load(r["path"], allow_pickle=False) as z:
                images, actions = z["image"].copy(), z["action"].copy()
            text = r.get("instruction", "")
        else:
            raise ValueError(dataset)
        cid = key(r)
        # Use numeric filenames: never interpolate an untrusted episode ID into paths.
        target = output / f"{len(cases):06d}.npz"
        np.savez_compressed(
            target, images=images, actions=actions, instruction=np.array(text)
        )
        cases.append(
            dict(
                id=cid,
                episode_id=r.get("id", cid),
                file=target.name,
                sha256=file_sha256(target),
                split=r["split"],
                frames=len(images),
                source_sha256=r["sha256"],
                start=r.get("start", 0),
                task=r.get("task", ""),
            )
        )
    manifest = dict(
        schema_version=1,
        dataset=dataset,
        source_manifest_sha256=file_sha256(source),
        cases=cases,
        action_alignment="actions[t]: frame t -> frame t+1",
        subset=limit is not None,
    )
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"
