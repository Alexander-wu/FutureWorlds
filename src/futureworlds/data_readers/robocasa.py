import hashlib, io, json
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def text_id(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


@lru_cache(maxsize=8)
def lowdim(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def read_frames(row, indices, view=None):
    assert view in (None, "observation.images.robot0_agentview_left")
    z = lowdim(row["arrays"])
    offset = z["offsets"]
    indices = np.asarray(indices, dtype=np.int64)
    assert indices.min() >= 0 and indices.max() < row["frames"]
    frames = []
    with open(row["images"], "rb") as f:
        for i in indices:
            start, n = offset[i]
            f.seek(int(start))
            b = f.read(int(n))
            assert len(b) == n
            with Image.open(io.BytesIO(b)) as im:
                frames.append(np.asarray(im.convert("RGB")).copy())
    return np.stack(frames)


def ordered_actions(raw, indices):
    raw = np.asarray(raw, dtype=np.float32)
    ids = np.asarray(indices, dtype=np.int64)
    assert (
        raw.ndim == 2
        and raw.shape[1] == 12
        and len(ids) >= 2
        and (np.diff(ids) == 4).all()
    )
    result = np.zeros((len(ids), 52), dtype=np.float32)
    for i in range(len(ids) - 1):
        block = raw[ids[i] : ids[i + 1]]
        assert block.shape == (4, 12)
        result[i] = np.pad(block, ((0, 0), (0, 1))).reshape(-1)
    return result


def read_images(row, length=None):
    n = length or row["window_frames"]
    assert 2 <= n <= row["window_frames"]
    start = row["start"]
    z = lowdim(row["arrays"])
    ids = np.arange(start, start + n)
    assert ids[-1] < row["frames"]
    images = read_frames(row, ids)
    acts = z["actions"][ids].copy()
    # The last row is unused: never condition on a control after the predicted sequence.
    acts[-1] = 0
    return images, acts, row["text"]


def choose_balanced(records, index, seed):
    groups = {}
    for r in records:
        groups.setdefault(r["task"], {}).setdefault(r["id"], []).append(r)
    tasks = sorted(groups)
    task = tasks[index % len(tasks)]
    rng = np.random.default_rng(seed + index * 9973)
    episodes = sorted(groups[task])
    eid = episodes[int(rng.integers(len(episodes)))]
    windows = groups[task][eid]
    return windows[int(rng.integers(len(windows)))]
