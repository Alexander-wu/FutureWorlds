"""Read official TF Example/RLDS records without a TensorFlow runtime.

Schema follows tensorflow/core/example/{feature,example}.proto (Apache-2.0).
The record bytes are never rewritten; indexed offsets retain original provenance.
"""

import hashlib, io, json, struct, unicodedata
from pathlib import Path
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from PIL import Image


def example_class():
    f = descriptor_pb2.FileDescriptorProto(
        name="rt1_example.proto", package="tensorflow", syntax="proto3"
    )

    def msg(name):
        return f.message_type.add(name=name)

    def field(m, name, n, t, label=1, type_name=None, oneof=None):
        x = m.field.add(name=name, number=n, type=t, label=label)
        if type_name:
            x.type_name = ".tensorflow." + type_name
        if oneof is not None:
            x.oneof_index = oneof

    for name, t in [("BytesList", 12), ("FloatList", 2), ("Int64List", 3)]:
        field(msg(name), "value", 1, t, 3)
    m = msg("Feature")
    m.oneof_decl.add(name="kind")
    for n, name in enumerate(["BytesList", "FloatList", "Int64List"], 1):
        field(
            m,
            ["bytes_list", "float_list", "int64_list"][n - 1],
            n,
            11,
            type_name=name,
            oneof=0,
        )
    m = msg("Features")
    entry = m.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    field(entry, "key", 1, 9)
    field(entry, "value", 2, 11, type_name="Feature")
    field(m, "feature", 1, 11, 3, "Features.FeatureEntry")
    field(msg("Example"), "features", 1, 11, type_name="Features")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(f)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("tensorflow.Example")
    )


Example = example_class()
ACTION_FIELDS = [
    ("base_displacement_vector", 2),
    ("base_displacement_vertical_rotation", 1),
    ("gripper_closedness_action", 1),
    ("rotation_delta", 3),
    ("terminate_episode", 3),
    ("world_vector", 3),
]
assert [x[0] for x in ACTION_FIELDS] == sorted(x[0] for x in ACTION_FIELDS)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_id(text):
    return hashlib.sha256(text.encode()).hexdigest()


def unpack(record):
    e = Example()
    e.ParseFromString(record)
    d = {}
    for k, v in e.features.feature.items():
        which = v.WhichOneof("kind")
        d[k] = list(getattr(v, which).value)
    return d


def parse(record):
    d = unpack(record)
    jpeg = d["steps/observation/image"]
    n = len(jpeg)
    acts = np.concatenate(
        [
            np.asarray(d["steps/action/" + name], np.float32).reshape(n, dim)
            for name, dim in ACTION_FIELDS
        ],
        axis=1,
    )
    text = [
        normalize(x.decode("utf-8"))
        for x in d["steps/observation/natural_language_instruction"]
    ]
    assert acts.shape == (n, 13) and np.isfinite(acts).all() and len(text) == n
    assert all(x == text[0] for x in text), (
        "Instruction changes within episode: implement segmented conditions before using this record"
    )
    first = d["steps/is_first"]
    last = d["steps/is_last"]
    # Official RT1 often omits the first-step flag; TFRecord is still one episode.
    assert len(first) == len(last) == n and last[-1] and sum(last) == 1
    assert sum(first) == 0 or (sum(first) == 1 and first[0])
    return jpeg, acts, text[0]


def read_record(row, verify=True):
    with open(row["path"], "rb") as f:
        f.seek(row["offset"])
        raw = f.read(row["length"])
    assert len(raw) == row["length"]
    if verify:
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    return raw


def read_images(row, start=None, length=None):
    jpeg, acts, text = parse(read_record(row))
    start = row.get("start", 0) if start is None else start
    length = min(34, len(jpeg) - start) if length is None else length
    images = []
    for b in jpeg[start : start + length]:
        x = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
        assert x.shape == (256, 320, 3)
        images.append(x)
    images = np.stack(images)
    assert len(images) == length
    return images, acts[start : start + length], text


def records(path):
    with open(path, "rb") as f:
        index = 0
        while True:
            raw = f.read(8)
            if not raw:
                break
            assert len(raw) == 8
            size = struct.unpack("<Q", raw)[0]
            assert 0 < size < 256 * 1024**2
            assert len(f.read(4)) == 4
            offset = f.tell()
            record = f.read(size)
            assert len(record) == size and len(f.read(4)) == 4
            yield index, offset, record
            index += 1


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def atomic(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def jsonlines(path, values):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values))
    tmp.replace(p)
