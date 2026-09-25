"""Validate every packaged source/config/doc checksum without GPU dependencies."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
for row in manifest["files"]:
    path = (root / row["path"]).resolve()
    if root not in path.parents:
        raise SystemExit("Unsafe source manifest path")
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
    ):
        raise SystemExit(f"Changed or missing packaged file: {row['path']}")
print(f"PASS: {len(manifest['files'])} packaged file hashes verified.")
