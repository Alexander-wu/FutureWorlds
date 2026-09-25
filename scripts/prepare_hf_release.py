"""Stage only allowlisted release files. Default: inspect, never copy/upload."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from futureworlds.checkpoints import file_sha256, safe_path


def prepare(plan, output, materialize=False):
    output = Path(output)
    if output.exists():
        raise ValueError(
            "Choose a new output directory; existing releases are immutable"
        )
    seen = set()
    for item in plan["files"]:
        name = item["destination"]
        safe_path(output, name)
        if name in seen or name == "manifest.json":
            raise ValueError(f"Duplicate/reserved path: {name}")
        seen.add(name)
        if "source" in item:
            source = Path(item["source"])
            if not source.is_file() or source.stat().st_size != item["bytes"]:
                raise ValueError(f"Missing source or size mismatch: {name}")
    summary = {
        "files": len(seen),
        "source_bytes": sum(x.get("bytes", 0) for x in plan["files"]),
        "materialized": materialize,
    }
    if not materialize:
        return summary
    output.mkdir(parents=True)
    records = []
    try:
        for item in plan["files"]:
            target = safe_path(output, item["destination"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if "source" in item:
                shutil.copyfile(item["source"], target)
                if file_sha256(target) != item["sha256"]:
                    raise ValueError(
                        f"Copied source hash mismatch: {item['destination']}"
                    )
            else:
                target.write_text(item["content"], encoding="utf-8")
            records.append(
                {
                    "path": item["destination"],
                    "bytes": target.stat().st_size,
                    "sha256": file_sha256(target),
                }
            )
        (output / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": records}, indent=2) + "\n"
        )
    except Exception:
        # No manifest is written for an incomplete release. Keep partial files for diagnosis.
        raise
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                json.loads(Path(args.plan).read_text()), args.output, args.materialize
            ),
            indent=2,
        )
    )
