import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from futureworlds.checkpoints import file_sha256, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_release", ROOT / "scripts/prepare_hf_release.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseToolsChecks(unittest.TestCase):
    def test_allowlisted_staging_and_offline_upload_dry_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "weights.bin"
            source.write_bytes(b"test-weight-bytes")
            plan = {
                "files": [
                    {
                        "source": str(source),
                        "destination": "test/weights.bin",
                        "bytes": source.stat().st_size,
                        "sha256": file_sha256(source),
                    },
                    {"destination": "README.md", "content": "Test bundle\n"},
                ]
            }
            output = root / "stage"
            module.prepare(plan, output)
            self.assertFalse(output.exists())
            module.prepare(plan, output, materialize=True)
            self.assertEqual(len(verify_bundle(output)["files"]), 2)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/upload_hf_release.py"),
                    "--folder",
                    str(output),
                    "--repo-id",
                    "test/unused",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertFalse(json.loads(result.stdout)["upload"])
            (output / "unlisted.txt").write_text("not in manifest")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/upload_hf_release.py"),
                    "--folder",
                    str(output),
                    "--repo-id",
                    "test/unused",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_staging_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "weights.bin"
            source.write_bytes(b"abc")
            plan = {
                "files": [
                    {
                        "source": str(source),
                        "destination": "weights.bin",
                        "bytes": 3,
                        "sha256": "0" * 64,
                    }
                ]
            }
            with self.assertRaises(ValueError):
                module.prepare(plan, root / "out", materialize=True)
            self.assertFalse((root / "out/manifest.json").exists())
