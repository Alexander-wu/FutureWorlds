"""Exercise two-process CPU SFT, checkpoint export and workflow command planning."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
import test_pipeline
from futureworlds.data import write_json
from futureworlds.exporting import export_model
from futureworlds.checkpoints import verify_bundle, load_world_model
from futureworlds.workflow import run


class WorkflowChecks(unittest.TestCase):
    @unittest.skipIf(
        sys.platform == "darwin",
        "Distributed regression targets the Linux training host; local macOS rendezvous timed out",
    )
    def test_two_process_cpu_sft_and_repeatable_export(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            helper = test_pipeline.PortablePipelineChecks()
            helper.make_bundle(root / "bundle")
            helper.make_data(root / "data")
            cfg = dict(
                method="sft",
                dataset="bridge",
                bundle=str(root / "bundle"),
                train_manifest=str(root / "data/manifest.json"),
                output=str(root / "train"),
                device="cpu",
                seed=7,
                global_batch=2,
                steps=1,
                lr=1e-5,
                warmup_steps=1,
                checkpoint_every=1,
            )
            write_json(root / "config.json", cfg)
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=127.0.0.1:0",
                "--rdzv_id=futureworlds-test",
                "--nproc_per_node=2",
                "-m",
                "futureworlds",
                "train",
                "--config",
                str(root / "config.json"),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=dict(
                    os.environ, CUDA_VISIBLE_DEVICES="", USE_TF="0", OMP_NUM_THREADS="1"
                ),
                timeout=180,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout[-1500:] + result.stderr[-5000:]
            )
            checkpoint = root / "train/checkpoints/step_000001"
            export_model(root / "bundle", "bridge", checkpoint, root / "export")
            export_model(root / "bundle", "bridge", checkpoint, root / "export")
            self.assertGreater(len(verify_bundle(root / "export")["files"]), 3)
            load_world_model(root / "export", "bridge", "sft", "cpu")
            cfg["resume"] = str(checkpoint)
            write_json(root / "config.json", cfg)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=dict(
                    os.environ, CUDA_VISIBLE_DEVICES="", USE_TF="0", OMP_NUM_THREADS="1"
                ),
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-5000:])

    def test_checkpoint_export_is_verified_and_repeatable(self):
        from futureworlds.training import train

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            helper = test_pipeline.PortablePipelineChecks()
            helper.make_bundle(root / "bundle")
            helper.make_data(root / "data")
            cfg = dict(
                method="sft",
                dataset="bridge",
                bundle=str(root / "bundle"),
                train_manifest=str(root / "data/manifest.json"),
                output=str(root / "run"),
                device="cpu",
                seed=7,
                global_batch=1,
                steps=1,
                lr=1e-5,
                warmup_steps=1,
                checkpoint_every=1,
            )
            write_json(root / "config.json", cfg)
            train(root / "config.json")
            checkpoint = root / "run/checkpoints/step_000001"
            export_model(root / "bundle", "bridge", checkpoint, root / "export")
            export_model(root / "bundle", "bridge", checkpoint, root / "export")
            self.assertGreater(len(verify_bundle(root / "export")["files"]), 3)
            load_world_model(root / "export", "bridge", "sft", "cpu")
            (checkpoint / "model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                export_model(
                    root / "bundle", "bridge", checkpoint, root / "other_export"
                )

    def test_main_table_plan_preserves_bridge_cpu_and_paper_cohorts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "paths.json"
            write_json(
                p,
                dict(
                    bundle="/weights",
                    data="/data",
                    output="/results",
                    gpus=8,
                    sources={
                        k: {"manifest": "/raw/" + k + ".jsonl"}
                        for k in ["rt1", "bridge", "robocasa"]
                    },
                ),
            )
            stream = StringIO()
            with redirect_stdout(stream):
                run(p, "main-table", True)
            text = stream.getvalue()
            start = text.index("{")
            commands = json.loads(text[start:])["commands"]
            self.assertEqual(len(commands), 3)
            bridge = next(
                c for c in commands if c[c.index("--dataset") + 1] == "bridge"
            )
            self.assertEqual(bridge[bridge.index("--device") + 1], "cpu")
            self.assertNotIn("--nproc_per_node=8", bridge)
            self.assertTrue(all("--ids" in c for c in commands))


if __name__ == "__main__":
    unittest.main()
