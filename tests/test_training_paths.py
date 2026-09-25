"""Exercise all post-training entry paths; synthetic metric is test-only."""

import tempfile, unittest, json, copy
from pathlib import Path
from unittest.mock import patch
import torch
import test_pipeline
from futureworlds.data import write_json
from futureworlds.training import train


class SquaredDistance(torch.nn.Module):
    def forward(self, a, b):
        return (a - b).square().mean((1, 2, 3), keepdim=True)


class TrainingPathChecks(unittest.TestCase):
    def test_three_posttraining_algorithms_save_updated_parameters(self):
        from safetensors.torch import load_file

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            helper = test_pipeline.PortablePipelineChecks()
            helper.make_bundle(root / "bundle")
            helper.make_data(root / "data")
            before = load_file(
                str(root / "bundle/bridge/models/main/model.safetensors")
            )
            for method in ["memspo", "ordinary", "grpo"]:
                cfg = dict(
                    method=method,
                    dataset="bridge",
                    bundle=str(root / "bundle"),
                    train_manifest=str(root / "data/manifest.json"),
                    output=str(root / method),
                    device="cpu",
                    seed=23,
                    global_batch=1,
                    steps=1,
                    lr=1e-6,
                    horizon=1,
                    reward="R0",
                    clip=0.2,
                    kl_coef=0.001,
                    checkpoint_every=1,
                )
                write_json(root / "train.json", cfg)
                with patch("lpips.LPIPS", return_value=SquaredDistance()):
                    train(root / "train.json")
                after = load_file(
                    str(root / method / "checkpoints/step_000001/model.safetensors")
                )
                self.assertTrue(
                    any(not torch.equal(before[k], after[k]) for k in before), method
                )
                self.assertEqual(
                    json.loads((root / method / "complete.json").read_text())["state"],
                    "COMPLETE",
                )


if __name__ == "__main__":
    unittest.main()
