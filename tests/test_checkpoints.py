import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from futureworlds.checkpoints import (
    file_sha256,
    load_world_model,
    safe_path,
    verify_bundle,
)
from futureworlds.text_conditioning import TextWorldModel


class BundleChecks(unittest.TestCase):
    def make_bundle(self, root, text):
        torch.manual_seed(9)
        core = (
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=30,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    attention_dropout=0.0,
                    _attn_implementation="sdpa",
                )
            )
            .float()
            .eval()
        )
        model = (
            TextWorldModel(core, text_width=24, layers=[0], inner=16) if text else core
        )
        folder = root / "test/models/main"
        folder.mkdir(parents=True)
        core.save_pretrained(folder)
        if text:
            torch.nn.init.normal_(model.adapters["0"].out.weight, std=0.03)
            save_file(
                model.adapters.state_dict(), str(folder / "text_adapters.safetensors")
            )
        settings = {
            "dataset": "test",
            "variants": {"main": {"path": "test/models/main"}},
            "text_conditioning": {
                "enabled": text,
                "width": 24,
                "layers": [0],
                "inner": 16,
            },
        }
        (root / "test/futureworlds_config.json").write_text(json.dumps(settings))
        files = [
            {
                "path": p.relative_to(root).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
            for p in root.rglob("*")
            if p.is_file()
        ]
        (root / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": files})
        )
        return model.eval()

    def test_plain_and_text_checkpoint_roundtrip(self):
        for text in [False, True]:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original = self.make_bundle(root, text)
                loaded = load_world_model(root, "test")
                ids = torch.tensor([[1, 3, 5]])
                if text:
                    features, mask = (
                        torch.randn(1, 4, 24),
                        torch.ones(1, 4, dtype=torch.bool),
                    )
                    original.set_text(features, mask)
                    loaded.set_text(features, mask)
                with torch.no_grad():
                    torch.testing.assert_close(
                        original(input_ids=ids).logits,
                        loaded(input_ids=ids).logits,
                        rtol=0,
                        atol=0,
                    )

    def test_missing_adapter_is_not_silently_initialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bundle(root, True)
            (root / "test/models/main/text_adapters.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                load_world_model(root, "test", verify=False)

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bundle(root, False)
            (root / "test/models/main/config.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify_bundle(root)

    def test_paths_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["../secret", "/tmp/secret", "x/../../secret", "x\\..\\secret"]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_path(tmp, name)
