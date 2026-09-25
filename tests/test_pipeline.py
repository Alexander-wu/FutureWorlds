"""Offline integration tests with the real native codec and a tiny random Llama."""

import json, tempfile, unittest, copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from futureworlds.data import write_json, read_manifest, read_case, disjoint
from futureworlds.checkpoints import file_sha256
from futureworlds.pipeline import WorldPipeline, encode_actions
from futureworlds.training import train, sft_example


class PortablePipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_bundle(self, root):
        from futureworlds.codec import CompressiveVQModelFSQ

        torch.manual_seed(18)
        codec = CompressiveVQModelFSQ(
            block_out_channels=(32, 32, 32, 32),
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            layers_per_block=1,
            latent_channels=8,
            norm_num_groups=8,
            mid_block_add_attention=False,
            lookup_from_codebook=True,
        )
        codec.save_pretrained(root / "bridge/codec")
        core = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=9010,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=4096,
                attention_dropout=0.0,
            )
        )
        core.save_pretrained(root / "bridge/models/main")
        torch.save(torch.tensor([[-1.0, 1.0]] * 13), root / "bridge/ranges.pt")
        write_json(
            root / "bridge/futureworlds_config.json",
            dict(
                dataset="bridge",
                variants={k: {"path": "bridge/models/main"} for k in ["main", "sft"]},
                codec="bridge/codec",
                action_ranges="bridge/ranges.pt",
                action_tokens_per_step=13,
                text_conditioning={"enabled": False},
            ),
        )
        files = [
            dict(
                path=str(p.relative_to(root)),
                bytes=p.stat().st_size,
                sha256=file_sha256(p),
            )
            for p in root.rglob("*")
            if p.is_file()
        ]
        write_json(root / "manifest.json", dict(schema_version=1, files=files))

    def make_data(self, root):
        root.mkdir()
        rng = np.random.default_rng(19)
        images = rng.integers(0, 256, (4, 256, 320, 3), dtype=np.uint8)
        actions = rng.uniform(-1, 1, (4, 13)).astype(np.float32)
        np.savez_compressed(
            root / "sample.npz",
            images=images,
            actions=actions,
            instruction=np.array(""),
        )
        row = dict(
            id="example",
            episode_id="episode",
            file="sample.npz",
            sha256=file_sha256(root / "sample.npz"),
            frames=4,
            split="train",
        )
        write_json(
            root / "manifest.json",
            dict(schema_version=1, dataset="bridge", cases=[row]),
        )
        return images, actions, row

    def test_rgb_causal_prediction_and_native_decode(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_bundle(root / "bundle")
            images, actions, _ = self.make_data(root / "data")
            pipe = WorldPipeline(root / "bundle", "bridge", device="cpu")
            pred = pipe.predict(images[:2], actions, horizon=2)
            self.assertEqual(pred["prediction"].shape, (2, 256, 320, 3))
            self.assertEqual(pred["tokens"].shape, (1, 2, 80))
            self.assertEqual(pred["trace"][1]["retained"], [1, 2])
            self.assertTrue(np.isfinite(pred["prediction"]).all())
            with self.assertRaises(ValueError):
                pipe.predict(images, actions, horizon=1)
            c, a = pipe.encode_observations(images[:2])
            cc, dyn, _, _, _ = pipe.encode_training_window(images, actions, "")
            self.assertTrue(torch.equal(c, cc))
            self.assertTrue(torch.equal(a, dyn[1:2]))

    def test_native_sft_and_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_bundle(root / "bundle")
            self.make_data(root / "data")
            cfg = dict(
                method="sft",
                dataset="bridge",
                bundle=str(root / "bundle"),
                train_manifest=str(root / "data/manifest.json"),
                output=str(root / "run"),
                device="cpu",
                seed=4,
                global_batch=1,
                steps=2,
                lr=1e-5,
                warmup_steps=1,
                checkpoint_every=1,
            )
            write_json(root / "train.json", cfg)
            import futureworlds.training as module

            original = module.save_checkpoint

            def stop_after_one(*args, **kwargs):
                original(*args, **kwargs)
                raise InterruptedError("controlled interruption")

            with patch.object(module, "save_checkpoint", side_effect=stop_after_one):
                with self.assertRaises(InterruptedError):
                    train(root / "train.json")
            checkpoint = root / "run/checkpoints/step_000001"
            self.assertTrue((checkpoint / "training_state.pt").exists())
            cfg["resume"] = str(checkpoint)
            write_json(root / "train.json", cfg)
            train(root / "train.json")
            state = torch.load(
                root / "run/checkpoints/step_000002/training_state.pt",
                weights_only=False,
            )
            self.assertEqual(state["step"], 2)
            self.assertEqual(
                {int(s["step"]) for s in state["optimizer"]["state"].values()}, {2}
            )
            cfg.pop("resume")
            cfg["output"] = str(root / "uninterrupted")
            write_json(root / "train.json", cfg)
            train(root / "train.json")
            from safetensors.torch import load_file

            a = load_file(str(root / "run/checkpoints/step_000002/model.safetensors"))
            b = load_file(
                str(root / "uninterrupted/checkpoints/step_000002/model.safetensors")
            )
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_action_order_and_split_guards(self):
        from futureworlds.data_readers.robocasa import ordered_actions

        native = np.arange(9 * 12, dtype=np.float32).reshape(9, 12)
        actions = ordered_actions(native, np.array([0, 4, 8]))
        np.testing.assert_array_equal(actions[0].reshape(4, 13)[:, :12], native[:4])
        self.assertTrue((actions[:2].reshape(8, 13)[:, 12] == 0).all())
        tokens = encode_actions(
            torch.zeros(3, 52), torch.tensor([[-1.0, 1.0]] * 13), 52
        )
        self.assertTrue((tokens == 8878).all())
        with self.assertRaises(ValueError):
            disjoint({"cases": [{"episode_id": "x"}]}, {"cases": [{"episode_id": "x"}]})

    def test_frozen_cache_matches_native_encoding(self):
        from futureworlds.cache import cache_dataset, load_cached
        from futureworlds.protocol import feature_fingerprint

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_bundle(root / "bundle")
            images, actions, row = self.make_data(root / "data")
            cache_dataset(
                root / "bundle",
                "bridge",
                root / "data/manifest.json",
                root / "cache",
                device="cpu",
            )
            actual = load_cached(
                root / "cache",
                row,
                feature_fingerprint(root / "bundle", "bridge", None),
                "cpu",
            )
            pipe = WorldPipeline(root / "bundle", "bridge", variant="sft", device="cpu")
            expected = pipe.encode_training_window(images, actions, "")
            for a, b in zip(actual[:3], expected[:3]):
                self.assertTrue(torch.equal(a, b))
            cfg = dict(
                method="sft",
                dataset="bridge",
                bundle=str(root / "bundle"),
                train_manifest=str(root / "data/manifest.json"),
                output=str(root / "run"),
                device="cpu",
                seed=4,
                global_batch=1,
                steps=1,
                lr=1e-5,
                warmup_steps=1,
                checkpoint_every=1,
                token_cache=str(root / "cache"),
            )
            write_json(root / "train.json", cfg)
            train(root / "train.json")
            self.assertTrue((root / "run/complete.json").is_file())

    def test_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, row = self.make_data(root / "data")
            read_case(root / "data/manifest.json", row, 13, 2)
            (root / "data/sample.npz").write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                read_case(root / "data/manifest.json", row, 13, 2)


if __name__ == "__main__":
    unittest.main()
