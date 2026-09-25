"""Native FP32 RGB/action/instruction -> autoregressive RGB predictions."""

import json
from pathlib import Path
import numpy as np
import torch
from .checkpoints import load_world_model, safe_path, verify_bundle
from .memory import make_prefix
from .decoding import beam_decode
from .rewards import CodecWorker
from .batch import TensorBatch


def encode_actions(actions, ranges, dimension):
    if actions.ndim != 2 or actions.shape[-1] != dimension:
        raise ValueError("Actions must already use the dataset temporal layout")
    if ranges.shape == (13, 2) and dimension == 52:
        ranges = ranges.repeat(4, 1)
    if ranges.shape != (dimension, 2) or not torch.isfinite(ranges).all():
        raise ValueError("Action ranges do not match model")
    if not torch.isfinite(actions).all() or (ranges[:, 1] < ranges[:, 0]).any():
        raise ValueError("Invalid action values/ranges")
    return (
        ((actions - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0] + 1e-8)).clamp(0, 1)
        * 256
    ).floor().clamp(0, 255).long() + 8750


class WorldPipeline:
    def __init__(
        self,
        bundle,
        dataset,
        variant="main",
        device="cpu",
        text_model=None,
        verify=True,
    ):
        from .codec import CompressiveVQModelFSQ

        self.bundle = Path(bundle)
        self.device = torch.device(device)
        if verify:
            verify_bundle(self.bundle)
        self.settings = json.loads(
            safe_path(self.bundle, f"{dataset}/futureworlds_config.json").read_text()
        )
        self.model = load_world_model(bundle, dataset, variant, device, verify=False)
        self.codec = (
            CompressiveVQModelFSQ.from_pretrained(
                safe_path(self.bundle, self.settings["codec"]), local_files_only=True
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.ranges = torch.load(
            safe_path(self.bundle, self.settings["action_ranges"]),
            map_location=self.device,
            weights_only=True,
        )
        self.action_dim = self.settings["action_tokens_per_step"]
        self.worker = CodecWorker(self.codec, None)
        self.text_model_path = text_model
        self.text_encoder = None
        self.text_tokenizer = None
        self.text_cache = {}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    @torch.no_grad()
    def text(self, instruction):
        tc = self.settings["text_conditioning"]
        if not tc["enabled"]:
            return {}
        if instruction not in self.text_cache:
            if self.text_encoder is None:
                from transformers import AutoTokenizer, T5EncoderModel

                source = self.text_model_path or tc["repo_id"]
                kw = {"local_files_only": bool(self.text_model_path)}
                if not self.text_model_path:
                    kw["revision"] = tc["revision"]
                self.text_tokenizer = AutoTokenizer.from_pretrained(source, **kw)
                self.text_encoder = (
                    T5EncoderModel.from_pretrained(
                        source, torch_dtype=torch.float32, **kw
                    )
                    .to(self.device)
                    .eval()
                    .requires_grad_(False)
                )
            tokens = self.text_tokenizer(
                [instruction], return_tensors="pt", truncation=False
            ).to(self.device)
            if tokens.input_ids.shape[1] > tc["max_tokens"]:
                raise ValueError(
                    "Instruction exceeds audited token limit; truncation prohibited"
                )
            mask = tokens.attention_mask.bool().cpu()
            if not instruction.strip() and self.settings["dataset"] == "robocasa":
                mask.zero_()
            self.text_cache[instruction] = (
                self.text_encoder(**tokens).last_hidden_state.detach().cpu(),
                mask,
            )
        f, m = self.text_cache[instruction]
        return dict(text_features=f.to(self.device), text_mask=m.to(self.device))

    @torch.no_grad()
    def condition(self, instruction):
        cond = self.text(instruction)
        if cond:
            self.model.set_text(cond["text_features"], cond["text_mask"])
        return cond

    @torch.no_grad()
    def encode_observations(self, observations):
        if observations.dtype != np.uint8 or observations.shape != (2, 256, 320, 3):
            raise ValueError("Exactly two initial uint8 RGB observations required")
        pixels = (
            torch.from_numpy(observations.copy())
            .permute(0, 3, 1, 2)
            .to(self.device)
            .float()[None]
            / 255
        )
        ctx, anchor = self.codec.tokenize(pixels.contiguous())
        if ctx.shape != (1, 1, 1280) or anchor.shape != (1, 1, 80):
            raise ValueError("Unexpected codec token dimensions")
        return ctx, anchor.reshape(1, 80)

    @torch.no_grad()
    def predict(
        self,
        observations,
        actions,
        instruction="",
        horizon=32,
        beam_width=4,
        memory="full",
    ):
        # No future image is accepted by this interface.
        if horizon < 1 or len(actions) < horizon + 1:
            raise ValueError("Insufficient planned controls")
        ctx, anchor = self.encode_observations(observations)
        acts = encode_actions(
            torch.as_tensor(actions, device=self.device), self.ranges, self.action_dim
        )
        self.condition(instruction)
        self.model.eval()
        history = {1: anchor}
        responses = []
        traces = []
        for t in range(2, horizon + 2):
            context = ctx.reshape(1, -1) + 4375
            if memory == "full":
                prefix, kept = make_prefix(context, history, acts, t)
            else:
                if memory == "recent":
                    kept = list(range(max(1, t - 6), t))
                elif memory == "minimal":
                    kept = [t - 1]
                else:
                    raise ValueError(memory)
                prefix = torch.cat(
                    [context]
                    + [x for k in kept for x in (history[k], acts[k : k + 1])],
                    1,
                )
            candidates, _, trace = beam_decode(self.model, prefix, width=beam_width)
            history[t] = candidates[:1]
            responses.append(candidates[:1])
            traces.append(dict(trace, target=t, retained=kept))
            # Bound host and device history as well as the model's context.
            history = {k: v for k, v in history.items() if k == 1 or k >= t - 5}
        codes = torch.stack(responses, 1)
        prediction = self.worker.detokenize(
            TensorBatch({"tokens": codes, "ctx_tokens": ctx}), None
        ).batch["pixels"][:, 1:]
        return dict(
            tokens=codes.cpu().numpy(),
            prediction=prediction[0].permute(0, 2, 3, 1).cpu().numpy(),
            trace=traces,
        )

    @torch.no_grad()
    def encode_training_window(self, images, actions, instruction):
        """Encode teacher-forced history once. Only reward/SFT may access future RGB."""
        frames = (
            torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to(self.device).float()
            / 255
        )
        first = frames[:1]
        ctx, initial = self.codec.tokenize(first.repeat(2, 1, 1, 1)[None].contiguous())
        parts = [initial]
        for start in range(1, len(frames), 8):
            c, d = self.codec.tokenize(
                torch.cat([first, frames[start : start + 8]])[None].contiguous()
            )
            if not torch.equal(c, ctx):
                raise ValueError("Context changes with future frames")
            parts.append(d)
        dynamics = torch.cat(parts, 1)[0]
        acts = encode_actions(
            torch.as_tensor(actions, device=self.device), self.ranges, self.action_dim
        )
        return ctx, dynamics, acts, frames[None], self.condition(instruction)
