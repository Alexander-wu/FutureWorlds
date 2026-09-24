"""Use the upstream msp_reward_fn via a local frozen-tokenizer worker adapter."""

from types import SimpleNamespace as NS
import torch
from .batch import TensorBatch
from .reward_core import msp_reward_fn


class CodecWorker:
    def __init__(self, codec, perceptual):
        self.codec = codec
        self.perceptual = perceptual

    @torch.no_grad()
    def detokenize(self, batch, dummy):
        codes = batch.batch["tokens"]
        ctx = batch.batch["ctx_tokens"]
        movies = []
        for b in range(codes.shape[0]):
            parts = []
            first = None
            for start in range(0, codes.shape[1], 2):
                decoded = (
                    self.codec.detokenize(
                        ctx[b : b + 1], codes[b : b + 1, start : start + 2]
                    )
                    .float()
                    .clamp(0, 1)
                )
                if first is None:
                    first = decoded[:, :1]
                parts.append(decoded[:, 1:])
            movies.append(torch.cat([first] + parts, 1))
        return TensorBatch({"pixels": torch.cat(movies, 0)})

    @torch.no_grad()
    def perceptual_loss(self, batch):
        real, pred = batch.batch["real"], batch.batch["pred"]
        values = []
        for r, p in zip(real, pred):
            pieces = []
            for i in range(0, len(p), 2):
                pieces.append(
                    self.perceptual(
                        p[i : i + 2] * 2 - 1, r[i : i + 2] * 2 - 1
                    ).flatten()
                )
            values.append(torch.cat(pieces))
        return TensorBatch({"perceptual_loss": torch.stack(values)})


class VideoReward:
    msp_reward_fn = msp_reward_fn

    def __init__(self, codec, perceptual, horizon, action_dim=13):
        self.config = NS(
            data=NS(video=NS(segment_length=horizon + 1)),
            processor=NS(
                tokens_per_frame=80, action_dim=action_dim, visual_token_num=4375
            ),
            trainer=NS(reward_fn="mae", msp_reward_aggregate="mean"),
        )
        self.tokenizer_wg = CodecWorker(codec, perceptual)

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels):
        # Only reward bookkeeping uses concatenated responses. Policy scoring uses actual per-frame prefixes.
        codes = torch.stack([f["response"] for f in frames], 1)
        assert codes.min() >= 0 and codes.max() < 4375, (
            "Illegal generated token; never hide it with upstream clamp"
        )
        group, horizon, _ = codes.shape
        act = actions[2 : 2 + horizon].unsqueeze(0).expand(group, -1, -1)
        responses = torch.cat([codes, act], -1).reshape(group, -1)
        prompt = frames[0]["prefix"]
        batch = TensorBatch(
            {
                "responses": responses,
                "prompts": prompt,
                "attention_mask": torch.ones(
                    group,
                    prompt.shape[1] + responses.shape[1],
                    device=codes.device,
                    dtype=torch.long,
                ),
                "ctx_tokens": context.expand(group, -1, -1),
            }
        )
        truth = pixels.expand(group, -1, -1, -1, -1)
        tensor, metrics = self.msp_reward_fn(batch, truth)
        reward = tensor.sum(-1)
        assert torch.isfinite(reward).all()
        return reward, metrics
