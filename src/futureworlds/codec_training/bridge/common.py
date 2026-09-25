"""One 16-rank, representation-preserving decoder fine-tune."""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import random
import time
from functools import lru_cache
from pathlib import Path

import lpips
import numpy as np
import piqa
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from futureworlds.codec import CompressiveVQModelFSQ
from .sampling import VALIDATION_TIMES, episode_order, gap_group, target_times


def augment_clip(images, seed):
    """Match the published crop -> PIL color-jitter pipeline, shared over the clip."""
    from torchvision import transforms
    from torchvision.transforms import functional as F

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            images, (0.8, 1.0), (1.0, 1.4)
        )
        order, brightness, contrast, saturation, hue = (
            transforms.ColorJitter.get_params(
                (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.05, 0.05)
            )
        )
    out = []
    for frame in images:
        image = F.to_pil_image(F.resized_crop(frame, i, j, h, w, [256, 320]) / 255)
        for index in order:
            if index == 0:
                image = F.adjust_brightness(image, brightness)
            elif index == 1:
                image = F.adjust_contrast(image, contrast)
            elif index == 2:
                image = F.adjust_saturation(image, saturation)
            else:
                image = F.adjust_hue(image, hue)
        out.append(F.to_tensor(image))
    return torch.stack(out)


def install_checkpointing(model):
    from torch.utils.checkpoint import checkpoint
    import types

    for module in (
        model.encoder,
        model.decoder,
        model.cond_encoder,
        model.cond_decoder,
    ):
        original = module.forward

        def forward(self, *args, _original=original, **kwargs):
            if self.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        module.forward = types.MethodType(forward, module)


def gradient_penalty(images, logits, coefficient=10.0):
    grads = torch.autograd.grad(
        logits.sum(), images, create_graph=True, retain_graph=True
    )[0]
    return coefficient * (grads.flatten(1).norm(2, dim=1) - 1).square().mean()


def synchronized_parameters(net):
    values = torch.stack([p.detach().double().sum() for p in net.parameters()])
    lower, upper = values.clone(), values.clone()
    dist.all_reduce(lower, op=dist.ReduceOp.MIN)
    dist.all_reduce(upper, op=dist.ReduceOp.MAX)
    assert torch.equal(lower, upper), "Cross-rank parameter divergence"
    return values


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


class Images(Dataset):
    def __init__(self, records, total, seed):
        self.records, self.total, self.seed = records, total, seed

    def __len__(self):
        return self.total

    @lru_cache(maxsize=4)
    def order(self, epoch):
        return episode_order(len(self.records), epoch, self.seed)

    def __getitem__(self, index):
        row = self.records[
            self.order(index // len(self.records))[index % len(self.records)]
        ]
        with np.load(row["path"], allow_pickle=False) as z:
            images = z["image"]
        assert len(images) == row["frames"] and images.shape[1:] == (256, 320, 3)
        times = target_times(len(images), self.seed + 1000003 + index)
        clip = torch.from_numpy(images[[0] + times].copy()).permute(0, 3, 1, 2).float()
        clip = augment_clip(clip, self.seed + 3000001 + index)
        return clip[0], clip[1:], torch.tensor(times)


class RankSampler(Sampler):
    def __init__(self, total, rank, world):
        self.total, self.rank, self.world = total, rank, world

    def __iter__(self):
        return iter(range(self.rank, self.total, self.world))

    def __len__(self):
        return len(range(self.rank, self.total, self.world))


@torch.inference_mode()
def validate(net, records, device, rank, world, perceptual, ssim, details=False):
    net.eval()
    old_matmul, old_cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    totals = torch.zeros(len(VALIDATION_TIMES), 5, device=device, dtype=torch.float64)
    local_cases = []
    for row in records[rank::world]:
        with np.load(row["path"], allow_pickle=False) as z:
            im = z["image"]
        times = [t for t in VALIDATION_TIMES if t < len(im)]
        ref = (
            torch.from_numpy(im[0].copy()).permute(2, 0, 1)[None].float().to(device)
            / 255
        ).contiguous()
        case = {"episode": row["output"], "frames": {}}
        for start in range(0, len(times), 4):
            selected = times[start : start + 4]
            truth = (
                torch.from_numpy(im[selected].copy())
                .permute(0, 3, 1, 2)
                .float()
                .to(device)
                / 255
            ).contiguous()
            pred = net(
                sample=ref, dyn_sample=truth, segment_len=len(selected)
            ).sample.clamp(0, 1)
            l1 = (pred - truth).abs().mean((1, 2, 3))
            mse = (pred - truth).square().mean((1, 2, 3))
            psnr = -10 * torch.log10(mse.clamp_min(1e-12))
            lp = perceptual(pred * 2 - 1, truth * 2 - 1).flatten()
            ss = ssim(pred, truth)
            for j, t in enumerate(selected):
                values = torch.stack(
                    [l1[j], lp[j], ss[j], psnr[j], torch.ones((), device=device)]
                )
                totals[VALIDATION_TIMES.index(t)] += values
                if details:
                    case["frames"][str(t)] = dict(
                        zip(("l1", "lpips", "ssim", "psnr"), values[:4].cpu().tolist())
                    )
        if details:
            local_cases.append(case)
    dist.all_reduce(totals)
    metrics = {}
    for group in ("short", "medium", "long", "all"):
        ids = [
            i
            for i, t in enumerate(VALIDATION_TIMES)
            if group == "all" or gap_group(t) == group
        ]
        v = totals[ids].sum(0)
        metrics[group] = {
            **dict(zip(("l1", "lpips", "ssim", "psnr"), (v[:4] / v[4]).cpu().tolist())),
            "frames": int(v[4]),
        }
    metrics["per_gap"] = {
        str(t): {
            **dict(
                zip(
                    ("l1", "lpips", "ssim", "psnr"),
                    (totals[i, :4] / totals[i, 4]).cpu().tolist(),
                )
            ),
            "frames": int(totals[i, 4]),
        }
        for i, t in enumerate(VALIDATION_TIMES)
        if totals[i, 4] > 0
    }
    if details:
        gathered = [None] * world
        dist.all_gather_object(gathered, local_cases)
        if rank == 0:
            metrics["cases"] = sorted(
                [r for shard in gathered for r in shard], key=lambda r: r["episode"]
            )
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
        old_matmul,
        old_cudnn,
    )
    return metrics
