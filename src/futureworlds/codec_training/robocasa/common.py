"""Full-DROID codec data and native-area reconstruction evaluation."""

import json, random
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from futureworlds.data_readers.robocasa import rows, read_frames
from futureworlds.data_readers.robocasa import atomic as write

VIEWS = ["observation.images.robot0_agentview_left"]
from .sampling import VALIDATION_TIMES, episode_order, gap_group, target_times


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


def augment_clip(images, seed):
    # Identical color jitter across resized context and targets.
    from torchvision import transforms
    from torchvision.transforms import functional as F

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        order, b, c, s, h = transforms.ColorJitter.get_params(
            (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.05, 0.05)
        )
    out = []
    for frame in images:
        image = F.to_pil_image(frame / 255)
        for j in order:
            image = [
                F.adjust_brightness,
                F.adjust_contrast,
                F.adjust_saturation,
                F.adjust_hue,
            ][j](image, [b, c, s, h][j])
        out.append(F.to_tensor(image))
    return torch.stack(out)


class Images(Dataset):
    def __init__(self, records, total, seed):
        self.records, self.total, self.seed = records, total, seed
        self.tasks = sorted({r["task"] for r in records})
        self.by_task = {t: [r for r in records if r["task"] == t] for t in self.tasks}

    def __len__(self):
        return self.total

    @lru_cache(maxsize=4)
    def order(self, epoch):
        return episode_order(len(self.records), epoch, self.seed)

    def __getitem__(self, index):
        pool = self.by_task[self.tasks[index % len(self.tasks)]]
        row = pool[random.Random(self.seed + index * 9973 + 13).randrange(len(pool))]
        rng = random.Random(self.seed + index * 9973)
        # Context anchors vary over the full episode; favor full32-frame ranges where available.
        start = rng.randrange(max(1, row["frames"] - 40))
        times = target_times(row["frames"] - start, self.seed + 1000003 + index)
        view = VIEWS[0]
        arr = read_frames(row, [start] + [start + t for t in times], view)
        clip = augment_clip(
            torch.from_numpy(arr).permute(0, 3, 1, 2).float(),
            self.seed + 3000001 + index,
        )
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
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    totals = torch.zeros(len(VALIDATION_TIMES), 5, device=device, dtype=torch.float64)
    cases = []
    for row in records[rank::world]:
        rng = random.Random(int(row["id"][:16], 16) + 20260916)
        start = rng.randrange(max(1, row["frames"] - 40))
        times = [t for t in VALIDATION_TIMES if start + t < row["frames"]]
        im = read_frames(row, [start] + [start + t for t in times])
        ref = torch.from_numpy(im[:1]).permute(0, 3, 1, 2).float().to(device) / 255
        case = {"episode": row["id"], "start": start, "view": VIEWS[0], "frames": {}}
        for off in range(0, len(times), 4):
            selected = times[off : off + 4]
            truth = (
                torch.from_numpy(im[1 + off : 1 + off + len(selected)])
                .permute(0, 3, 1, 2)
                .float()
                .to(device)
                / 255
            )
            pred = (
                net(sample=ref, dyn_sample=truth, segment_len=len(selected))
                .sample.float()
                .clamp(0, 1)
            )
            l1 = (pred - truth).abs().mean((1, 2, 3))
            psnr = (
                -10 * (pred - truth).square().mean((1, 2, 3)).clamp_min(1e-12).log10()
            )
            lp = perceptual(pred * 2 - 1, truth * 2 - 1).flatten()
            ss = ssim(pred.contiguous(), truth.contiguous())
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
            cases.append(case)
    dist.all_reduce(totals)
    metrics = {}
    for group in ("short", "medium", "long", "all"):
        ids = [
            i
            for i, t in enumerate(VALIDATION_TIMES)
            if group == "all" or gap_group(t) == group
        ]
        v = totals[ids].sum(0)
        assert v[4] > 0
        metrics[group] = {
            **dict(zip(("l1", "lpips", "ssim", "psnr"), (v[:4] / v[4]).cpu().tolist())),
            "frames": int(v[4]),
        }
    metrics["valid_area"] = "256x320; direct bicubic resize of full native256x256 frame"
    metrics["native_hz"] = 20
    metrics["sampled_hz"] = 5
    if details:
        gathered = [None] * world
        dist.all_gather_object(gathered, cases)
        if rank == 0:
            metrics["cases"] = sum(gathered, [])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    return metrics
