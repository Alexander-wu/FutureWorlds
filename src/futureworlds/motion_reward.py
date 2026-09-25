"""GT motion-proxy ROI and GT-referenced dynamics. No actor/search sees future GT."""

import torch
import torch.nn.functional as F
from .rewards import CodecWorker
from .batch import TensorBatch


@torch.no_grad()
def motion_regions(truth, previous):
    """Shared across candidates: GT-derived, never shrinks after predicted disappearance."""
    prev = torch.cat([previous[None], truth[:-1]], 0)
    motion = (truth - prev).abs().mean(1, keepdim=True)
    thresholds = motion.flatten(1).quantile(0.90, dim=1).clamp_min(0.02)
    masks = (motion > thresholds[:, None, None, None]).float()
    masks = F.max_pool2d(masks, 11, stride=1, padding=5)
    fallback = masks.mean((1, 2, 3)) < 0.01
    masks[fallback] = 1
    crops = []
    height, width = truth.shape[-2:]
    for mask in masks:
        y, x = torch.where(mask[0] > 0)
        y0, y1 = int(y.min()), int(y.max()) + 1
        x0, x1 = int(x.min()), int(x.max()) + 1
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        h = max(64, y1 - y0)
        w = max(64, x1 - x0)
        y0 = max(0, min(cy - h // 2, height - h))
        x0 = max(0, min(cx - w // 2, width - w))
        crops.append((y0, min(height, y0 + h), x0, min(width, x0 + w)))
    return masks, crops, fallback


class MotionReward:
    def __init__(self, codec, perceptual, scales=None, alpha=0.5, beta=0.25):
        self.worker = CodecWorker(codec, perceptual)
        self.perceptual = perceptual
        self.scales = scales or dict(global_scale=1.0, roi_scale=1.0, delta_scale=1.0)
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def components(self, pred, pixels):
        truth = pixels[0, 2:]
        previous = pixels[0, 1]
        n, h = pred.shape[:2]
        real = truth[None].expand(n, -1, -1, -1, -1)
        mae = (pred - real).abs().mean((2, 3, 4))
        percept = self.worker.perceptual_loss(
            TensorBatch(dict(real=real, pred=pred))
        ).batch["perceptual_loss"]
        masks, crops, fallback = motion_regions(truth, previous)
        roi_mae = ((pred - real).abs() * masks).sum((2, 3, 4)) / (
            3 * masks.sum((1, 2, 3))
        )[None]
        roi_percept = torch.empty((n, h), device=pred.device)
        for t, (y0, y1, x0, x1) in enumerate(crops):
            r = truth[t : t + 1, :, y0:y1, x0:x1]
            for start in range(0, n, 2):
                pp = pred[start : start + 2, t, :, y0:y1, x0:x1]
                roi_percept[start : start + len(pp), t] = self.perceptual(
                    pp * 2 - 1, r.expand(len(pp), -1, -1, -1) * 2 - 1
                ).flatten()
        pred_prev = torch.cat(
            [previous[None, None].expand(n, 1, -1, -1, -1), pred[:, :-1]], 1
        )
        true_prev = torch.cat([previous[None], truth[:-1]], 0)
        delta = ((pred - pred_prev) - (truth - true_prev)[None]).abs().mean((2, 3, 4))
        return dict(
            mae=mae,
            lpips=percept,
            global_loss=mae + percept,
            roi_loss=roi_mae + roi_percept,
            roi_mae=roi_mae,
            roi_lpips=roi_percept,
            delta_loss=delta,
            roi_fallback=fallback,
            roi_area=masks.mean((1, 2, 3)),
        )

    def combine(self, c):
        s = self.scales
        return -(
            c["global_loss"]
            + self.alpha * s["global_scale"] / s["roi_scale"] * c["roi_loss"]
            + self.beta * s["global_scale"] / s["delta_scale"] * c["delta_loss"]
        ).mean(-1)

    @torch.no_grad()
    def decode(self, frames, context):
        codes = torch.stack([f["response"] for f in frames], 1)
        assert codes.min() >= 0 and codes.max() < 4375
        return self.worker.detokenize(
            TensorBatch(
                dict(tokens=codes, ctx_tokens=context.expand(len(codes), -1, -1))
            ),
            None,
        ).batch["pixels"][:, 1:]

    @torch.no_grad()
    def __call__(self, frames, context, actions, pixels):
        c = self.components(self.decode(frames, context), pixels)
        reward = self.combine(c)
        assert torch.isfinite(reward).all()
        metrics = {
            "critic/recon_loss/mean": float(c["mae"].mean()),
            "critic/perceptual_loss/mean": float(c["lpips"].mean()),
        }
        metrics.update({k: float(v.float().mean()) for k, v in c.items()})
        return reward, metrics
