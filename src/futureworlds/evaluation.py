"""Float-pixel metrics, causal rollout, explicit cohort and restart fingerprints."""

import hashlib, json, os, platform
from pathlib import Path
import numpy as np
import torch
from .data import read_manifest, read_case, write_json
from .checkpoints import file_sha256
from .protocol import implementation, text_identity


class Metrics:
    def __init__(self, device):
        import lpips, piqa

        self.device = torch.device(device)
        self.lpips = lpips.LPIPS(net="vgg").to(self.device).eval().requires_grad_(False)
        self.ssim = (
            piqa.SSIM(reduction="none").to(self.device).eval().requires_grad_(False)
        )

    @torch.inference_mode()
    def __call__(self, prediction, truth, horizons):
        p = torch.as_tensor(prediction, device=self.device).float().permute(0, 3, 1, 2)
        t = torch.as_tensor(truth, device=self.device).float().permute(0, 3, 1, 2) / 255
        if (
            p.shape != t.shape
            or not torch.isfinite(p).all()
            or p.min() < 0
            or p.max() > 1
        ):
            raise ValueError("Invalid prediction pixels")
        mse = (p - t).square().mean((1, 2, 3))
        mae = (p - t).abs().mean((1, 2, 3))
        lp = []
        ss = []
        for i in range(0, len(p), 2):
            lp.append(self.lpips(p[i : i + 2] * 2 - 1, t[i : i + 2] * 2 - 1).flatten())
            ss.append(self.ssim(p[i : i + 2], t[i : i + 2]))
        vectors = dict(
            mse=mse,
            mae=mae,
            psnr=-10 * mse.clamp_min(1e-12).log10(),
            lpips=torch.cat(lp),
            ssim=torch.cat(ss),
        )
        per_frame = {k: v.cpu().tolist() for k, v in vectors.items()}
        return dict(
            per_frame=per_frame,
            metrics={
                str(h): {k: float(np.mean(v[:h])) for k, v in per_frame.items()}
                for h in horizons
            },
        )


def evaluate(
    bundle,
    dataset,
    variant,
    manifest,
    output,
    device="cpu",
    metric_device=None,
    text_model=None,
    horizons=(10, 20, 32),
    beam_width=4,
    memory="full",
    expected_ids=None,
    limit=None,
):
    from .pipeline import WorldPipeline

    m = read_manifest(manifest, dataset)
    rows = m["cases"]
    if expected_ids is not None and [r["id"] for r in rows] != expected_ids:
        raise ValueError("Cohort differs from the fixed recipe")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    horizons = sorted(set(horizons))
    if not horizons or min(horizons) < 1:
        raise ValueError("Invalid horizons")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if device.startswith("cuda") or (metric_device or "").startswith("cuda"):
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        if device.startswith("cuda"):
            device = f"cuda:{local}"
        if (metric_device or "").startswith("cuda"):
            metric_device = f"cuda:{local}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Full bundle checksum validation takes place in the pipeline before model loading.
    pipe = WorldPipeline(bundle, dataset, variant, device, text_model)
    metrics = Metrics(metric_device or device)
    protocol = dict(
        dataset=dataset,
        variant=variant,
        manifest_sha256=file_sha256(manifest),
        bundle_manifest_sha256=file_sha256(Path(bundle) / "manifest.json"),
        horizons=horizons,
        beam_width=beam_width,
        memory=memory,
        metric_device=str(metrics.device).split(":")[0],
        generation_device=device.split(":")[0],
        torch=torch.__version__,
        numpy=np.__version__,
        implementation=implementation(),
        text_assets=text_identity(bundle, dataset, text_model),
        precision="float32",
        tf32=False,
        initial_observations=2,
        gt_refresh=False,
        cases=[r["id"] for r in rows],
        subset=limit is not None,
    )
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    for i in range(rank, len(rows), world):
        row = rows[i]
        dest = output / f"case_{i:06d}"
        receipt = dest / "result.json"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["fingerprint"] != fingerprint or old["id"] != row["id"]:
                raise ValueError("Refusing to mix old results with a changed protocol")
            if file_sha256(dest / "prediction.npz") != old["prediction_sha256"]:
                raise ValueError("Corrupt saved prediction")
            continue
        images, actions, text = read_case(manifest, row, pipe.action_dim, max(horizons))
        result = pipe.predict(
            images[:2], actions, text, max(horizons), beam_width, memory
        )
        values = metrics(result["prediction"], images[2:], horizons)
        dest.mkdir(exist_ok=True)
        tmp = dest / "prediction.partial"
        with tmp.open("wb") as f:
            np.savez_compressed(
                f,
                prediction=result["prediction"],
                tokens=result["tokens"],
                truth=images[2:],
                observations=images[:2],
            )
        tmp.replace(dest / "prediction.npz")
        write_json(
            receipt,
            dict(
                id=row["id"],
                fingerprint=fingerprint,
                prediction_sha256=file_sha256(dest / "prediction.npz"),
                **values,
                trace=result["trace"],
            ),
        )
        print(
            json.dumps(
                dict(
                    event="case_complete",
                    dataset=dataset,
                    variant=variant,
                    rank=rank,
                    index=i + 1,
                    total=len(rows),
                )
            ),
            flush=True,
        )
    if world > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("gloo")
        dist.barrier()
    if rank == 0:
        cases = [
            json.loads((output / f"case_{i:06d}/result.json").read_text())
            for i in range(len(rows))
        ]
        if any(r["fingerprint"] != fingerprint for r in cases):
            raise ValueError("Mixed protocol results")
        aggregate = {
            str(h): {
                k: float(np.mean([r["metrics"][str(h)][k] for r in cases]))
                for k in cases[0]["metrics"][str(h)]
            }
            for h in horizons
        }
        write_json(
            output / "summary.json",
            dict(
                state="COMPLETE",
                protocol=protocol,
                fingerprint=fingerprint,
                aggregate=aggregate,
                per_case=cases,
            ),
        )
        print(json.dumps(aggregate), flush=True)
    if world > 1:
        dist.barrier()
