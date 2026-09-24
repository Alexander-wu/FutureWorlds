"""Bridge paper recipe with distant targets; one full-tokenizer 16-rank run."""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import lpips
import numpy as np
import piqa
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from futureworlds.codec import CompressiveVQModelFSQ
from .common import Images, RankSampler, rows, write, validate, augment_clip
from .common import install_checkpointing, gradient_penalty, synchronized_parameters
from .discriminator import Discriminator
from .sampling import is_generator_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    run, data = Path(cfg["run_root"]), Path(cfg["data_root"])
    rank, local, world = (
        int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    assert world == 16 and cfg["early_stopping"] is False
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(minutes=20), device_id=device
    )
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    def log(event, **values):
        if rank == 0:
            print(
                "LONG_RECON="
                + json.dumps({"event": event, "smoke": args.smoke, **values}),
                flush=True,
            )

    train = rows(data / "train_manifest.jsonl")
    dev = sorted(rows(data / "dev_manifest.jsonl"), key=lambda r: r["output"])
    assert len(train) == 24948 and len(dev) == 512
    assert not {r["output"] for r in train} & {r["output"] for r in dev}
    assert all(r["source_split"] == "train" and r["frames"] >= 2 for r in train + dev)
    if rank == 0:
        for part in ("train", "dev"):
            assert (
                hashlib.sha256(
                    (data / f"{part}_manifest.jsonl").read_bytes()
                ).hexdigest()
                == cfg[f"{part}_manifest_sha256"]
            )
        assert (
            hashlib.sha256(
                (
                    Path(cfg["tokenizer"]) / "diffusion_pytorch_model.safetensors"
                ).read_bytes()
            ).hexdigest()
            == cfg["tokenizer_sha256"]
        )
    dist.barrier()
    model = CompressiveVQModelFSQ.from_pretrained(
        cfg["tokenizer"], local_files_only=True
    ).to(device)
    assert model.patch_size == 4 and model.context_length == 1
    model.requires_grad_(True)
    install_checkpointing(model)
    discriminator = Discriminator(depth=6).to(device)
    perceptual = lpips.LPIPS(net="vgg").eval().to(device).requires_grad_(False)
    ssim = piqa.SSIM(reduction="none").eval().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    doptimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=cfg["discriminator_lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    ddp = DDP(
        model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True
    )
    disc = DDP(discriminator, device_ids=[local], broadcast_buffers=True)
    model_sums = synchronized_parameters(model)
    disc_sums = synchronized_parameters(discriminator)

    def evaluation(net, records, details=False):
        return validate(net, records, device, rank, world, perceptual, ssim, details)

    # Check native reconstruction and clip-consistent augmentation before any update.
    with np.load(dev[rank]["path"], allow_pickle=False) as z:
        arr = z["image"][[0, 1, min(32, len(z["image"]) - 1)]].copy()
    probe = (
        torch.from_numpy(arr).permute(0, 3, 1, 2)[None].float().to(device) / 255
    ).contiguous()
    model.eval()
    with torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        c, d = model.tokenize(probe)
        assert c.shape == (1, 1, 1280) and d.shape == (1, 2, 80)
        direct = model(
            sample=probe[:, 0], dyn_sample=probe[0, 1:], segment_len=2
        ).sample
        torch.testing.assert_close(
            direct, model.detokenize(c, d)[0, 1:], atol=1e-5, rtol=0
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    repeated = (
        torch.from_numpy(np.repeat(arr[:1], 8, axis=0)).permute(0, 3, 1, 2).float()
    )
    augmented = augment_clip(repeated, cfg["seed"])
    assert (
        torch.equal(augmented[0], augmented[7])
        and 0 <= augmented.min()
        and augmented.max() <= 1
    )

    steps = 6 if args.smoke else cfg["steps"]
    disc_start = 2 if args.smoke else cfg["discriminator_start"]
    batch, accum = cfg["global_batch"], cfg["global_batch"] // world
    assert batch == accum * world
    dataset = Images(train, steps * batch, cfg["seed"])
    loader = iter(
        DataLoader(
            dataset,
            batch_size=1,
            sampler=RankSampler(len(dataset), rank, world),
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )
    )
    validation_rows = dev[: 16 if args.smoke else 128]
    initial = evaluation(model, validation_rows)
    best = {
        "step": 0,
        "checkpoint": cfg["tokenizer"],
        "validation": initial,
        "published_baseline": True,
    }
    if rank == 0 and not args.smoke:
        assert not (run / "metrics.jsonl").exists()
        (run / "metrics.jsonl").write_text(
            json.dumps({"shared_step": 0, "validation": initial}) + "\n"
        )
        write(run / "best.json", best)
    log(
        "START",
        shared_steps=steps,
        world_size=world,
        global_batch=batch,
        accumulation=accum,
        train_episodes=len(train),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        discriminator_parameters=sum(p.numel() for p in discriminator.parameters()),
        initial=initial,
    )
    g_updates = d_updates = 0
    started = time.monotonic()
    for shared in range(steps):
        step = shared + 1
        generator = is_generator_step(shared, disc_start)
        gan_active = shared >= disc_start
        count = (g_updates if generator else d_updates) + 1
        warm = min(1.0, count / cfg["warmup"])
        cosine = 0.5 * (
            1
            + math.cos(
                math.pi
                * max(0.0, (count - cfg["warmup"]) / (cfg["steps"] - cfg["warmup"]))
            )
        )
        lr = (
            (cfg["lr"] * warm * cosine) if generator else cfg["discriminator_lr"] * warm
        )
        active_opt = optimizer if generator else doptimizer
        for group in active_opt.param_groups:
            group["lr"] = lr
        active_opt.zero_grad(set_to_none=True)
        model.train()
        discriminator.train().requires_grad_(not generator)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        totals = torch.zeros(8, device=device, dtype=torch.float64)
        gaps = torch.zeros(3, device=device, dtype=torch.float64)
        for micro in range(accum):
            ref, truth, times = next(loader)
            ref = ref.to(device, non_blocking=True)
            truth = truth.to(device, non_blocking=True).reshape(-1, 3, 256, 320)
            for j, (lo, hi) in enumerate(((1, 7), (8, 16), (17, 32))):
                gaps[j] += int(((times >= lo) & (times <= hi)).sum())
            if generator:
                with ddp.no_sync() if micro < accum - 1 else contextlib.nullcontext():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred, pred_ref, commit, dyn_commit = ddp(
                            sample=ref,
                            dyn_sample=truth,
                            segment_len=7,
                            return_dict=False,
                            return_loss=True,
                        )
                    pred, pred_ref = pred.float(), pred_ref.float()
                    l1, ref_l1 = (
                        (pred - truth).abs().mean(),
                        (pred_ref - ref).abs().mean(),
                    )
                    lp = perceptual(pred * 2 - 1, truth * 2 - 1).mean()
                    ref_lp = perceptual(pred_ref * 2 - 1, ref * 2 - 1).mean()
                    adv = (
                        -discriminator(torch.cat([pred_ref, pred])).mean()
                        if gan_active
                        else torch.zeros((), device=device)
                    )
                    loss = (
                        l1
                        + ref_l1
                        + lp
                        + ref_lp
                        + cfg["gan_weight"] * adv
                        + commit
                        + dyn_commit
                    )
                    (loss / accum).backward()
                values = [
                    loss.detach(),
                    l1.detach(),
                    ref_l1.detach(),
                    lp.detach(),
                    ref_lp.detach(),
                    adv.detach(),
                    torch.zeros((), device=device),
                    torch.zeros((), device=device),
                ]
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, pred_ref, _, _ = model(
                        sample=ref,
                        dyn_sample=truth,
                        segment_len=7,
                        return_dict=False,
                        return_loss=True,
                    )
                truth.requires_grad_(True)
                # Full FP32 discriminator/GP avoids low-precision second-derivative overflow.
                with disc.no_sync() if micro < accum - 1 else contextlib.nullcontext():
                    all_logits = disc(
                        torch.cat(
                            [
                                ref,
                                truth,
                                pred_ref.detach().float(),
                                pred.detach().float(),
                            ]
                        )
                    )
                    real, fake = all_logits.chunk(2)
                    hinge = torch.relu(1 - real).mean() + torch.relu(1 + fake).mean()
                    gp = gradient_penalty(truth, real, cfg["gradient_penalty_weight"])
                    loss = hinge + gp
                    (loss / accum).backward()
                zero = torch.zeros((), device=device)
                values = [
                    loss.detach(),
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    hinge.detach(),
                    gp.detach(),
                ]
            totals += torch.stack(values).double() / accum
        active_model = model if generator else discriminator
        grad = torch.nn.utils.clip_grad_norm_(
            active_model.parameters(), 1.0, error_if_nonfinite=True
        )
        active_opt.step()
        if generator:
            g_updates += 1
        else:
            d_updates += 1
        torch.cuda.synchronize()
        timing = torch.tensor(
            [time.monotonic() - tick, torch.cuda.max_memory_allocated() / 1024**2],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(totals)
        totals /= world
        dist.all_reduce(gaps)
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        info = dict(
            zip(
                (
                    "loss",
                    "dynamics_l1",
                    "context_l1",
                    "dynamics_lpips",
                    "context_lpips",
                    "adversarial",
                    "hinge",
                    "gradient_penalty",
                ),
                totals.cpu().tolist(),
            )
        )
        info.update(
            shared_step=step,
            step=step,
            branch="generator" if generator else "discriminator",
            generator_updates=g_updates,
            discriminator_updates=d_updates,
            gan_active=gan_active,
            lr=lr,
            grad_norm=float(grad),
            max_step_seconds=float(timing[0]),
            max_peak_allocated_MiB=float(timing[1]),
            gap_counts=gaps.cpu().tolist(),
            episode_presentations=step * batch,
            elapsed_seconds=time.monotonic() - started,
            updated_unix=time.time(),
        )
        if args.smoke or step % cfg["validate_every"] == 0 or step == steps:
            sums = synchronized_parameters(model)
            assert not torch.equal(sums, model_sums), "Generator weights did not update"
            if d_updates:
                assert not torch.equal(
                    synchronized_parameters(discriminator), disc_sums
                ), "Discriminator weights did not update"
            if not args.smoke:
                val = evaluation(model, validation_rows)
                info["validation"] = val
                eligible = (
                    val["short"]["lpips"]
                    <= initial["short"]["lpips"] * cfg["short_tolerance"]
                )
                improved = (
                    eligible
                    and val["long"]["lpips"] < best["validation"]["long"]["lpips"]
                )
                info.update(eligible=eligible, improved=improved)
                save_due = (
                    improved or step % cfg["checkpoint_every"] == 0 or step == steps
                )
                if save_due:
                    path = run / "checkpoints" / f"step_{step:06d}"
                    if rank == 0:
                        assert not path.exists()
                        temp = path.with_name(path.name + ".partial")
                        model.save_pretrained(temp)
                        torch.save(
                            {
                                "shared_step": step,
                                "generator_updates": g_updates,
                                "discriminator_updates": d_updates,
                                "optimizer": optimizer.state_dict(),
                                "discriminator_optimizer": doptimizer.state_dict(),
                                "discriminator": discriminator.state_dict(),
                                "config": cfg,
                                "torch_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all(),
                            },
                            temp / "training_state.pt",
                        )
                        write(
                            temp / "experiment.json",
                            {
                                "shared_step": step,
                                "validation": val,
                                "config": cfg,
                                "weight_sha256": hashlib.sha256(
                                    (
                                        temp / "diffusion_pytorch_model.safetensors"
                                    ).read_bytes()
                                ).hexdigest(),
                            },
                        )
                        temp.replace(path)
                    dist.barrier()
                    if improved:
                        best = {
                            "step": step,
                            "checkpoint": str(path),
                            "validation": val,
                            "published_baseline": False,
                        }
                        if rank == 0:
                            write(run / "best.json", best)
        if rank == 0:
            write(
                run
                / (
                    "smoke_training_status.json"
                    if args.smoke
                    else "training_status.json"
                ),
                info,
            )
            if not args.smoke:
                with (run / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(info) + "\n")
        if args.smoke or step % 100 in (0, 1) or "validation" in info:
            log("STEP", **info)

    discriminator.requires_grad_(True)
    if args.smoke:
        assert g_updates == 4 and d_updates == 2
        if rank == 0:
            import tempfile

            with tempfile.TemporaryDirectory(
                prefix="native_gan_reload_", dir=run
            ) as temp:
                model.save_pretrained(temp)
                restored = (
                    CompressiveVQModelFSQ.from_pretrained(temp, local_files_only=True)
                    .eval()
                    .to(device)
                )
                assert all(
                    torch.equal(t, restored.state_dict()[n])
                    for n, t in model.state_dict().items()
                )
                model.eval()
                with torch.inference_mode():
                    expected, actual = model.tokenize(probe), restored.tokenize(probe)
                    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
                    torch.testing.assert_close(
                        model.detokenize(*expected),
                        restored.detokenize(*actual),
                        rtol=0,
                        atol=1e-4,
                    )
            write(
                run / "gpu_preflight.json",
                {
                    "state": "PASSED",
                    "world_size": world,
                    "shared_steps": steps,
                    "generator_updates": g_updates,
                    "discriminator_updates": d_updates,
                    "real_gan_gp_updates": True,
                    "synchronized_updates": True,
                    "native_reload": True,
                    "clip_consistent_augmentation": True,
                    "final": info,
                },
            )
        dist.barrier()
    else:
        assert g_updates == 35000 and d_updates == 25000
        log("FINAL_EVALUATION", final_shared_step=step, best=best)
        final_metrics = evaluation(model, dev, details=True)
        if rank == 0:
            write(
                run / "final_dev_evaluation.json",
                {
                    "final": final_metrics,
                    "best_monitored": best,
                    "episodes": len(dev),
                    "scope": "development, not untouched test",
                },
            )
            write(
                run / "complete.json",
                {
                    "state": "SUCCEEDED",
                    "shared_steps": step,
                    "generator_updates": g_updates,
                    "discriminator_updates": d_updates,
                    "final_checkpoint": str(run / "checkpoints" / f"step_{step:06d}"),
                    "best": best,
                    "initial": initial,
                    "final": info,
                    "seconds": time.monotonic() - started,
                    "world_size": world,
                },
            )
    log(
        "COMPLETE",
        shared_steps=step,
        generator_updates=g_updates,
        discriminator_updates=d_updates,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import sys, traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
