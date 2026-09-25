"""Portable SFT / MemSPO / GRPO / ordinary-beam training with torchrun.

RL starts from a named SFT checkpoint with a fresh optimizer and frozen reference.
The checkpoint includes optimizer/RNG state; resumption never resets the cursor.
"""

import contextlib, copy, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import save_file, load_file
from .data import read_manifest, read_case, write_json, disjoint
from .checkpoints import file_sha256
from .pipeline import WorldPipeline
from .memory import make_prefix, frame_logp, rollout
from .search import rollout_group_beam
from .ordinary_search import rollout_ordinary_beam
from .objective import compute_grpo_outcome_advantage, compute_policy_loss, kl_penalty
from .rewards import VideoReward
from .motion_reward import MotionReward


def sft_example(context, dynamics, actions, target, anchor=1):
    if anchor == 0:
        kept = [0] + list(range(max(1, target - 6), target))
        prefix = torch.cat(
            [context.reshape(1, -1) + 4375]
            + [x for i in kept for x in (dynamics[i : i + 1], actions[i : i + 1])],
            1,
        )
    else:
        prefix, _ = make_prefix(
            context.reshape(1, -1) + 4375,
            {i: dynamics[i : i + 1] for i in range(1, target)},
            actions,
            target,
        )
    ids = torch.cat([prefix, dynamics[target : target + 1]], 1)
    labels = torch.full_like(ids, -100)
    labels[:, -80:] = dynamics[target : target + 1]
    return dict(
        input_ids=ids,
        labels=labels,
        attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(ids.shape[1], device=ids.device)[None],
    )


def policy_backward(
    actor,
    forward,
    reference,
    frames,
    reward,
    condition,
    horizon,
    clip,
    kl_coef,
    scale=1.0,
    sync_context=None,
):
    """One immutable set of candidate histories for old/current/reference scores."""
    n = len(reward)
    old = []
    ref = []
    with torch.no_grad():
        for f in frames:
            old.append(frame_logp(actor, f["prefix"], f["response"]))
            ref.append(frame_logp(reference, f["prefix"], f["response"]))
            saved = f.get("model_logp", f.get("sample_logp"))
            if saved is not None and float((saved - old[-1]).abs().max()) > 0.002:
                raise ValueError("Proposal and rescoring histories disagree")
        terminal = torch.zeros(n, horizon * 80, device=reward.device)
        terminal[:, -1] = reward
        adv, _ = compute_grpo_outcome_advantage(
            terminal, torch.ones_like(terminal), np.zeros(n, dtype=int)
        )
        if float(reward.std()) <= 1e-6:
            adv.zero_()
    total = 0.0
    for t, f in enumerate(frames):
        cm = sync_context(t) if sync_context else contextlib.nullcontext()
        with cm:
            ids = torch.cat([f["prefix"], f["response"][:, :-1]], 1)
            p = f["prefix"].shape[1]
            logits = (
                forward(input_ids=ids, use_cache=False, **condition)
                .logits[:, p - 1 : p + 79, :4375]
                .float()
            )
            lp = (
                logits.log_softmax(-1)
                .gather(-1, f["response"].unsqueeze(-1))
                .squeeze(-1)
            )
            if float((lp.detach() - old[t]).abs().max()) > 0.002:
                raise ValueError(
                    "Current/old likelihood mismatch before optimizer step"
                )
            pg = compute_policy_loss(
                old[t],
                lp,
                adv[:, t * 80 : (t + 1) * 80],
                torch.ones_like(lp),
                cliprange=clip,
            )[0]
            kl = kl_penalty(lp, ref[t], "low_var_kl").mean()
            loss = (pg + kl_coef * kl) / horizon * scale
            loss.backward()
            total += float(loss.detach())
    return total


def save_checkpoint(
    actor, opt, folder, step, config, fingerprint, world, rank, generator
):
    # Do not publish partially written checkpoints. All ranks contribute RNG state.
    rng = dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
        generator=generator.get_state(),
        cuda=torch.cuda.get_rng_state() if next(actor.parameters()).is_cuda else None,
    )
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states = [rng]
    if rank == 0:
        folder = Path(folder)
        tmp = folder.with_name(folder.name + ".partial")
        tmp.mkdir(parents=True, exist_ok=False)
        core = actor.core if hasattr(actor, "core") else actor
        core.save_pretrained(tmp)
        if hasattr(actor, "adapters"):
            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in actor.adapters.state_dict().items()
                },
                str(tmp / "text_adapters.safetensors"),
            )
        torch.save(
            dict(
                step=step,
                optimizer=opt.state_dict(),
                rng=states,
                world_size=world,
                fingerprint=fingerprint,
            ),
            tmp / "training_state.pt",
        )
        write_json(
            tmp / "experiment.json",
            dict(
                step=step,
                config=config,
                fingerprint=fingerprint,
                core_sha256=file_sha256(tmp / "model.safetensors"),
                files={p.name: file_sha256(p) for p in tmp.iterdir() if p.is_file()},
            ),
        )
        if folder.exists():
            raise FileExistsError(folder)
        tmp.replace(folder)
    if world > 1:
        dist.barrier()


def train(config_path):
    cfg = json.loads(Path(config_path).read_text())
    method = cfg["method"]
    if method not in ("sft", "memspo", "grpo", "ordinary"):
        raise ValueError(method)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = cfg.get("device", "cuda")
    if device == "cuda":
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    if world > 1:
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    batch = cfg["global_batch"]
    if batch < world or batch % world:
        raise ValueError("global_batch must be divisible by WORLD_SIZE")
    accum = batch // world
    trainset = read_manifest(cfg["train_manifest"], cfg["dataset"])
    if any(row.get("split") != "train" for row in trainset["cases"]):
        raise ValueError("Training requires an explicitly designated train split")
    if cfg.get("evaluation_manifest"):
        disjoint(trainset, read_manifest(cfg["evaluation_manifest"], cfg["dataset"]))
    rows = trainset["cases"]
    index = {r["id"]: r for r in rows}
    # A paper-locked schedule is an explicit ordered list. Generic schedules are
    # recorded as new runs and are not silently represented as paper reproduction.
    schedule = None
    if cfg.get("schedule"):
        schedule = json.loads(Path(cfg["schedule"]).read_text())
        for v in schedule:
            if v["id"] not in index:
                raise ValueError("Schedule refers to missing training data")
        if len(schedule) < cfg["steps"] * batch:
            raise ValueError("Incomplete training schedule")
    elif cfg.get("paper_locked", False) and method != "sft":
        raise ValueError("Paper-locked training requires the original schedule")
    from .protocol import implementation, text_identity, feature_fingerprint

    effective = {k: v for k, v in cfg.items() if k not in ("resume", "output")}
    effective["implementation"] = implementation()
    effective["text_assets"] = text_identity(
        cfg["bundle"], cfg["dataset"], cfg.get("text_model")
    )
    feature_hash = feature_fingerprint(
        cfg["bundle"], cfg["dataset"], cfg.get("text_model")
    )
    effective.update(
        train_manifest_sha256=file_sha256(cfg["train_manifest"]),
        bundle_manifest_sha256=file_sha256(Path(cfg["bundle"]) / "manifest.json"),
        world_size=world,
        schedule_sha256=file_sha256(cfg["schedule"]) if schedule is not None else None,
    )
    fingerprint = hashlib.sha256(
        json.dumps(effective, sort_keys=True).encode()
    ).hexdigest()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        if (out / "run.json").exists():
            if (
                not cfg.get("resume")
                or json.loads((out / "run.json").read_text())["fingerprint"]
                != fingerprint
            ):
                raise ValueError("Use a fresh output or resume identical configuration")
        write_json(out / "run.json", dict(fingerprint=fingerprint, config=effective))
    if world > 1:
        dist.barrier()
    pipe = WorldPipeline(
        cfg["bundle"],
        cfg["dataset"],
        cfg.get("initial_variant", "sft"),
        device,
        cfg.get("text_model"),
    )
    actor = pipe.model
    if cfg.get("initial_checkpoint"):
        from transformers import AutoConfig, AutoModelForCausalLM
        from .text_conditioning import TextWorldModel

        initial = Path(cfg["initial_checkpoint"])
        if file_sha256(initial / "model.safetensors") != cfg["initial_sha256"]:
            raise ValueError("Initial checkpoint hash mismatch")
        core = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(initial, local_files_only=True),
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        core.load_state_dict(load_file(str(initial / "model.safetensors")), strict=True)
        if pipe.settings["text_conditioning"]["enabled"]:
            tc = pipe.settings["text_conditioning"]
            actor = TextWorldModel(
                core, text_width=tc["width"], layers=tc["layers"], inner=tc["inner"]
            )
            adapter = initial / "text_adapters.safetensors"
            if adapter.exists():
                actor.adapters.load_state_dict(load_file(str(adapter)), strict=True)
            elif method != "sft" or not cfg.get("initialize_text_adapters", False):
                raise ValueError("Missing trained text adapters")
        else:
            actor = core
        pipe.model = actor = actor.to(device)
    reference = (
        copy.deepcopy(actor).eval().requires_grad_(False) if method != "sft" else None
    )
    if method == "sft" and hasattr(actor, "adapters"):
        groups = [
            dict(params=actor.core.parameters(), lr=cfg["lr"]),
            dict(params=actor.adapters.parameters(), lr=cfg["adapter_lr"]),
        ]
    else:
        groups = actor.parameters()
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=0)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    step0 = 0
    if cfg.get("resume"):
        folder = Path(cfg["resume"])
        receipt = json.loads((folder / "experiment.json").read_text())
        if any(file_sha256(folder / k) != v for k, v in receipt["files"].items()):
            raise ValueError("Checkpoint checksum mismatch")
        # training_state is a local trusted artifact produced by this command.
        state = torch.load(
            folder / "training_state.pt", map_location="cpu", weights_only=False
        )
        if state["fingerprint"] != fingerprint or state["world_size"] != world:
            raise ValueError("Resume protocol/world-size mismatch")
        (actor.core if hasattr(actor, "core") else actor).load_state_dict(
            load_file(str(folder / "model.safetensors")), strict=True
        )
        if hasattr(actor, "adapters"):
            actor.adapters.load_state_dict(
                load_file(str(folder / "text_adapters.safetensors")), strict=True
            )
        opt.load_state_dict(state["optimizer"])
        step0 = state["step"]
        s = state["rng"][rank]
        torch.set_rng_state(s["torch"])
        np.random.set_state(s["numpy"])
        random.setstate(s["python"])
        generator.set_state(s["generator"])
        if s["cuda"] is not None:
            torch.cuda.set_rng_state(s["cuda"])
    forward = (
        DDP(
            actor,
            device_ids=[local] if device.startswith("cuda") else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if world > 1
        else actor
    )
    if method != "sft":
        import lpips

        perceptual = lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)
        if cfg["reward"] == "R1":
            reward_fn = MotionReward(
                pipe.codec,
                perceptual,
                cfg["reward_scales"],
                cfg.get("alpha", 0.5),
                cfg.get("beta", 0.25),
            )
        else:
            reward_fn = VideoReward(
                pipe.codec, perceptual, cfg["horizon"], action_dim=pipe.action_dim
            )
    from .sampling import Schedule

    sampler = Schedule(rows, cfg["dataset"], seed)
    if method == "sft" and cfg["dataset"] == "bridge":
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for step in range(step0 + 1, cfg["steps"] + 1):
        started = time.monotonic()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        meaningful = 0
        # Keep transformer dropout disabled just as in the original RL worker.
        actor.train(method == "sft")
        if method == "sft":
            warm = cfg.get("warmup_steps", 500)
            progress = max(0, step - warm) / max(1, cfg["steps"] - warm)
            scale = min(step / max(1, warm), 1.0) * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            )
            if cfg["dataset"] == "bridge":
                scale = min(step / max(1, warm), 1.0)
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = (cfg["lr"] if gi == 0 else cfg["adapter_lr"]) * scale
        for micro in range(accum):
            cursor = (step - 1) * batch + micro * world + rank
            if schedule is not None:
                entry = schedule[cursor]
                row = index[entry["id"]]
            else:
                row, target = sampler.pick(cursor)
                entry = {"target": target}
            images, actions, text = read_case(
                cfg["train_manifest"],
                row,
                pipe.action_dim,
                cfg["horizon"] if method != "sft" else None,
            )
            if cfg.get("token_cache"):
                from .cache import load_cached

                ctx, dynamics, acts, cond = load_cached(
                    cfg["token_cache"], row, feature_hash, device
                )
                pixels = (
                    torch.from_numpy(images)
                    .permute(0, 3, 1, 2)
                    .to(device)
                    .float()[None]
                    / 255
                )
                if cond:
                    actor.set_text(cond["text_features"], cond["text_mask"])
            else:
                ctx, dynamics, acts, pixels, cond = pipe.encode_training_window(
                    images, actions, text
                )
            if method == "sft":
                target = entry.get(
                    "target",
                    int(
                        np.random.default_rng(seed + cursor * 9973).integers(
                            2, len(dynamics)
                        )
                    ),
                )
                if (
                    not (1 if cfg["dataset"] == "bridge" else 2)
                    <= target
                    < len(dynamics)
                ):
                    raise ValueError("Invalid SFT target")
                batch_input = sft_example(
                    ctx,
                    dynamics,
                    acts,
                    target,
                    anchor=0 if cfg["dataset"] == "bridge" else 1,
                )
                sync = (
                    forward.no_sync()
                    if world > 1 and micro < accum - 1
                    else contextlib.nullcontext()
                )
                with sync:
                    with torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=device.startswith("cuda") and cfg.get("sft_bf16", True),
                    ):
                        loss = (
                            forward(**batch_input, **cond, use_cache=False).loss / accum
                        )
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite SFT loss")
                    loss.backward()
                    total += float(loss.detach())
            else:
                if cond:
                    reference.set_text(cond["text_features"], cond["text_mask"])
                args = (
                    actor,
                    ctx.reshape(1, -1) + 4375,
                    dynamics[1:2],
                    acts,
                    cfg["horizon"],
                )
                if method == "grpo":
                    virtual_rank = cursor % cfg.get("paper_world_size", 16)
                    sample_seed = (
                        seed + (step - 1) * 1000 + virtual_rank
                        if cfg["dataset"] == "bridge"
                        else seed + virtual_rank * 1000003 + step
                    )
                    generator.manual_seed(sample_seed)
                    frames, trace = rollout(*args, group=4, generator=generator)
                else:
                    proposer = (
                        rollout_group_beam
                        if method == "memspo"
                        else rollout_ordinary_beam
                    )
                    frames, trace = proposer(
                        *args,
                        penalty=cfg.get("diversity_penalty", 0.1),
                        groups=4,
                        beams=2,
                    )
                reward, stats = reward_fn(frames, ctx, acts, pixels)
                meaningful += int(float(reward.std()) > 1e-6)

                def sync_context(t):
                    return (
                        forward.no_sync()
                        if world > 1 and (micro < accum - 1 or t < cfg["horizon"] - 1)
                        else contextlib.nullcontext()
                    )

                total += policy_backward(
                    actor,
                    forward,
                    reference,
                    frames,
                    reward,
                    cond,
                    cfg["horizon"],
                    cfg["clip"],
                    cfg["kl_coef"],
                    1 / accum,
                    sync_context,
                )
        if method != "sft":
            count = torch.tensor(meaningful, device=device)
            if world > 1:
                dist.all_reduce(count)
            if int(count) == 0:
                raise ValueError("All candidate rewards tie in this global batch")
            if any(p.grad is not None for p in reference.parameters()) or any(
                p.grad is not None for p in pipe.codec.parameters()
            ):
                raise ValueError("Frozen model receives gradients")
        norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), 1.0, error_if_nonfinite=True
        )
        opt.step()
        value = torch.tensor(total, device=device)
        if world > 1:
            dist.all_reduce(value)
            value /= world
        if rank == 0:
            record = dict(
                step=step,
                loss=float(value),
                grad_norm=float(norm),
                seconds=time.monotonic() - started,
                method=method,
            )
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % cfg.get("checkpoint_every", 50) == 0 or step == cfg["steps"]:
            save_checkpoint(
                actor,
                opt,
                out / "checkpoints" / f"step_{step:06d}",
                step,
                cfg,
                fingerprint,
                world,
                rank,
                generator,
            )
    if rank == 0:
        write_json(
            out / "complete.json",
            dict(state="COMPLETE", steps=cfg["steps"], fingerprint=fingerprint),
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
