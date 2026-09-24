"""One-command workflows with explicit roots, dependencies, and durable receipts."""

import argparse, json, os, subprocess, sys, hashlib
from pathlib import Path
from .data import convert_manifest, write_json
from .checkpoints import file_sha256


def run(config_path, suite, dry_run=False):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    package = Path(__file__).resolve().parents[2]
    # Source checkout stores recipes beside src. Installed users pass recipe_root.
    recipe_root = Path(cfg.get("recipe_root", package / "configs")).resolve()
    if suite in ("main-table", "matched-200", "memory"):
        from .cli import reproduce

        recipe = recipe_root / "recipes" / f"{suite}.json"
        targets = cfg.get("datasets", ["rt1", "bridge", "robocasa"])
        for ds in targets:
            dest = Path(cfg["data"]) / ds / "evaluation"
            if not (dest / "manifest.json").is_file():
                raw = cfg.get("sources", {}).get(ds)
                if raw is None:
                    raise ValueError(f"Missing prepared {ds} data and sources.{ds}")
                ids = json.loads(
                    (recipe_root / "cohorts" / f"{ds}_128.json").read_text()
                )
                if dry_run:
                    print(f"PREPARE {ds}: {raw['manifest']} -> {dest}")
                else:
                    convert_manifest(
                        ds, raw["manifest"], dest, ids, raw.get("path_map")
                    )
        args = argparse.Namespace(
            recipe=str(recipe),
            bundle=cfg["bundle"],
            data=cfg["data"],
            output=str(Path(cfg["output"]) / suite),
            text_model=cfg.get("text_model"),
            gpus=cfg.get("gpus", 1),
            datasets=targets,
            dry_run=dry_run,
            limit=cfg.get("limit"),
        )
        reproduce(args)
    elif suite == "train" and "training" in cfg:
        training_workflow(cfg, recipe_root, dry_run)
    elif suite == "train":
        allowed = {
            "prepare",
            "cache",
            "train",
            "train-codec",
            "export-model",
            "evaluate",
        }
        for i, stage in enumerate(cfg["training_stages"]):
            command = stage["command"]
            if command not in allowed:
                raise ValueError(f"Unsupported stage: {command}")
            args = [str(v).format(**cfg["roots"]) for v in stage["args"]]
            count = stage.get("processes", 1)
            if count < 1:
                raise ValueError("Invalid process count")
            cmd = [sys.executable, "-m", "futureworlds", command] + args
            if count > 1:
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={count}",
                    "-m",
                    "futureworlds",
                    command,
                ] + args
            if dry_run:
                print(json.dumps(cmd))
                continue
            print(f"Stage {i + 1}/{len(cfg['training_stages'])}: {command}", flush=True)
            subprocess.run(cmd, check=True, env=dict(os.environ, USE_TF="0"))
    else:
        raise ValueError(suite)


def training_workflow(cfg, recipe_root, dry_run=False):
    """Prepare -> frozen-feature cache -> SFT (optional) -> RL -> evaluate."""
    jobs = cfg["training"]
    jobs = [jobs] if isinstance(jobs, dict) else jobs

    def execute(command, args, processes=1):
        cmd = [sys.executable, "-m", "futureworlds", command] + list(map(str, args))
        if processes > 1:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={processes}",
                "-m",
                "futureworlds",
                command,
            ] + list(map(str, args))
        if dry_run:
            print(json.dumps(cmd))
            return
        subprocess.run(cmd, check=True, env=dict(os.environ, USE_TF="0"))

    for job in jobs:
        ds = job["dataset"]
        method = job.get("method", "memspo")
        mode = job.get("mode", "post-only")
        if mode not in ("post-only", "sft+post"):
            raise ValueError("Invalid training mode")
        root = Path(cfg["output"]) / "training" / ds / method
        data = Path(cfg["data"]) / ds
        source = cfg.get("sources", {}).get(ds, {})
        for split in ("train", "evaluation"):
            target = data / split / "manifest.json"
            if not target.exists():
                key = "train_manifest" if split == "train" else "manifest"
                if key not in source:
                    raise ValueError(f"Missing sources.{ds}.{key}")
                if dry_run:
                    print(f"PREPARE {ds} {split}: {source[key]} -> {target}")
                else:
                    ids = (
                        None
                        if split == "train"
                        else json.loads(
                            (recipe_root / "cohorts" / f"{ds}_128.json").read_text()
                        )
                    )
                    convert_manifest(
                        ds, source[key], target.parent, ids, source.get("path_map")
                    )
        bundle = cfg["bundle"]
        processes = cfg.get("gpus", 1)
        phases = ["sft", method] if mode == "sft+post" else [method]
        for phase in phases:
            name = (
                f"{ds}_sft"
                if phase == "sft"
                else f"{ds}_main"
                if job.get("main_table", False) and phase == "memspo"
                else f"{ds}_{phase}"
            )
            recipe = json.loads((recipe_root / "train" / f"{name}.json").read_text())
            recipe.update(
                bundle=str(bundle),
                train_manifest=str(data / "train/manifest.json"),
                evaluation_manifest=str(data / "evaluation/manifest.json"),
                output=str(root / phase),
                text_model=cfg.get("text_model"),
            )
            if phase == "sft":
                if not job.get("initial_checkpoint"):
                    raise ValueError(
                        "SFT requires an explicit verified initial_checkpoint"
                    )
                recipe["initial_checkpoint"] = job["initial_checkpoint"]
            else:
                recipe["schedule"] = str(
                    recipe_root / "schedules" / f"{ds}_posttraining.json"
                )
            cache = root / (phase + "_cache")
            recipe["token_cache"] = str(cache)
            # Re-running the same workflow resumes a fully saved checkpoint.
            candidates = sorted(
                (root / phase / "checkpoints").glob(
                    "step_[0-9][0-9][0-9][0-9][0-9][0-9]"
                )
            )
            if candidates:
                recipe["resume"] = str(candidates[-1])
            path = root / (phase + ".json")
            if dry_run:
                print(json.dumps({"effective_training_config": recipe}, indent=2))
            else:
                write_json(path, recipe)
            args = [
                "--bundle",
                bundle,
                "--dataset",
                ds,
                "--manifest",
                data / "train/manifest.json",
                "--output",
                cache,
            ]
            if cfg.get("text_model"):
                args += ["--text-model", cfg["text_model"]]
            execute("cache", args, processes)
            execute("train", ["--config", path], processes)
            exported = root / (phase + "_bundle")
            execute(
                "export-model",
                [
                    "--bundle",
                    bundle,
                    "--dataset",
                    ds,
                    "--checkpoint",
                    root / phase / "checkpoints" / f"step_{recipe['steps']:06d}",
                    "--output",
                    exported,
                ],
            )
            bundle = exported
        args = [
            "--bundle",
            bundle,
            "--dataset",
            ds,
            "--variant",
            "main",
            "--manifest",
            data / "evaluation/manifest.json",
            "--output",
            root / "evaluation",
            "--device",
            "cuda",
            "--metric-device",
            "cuda",
            "--ids",
            recipe_root / "cohorts" / f"{ds}_128.json",
            "--horizons",
            10,
            20,
            32,
        ]
        if cfg.get("text_model"):
            args += ["--text-model", cfg["text_model"]]
        execute("evaluate", args, processes)
