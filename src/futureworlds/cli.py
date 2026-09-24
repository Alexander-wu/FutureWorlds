"""Command-line entry point; imports GPU/metric dependencies only on execution."""

import argparse, json, os, subprocess, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(prog="futureworlds")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("run")
    q.add_argument("--config", required=True)
    q.add_argument(
        "--suite",
        required=True,
        choices=["main-table", "matched-200", "memory", "train"],
    )
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("export-model")
    q.add_argument("--bundle", required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("prepare", help="Losslessly convert audited raw-window JSONL")
    q.add_argument("--dataset", required=True, choices=["rt1", "bridge", "robocasa"])
    q.add_argument("--source", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--ids")
    q.add_argument("--path-map")
    q.add_argument("--limit", type=int)
    q = sub.add_parser("evaluate")
    q.add_argument("--bundle", required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--variant", default="main")
    q.add_argument("--manifest", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--device", default="cuda")
    q.add_argument("--metric-device")
    q.add_argument("--text-model")
    q.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 32])
    q.add_argument("--beam-width", type=int, default=4)
    q.add_argument("--memory", choices=["full", "recent", "minimal"], default="full")
    q.add_argument("--ids")
    q.add_argument("--limit", type=int)
    q = sub.add_parser("cache")
    q.add_argument("--bundle", required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--manifest", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--device", default="cuda")
    q.add_argument("--text-model")
    q = sub.add_parser("train-codec")
    q.add_argument("--dataset", required=True, choices=["bridge", "robocasa"])
    q.add_argument("--config", required=True)
    q.add_argument("--smoke", action="store_true")
    q = sub.add_parser("train")
    q.add_argument("--config", required=True)
    q = sub.add_parser("reproduce")
    q.add_argument("--recipe", required=True)
    q.add_argument("--bundle", required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--text-model")
    q.add_argument("--gpus", type=int, default=1)
    q.add_argument("--datasets", nargs="+", choices=["rt1", "bridge", "robocasa"])
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--limit", type=int)
    q = sub.add_parser("download")
    q.add_argument("--repo-id", required=True)
    q.add_argument("--revision", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("doctor")
    q.add_argument("--bundle")
    q.add_argument("--manifest")
    args = p.parse_args()
    if args.command == "run":
        from .workflow import run

        run(args.config, args.suite, args.dry_run)
    elif args.command == "export-model":
        from .exporting import export_model

        export_model(args.bundle, args.dataset, args.checkpoint, args.output)
    elif args.command == "prepare":
        from .data import convert_manifest

        ids = json.loads(Path(args.ids).read_text()) if args.ids else None
        mapping = json.loads(Path(args.path_map).read_text()) if args.path_map else None
        print(
            convert_manifest(
                args.dataset, args.source, args.output, ids, mapping, args.limit
            )
        )
    elif args.command == "cache":
        from .cache import cache_dataset

        kw = vars(args)
        kw.pop("command")
        cache_dataset(**kw)
    elif args.command == "train-codec":
        import importlib

        module = importlib.import_module(
            "futureworlds.codec_training." + args.dataset + ".worker"
        )
        sys.argv = [module.__file__, "--config", args.config] + (
            ["--smoke"] if args.smoke else []
        )
        module.main()
    elif args.command == "train":
        from .training import train

        train(args.config)
    elif args.command == "evaluate":
        from .evaluation import evaluate

        kw = vars(args)
        kw.pop("command")
        ids = kw.pop("ids")
        kw["expected_ids"] = json.loads(Path(ids).read_text()) if ids else None
        evaluate(**kw)
    elif args.command == "reproduce":
        reproduce(args)
    elif args.command == "download":
        from huggingface_hub import snapshot_download
        from .checkpoints import verify_bundle

        snapshot_download(
            repo_id=args.repo_id, revision=args.revision, local_dir=args.output
        )
        verify_bundle(args.output)
        print("Bundle checksums verified")
    elif args.command == "doctor":
        import importlib.metadata, torch

        result = {
            "python": sys.version,
            "cuda_available": torch.cuda.is_available(),
            "packages": {},
        }
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "diffusers",
            "lpips",
            "piqa",
            "numpy",
            "einops",
            "safetensors",
        ):
            try:
                result["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["packages"][name] = "MISSING"
        if args.bundle:
            from .checkpoints import verify_bundle

            result["bundle_files"] = len(verify_bundle(args.bundle)["files"])
        if args.manifest:
            from .data import read_manifest, read_case

            m = read_manifest(args.manifest)
            dimension = 52 if m["dataset"] == "robocasa" else 13
            for r in m["cases"]:
                read_case(args.manifest, r, dimension)
            result["verified_cases"] = len(m["cases"])
        print(json.dumps(result, indent=2))


def reproduce(args):
    recipe_path = Path(args.recipe).resolve()
    recipe = json.loads(recipe_path.read_text())
    if args.gpus < 1:
        raise ValueError("--gpus must be positive")
    output = Path(args.output).resolve()
    commands = []
    for dataset, settings in recipe["datasets"].items():
        if args.datasets and dataset not in args.datasets:
            continue
        for variant in recipe["variants"]:
            for memory in recipe["memories"]:
                dest = output / dataset / variant / memory
                prefix = [sys.executable, "-m", "futureworlds"]
                if args.gpus > 1 and settings["generation_device"] == "cuda":
                    prefix = [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc_per_node={args.gpus}",
                        "-m",
                        "futureworlds",
                    ]
                cmd = (
                    prefix
                    + [
                        "evaluate",
                        "--bundle",
                        str(Path(args.bundle).resolve()),
                        "--dataset",
                        dataset,
                        "--variant",
                        variant,
                        "--manifest",
                        str(
                            Path(args.data).resolve()
                            / dataset
                            / "evaluation/manifest.json"
                        ),
                        "--output",
                        str(dest),
                        "--device",
                        settings["generation_device"],
                        "--metric-device",
                        settings["metric_device"],
                        "--ids",
                        str((recipe_path.parent / settings["cohort"]).resolve()),
                        "--memory",
                        memory,
                        "--beam-width",
                        str(recipe["beam_width"]),
                        "--horizons",
                    ]
                    + list(map(str, recipe["horizons"]))
                )
                if args.text_model:
                    cmd += ["--text-model", str(Path(args.text_model).resolve())]
                if args.limit:
                    cmd += ["--limit", str(args.limit)]
                commands.append(cmd)
    if args.dry_run:
        print(json.dumps({"recipe": recipe["name"], "commands": commands}, indent=2))
        return
    env = dict(os.environ, USE_TF="0", TOKENIZERS_PARALLELISM="false")
    for cmd in commands:
        subprocess.run(cmd, env=env, check=True)
    from .reporting import report

    report(output)


if __name__ == "__main__":
    main()
