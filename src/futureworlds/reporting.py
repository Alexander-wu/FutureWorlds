"""Aggregate completed experiment receipts, never fill missing model results."""

import csv, json
from pathlib import Path


def report(root):
    root = Path(root)
    rows = []
    for p in sorted(root.glob("*/*/*/summary.json")):
        value = json.loads(p.read_text())
        if value.get("state") != "COMPLETE":
            raise ValueError(f"Incomplete: {p}")
        protocol = value["protocol"]
        for horizon, metrics in value["aggregate"].items():
            rows.append(
                dict(
                    dataset=protocol["dataset"],
                    variant=protocol["variant"],
                    memory=protocol["memory"],
                    frames=int(horizon),
                    cases=len(protocol["cases"]),
                    subset=protocol["subset"],
                    **metrics,
                )
            )
    if not rows:
        raise ValueError("No complete summaries")
    keys = list(rows[0])
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "| Dataset | Model | Memory | Frames | N | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        text.append(
            f"| {r['dataset']} | {r['variant']} | {r['memory']} | {r['frames']} | {r['cases']} | {r['psnr']:.4f} | {r['ssim']:.6f} | {r['lpips']:.6f} |"
        )
    (root / "results.md").write_text("\n".join(text) + "\n")
    return rows
