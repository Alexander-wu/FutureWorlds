"""Inspect search and memory on a tiny CPU model; not an experiment result.

Run from the repository root:
    USE_TF=0 PYTHONPATH=src python examples/trace_search.py
"""

import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from futureworlds.memory import frame_logp
from futureworlds.search import rollout_group_beam


def main():
    torch.set_num_threads(2)
    torch.manual_seed(321)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=60,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            attention_dropout=0.0,
        )
    ).eval()

    # Tiny synthetic tokens keep this example independent of data and weights.
    context = torch.tensor([[21, 22]])
    anchor = torch.tensor([[1, 2]])
    actions = torch.arange(12)[:, None] + 30
    frames, report = rollout_group_beam(
        model,
        context,
        anchor,
        actions,
        horizon=10,
        penalty=0.1,
        groups=4,
        beams=2,
        visual=7,
        width=2,
    )

    # Score the exact candidate prefixes again without a generation cache.
    with torch.no_grad():
        error = max(
            (frame_logp(model, f["prefix"], f["response"], visual=7) - f["model_logp"])
            .abs()
            .max()
            .item()
            for f in frames
        )
    assert error < 2e-5
    print(
        json.dumps(
            {
                "scope": "synthetic CPU code walkthrough, not paper metrics",
                "active_paths": report["active_paths"],
                "returned_trajectories": report["returned_trajectories"],
                "prediction_shape": list(
                    torch.stack([f["response"] for f in frames], 1).shape
                ),
                "max_log_probability_reconstruction_error": error,
                "memory_trace": [
                    {
                        "target_state": f["target"],
                        "retained_states": f["retained"],
                        "prefix_tokens": f["prefix_length"],
                    }
                    for f in report["frames"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
