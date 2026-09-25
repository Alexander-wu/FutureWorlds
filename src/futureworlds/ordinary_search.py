"""Global beam width 8; return top 4 by raw cumulative model score, no diversity penalty."""

import torch
from .memory import make_prefix


@torch.no_grad()
def rollout_ordinary_beam(
    model,
    context,
    anchor,
    actions,
    horizon,
    penalty,
    groups=4,
    beams=2,
    visual=4375,
    width=80,
):
    assert groups == 4 and beams == 2
    groups, beams, penalty = 1, 8, 0.0
    returned = 4
    model.eval()
    device = context.device
    count = groups * beams
    history = {1: anchor.expand(count, -1)}
    frames = []
    traces = []
    scores = torch.full((count,), float("-inf"), device=device, dtype=torch.float64)
    scores[::beams] = 0
    raw = torch.zeros(count, device=device, dtype=torch.float64)
    for target in range(2, horizon + 2):
        original, kept = make_prefix(
            context.expand(count, -1), history, actions, target
        )
        ancestors = torch.arange(count, device=device)
        codes = torch.empty((count, 0), device=device, dtype=torch.long)
        logs = torch.empty((count, 0), device=device)
        pending = original
        cache = None
        for token in range(width):
            out = model(input_ids=pending, past_key_values=cache, use_cache=True)
            lp = out.logits[:, -1, :visual].float().log_softmax(-1)
            parents = []
            choices = []
            next_scores = []
            for g in range(groups):
                start = g * beams
                if g:
                    frequency = torch.bincount(
                        torch.cat(choices), minlength=visual
                    ).double() / (g * beams)
                else:
                    frequency = torch.zeros(visual, device=device, dtype=torch.float64)
                totals = (
                    scores[start : start + beams, None]
                    + lp[start : start + beams].double()
                    - penalty * frequency
                )
                score, flat = totals.flatten().topk(beams, sorted=True)
                parents.append(start + flat // visual)
                choices.append(flat % visual)
                next_scores.append(score)
            parent = torch.cat(parents)
            chosen = torch.cat(choices)
            assert torch.equal(
                parent // beams, torch.arange(count, device=device) // beams
            )
            raw = raw[parent] + lp[parent, chosen].double()
            scores = torch.cat(next_scores)
            codes = torch.cat([codes[parent], chosen[:, None]], 1)
            logs = torch.cat([logs[parent], lp[parent, chosen, None]], 1)
            ancestors = ancestors[parent]
            cache = out.past_key_values
            cache.reorder_cache(parent)
            pending = chosen[:, None]
        for f in frames:
            for key in ["prefix", "response", "model_logp"]:
                f[key] = f[key][ancestors]
        history = {t: x[ancestors] for t, x in history.items()}
        history[target] = codes
        frames.append(
            dict(
                target=target,
                prefix=original[ancestors],
                response=codes,
                model_logp=logs,
            )
        )
        traces.append(
            dict(
                target=target,
                retained=kept,
                prefix_length=original.shape[1],
                frame_parent_indices=ancestors.tolist(),
                positions_reset=True,
                kv_reset=True,
                future_observations=False,
                random_sampling=False,
            )
        )
    selected = torch.arange(returned, device=device)
    frames = [
        {k: (v[selected] if torch.is_tensor(v) else v) for k, v in f.items()}
        for f in frames
    ]
    history = {t: x[selected] for t, x in history.items()}
    for f in frames:
        rebuilt, _ = make_prefix(
            context.expand(returned, -1), history, actions, f["target"]
        )
        assert torch.equal(rebuilt, f["prefix"])
    rebuilt = sum(f["model_logp"].double().sum(-1) for f in frames)
    assert torch.allclose(rebuilt, raw[selected], atol=1e-7, rtol=0)
    codes = torch.stack([f["response"] for f in frames], 1)
    pair = [
        float((codes[i] != codes[j]).float().mean())
        for i in range(returned)
        for j in range(i + 1, returned)
    ]
    perframe = (codes != codes[:1]).any(0).sum(-1).tolist()
    return frames, dict(
        strategy="ordinary_global_beam8_top4_full_video",
        groups=groups,
        beams_per_group=beams,
        active_paths=count,
        returned_trajectories=returned,
        penalty=penalty,
        unique_trajectories=len({tuple(c.flatten().tolist()) for c in codes}),
        mean_pairwise_token_difference_fraction=sum(pair) / len(pair),
        pairwise_token_difference_fraction=pair,
        different_token_positions=sum(perframe),
        different_positions_per_frame=perframe,
        model_log_scores=raw[selected].tolist(),
        search_scores=scores[selected].tolist(),
        score_is_behavior_logprob=False,
        on_policy_grpo=False,
        frames=traces,
    )
