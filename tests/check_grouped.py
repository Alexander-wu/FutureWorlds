"""Independent causal search and reward invariants; CPU only."""

import json
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from futureworlds.search import rollout_group_beam
from futureworlds.memory import frame_logp
from futureworlds.motion_reward import MotionReward, motion_regions


def tests():
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
    ctx = torch.tensor([[21, 22]])
    anchor = torch.tensor([[1, 2]])
    actions = torch.arange(12)[:, None] + 30
    errors = []
    for penalty in [0.1, 1.0]:
        frames, trace = rollout_group_beam(
            model, ctx, anchor, actions, 10, penalty, visual=7, width=2
        )
        states = [[([], [], 0.0, 0.0)] for _ in range(4)]
        with torch.no_grad():
            for target in range(2, 12):
                states = [
                    [(movie, [], raw, score) for movie, _, raw, score in st]
                    for st in states
                ]
                for token in range(2):
                    chosen = []
                    for g in range(4):
                        choices = []
                        for movie, current, raw, score in states[g]:
                            prompt = ctx[0].tolist()
                            for t in [1] + list(range(max(2, target - 6), target)):
                                prompt += (
                                    anchor[0].tolist() if t == 1 else movie[t - 2]
                                ) + actions[t].tolist()
                            lp = (
                                model(
                                    input_ids=torch.tensor([prompt + current]),
                                    use_cache=False,
                                )
                                .logits[0, -1, :7]
                                .log_softmax(-1)
                            )
                            for v in range(7):
                                cost = (
                                    penalty * chosen.count(v) / len(chosen)
                                    if chosen
                                    else 0
                                )
                                choices.append(
                                    (
                                        movie,
                                        current + [v],
                                        raw + float(lp[v]),
                                        score + float(lp[v]) - cost,
                                    )
                                )
                        states[g] = sorted(choices, key=lambda z: z[3], reverse=True)[
                            :2
                        ]
                        chosen.extend(z[1][-1] for z in states[g])
                states = [
                    [
                        (movie + [current], current, raw, score)
                        for movie, current, raw, score in st
                    ]
                    for st in states
                ]
        expected = torch.tensor([st[0][0] for st in states])
        actual = torch.stack([f["response"] for f in frames], 1)
        assert torch.equal(expected, actual), (
            "Grouped beam differs from independent uncached enumeration"
        )
        err = max(
            float(
                (
                    frame_logp(model, f["prefix"], f["response"], visual=7)
                    - f["model_logp"]
                )
                .abs()
                .max()
            )
            for f in frames
        )
        assert err < 2e-5
        assert (
            max(abs(st[0][2] - v) for st, v in zip(states, trace["model_log_scores"]))
            < 2e-4
        )
        assert (
            max(abs(st[0][3] - v) for st, v in zip(states, trace["search_scores"]))
            < 2e-4
        )
        errors.append(err)

    class Percept(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).square().mean((1, 2, 3), keepdim=True)

    reward = MotionReward(None, Percept())
    pixels = torch.zeros(1, 12, 3, 80, 96)
    for t in range(12):
        pixels[0, t, :, 20:36, 10 + t * 3 : 26 + t * 3] = 0.7
    truth = pixels[:, 2:].clone()
    bad = truth.clone()
    bad[:] = pixels[:, 1:2]
    good = reward.components(truth, pixels)
    frozen = reward.components(bad, pixels)
    assert (
        good["global_loss"].max() == 0
        and good["roi_loss"].max() == 0
        and good["delta_loss"].max() == 0
    )
    assert reward.combine(good).item() == 0 and reward.combine(frozen).item() < 0
    missing = truth.clone()
    missing.zero_()
    mc = reward.components(missing, pixels)
    assert mc["roi_loss"].mean() > 0 and torch.equal(mc["roi_area"], good["roi_area"])
    first = (bad[:, 0] - pixels[:, 1]) - (truth[:, 0] - pixels[:, 1])
    assert torch.allclose(frozen["delta_loss"][:, 0], first.abs().mean((1, 2, 3)))
    masks, _, fallback = motion_regions(
        torch.zeros(10, 3, 80, 96), torch.zeros(3, 80, 96)
    )
    assert fallback.all() and masks.min() == 1
    return dict(
        grouped_search_uncached_exact=True,
        causal_eviction_ancestry_and_raw_scores=True,
        max_logprob_errors=errors,
        GT_exact_zero_error=True,
        freeze_penalized=True,
        disappearance_cannot_shrink_mask=True,
        first_delta_uses_last_real_observation=True,
        static_mask_fallback_finite=True,
    )


if __name__ == "__main__":
    print("NEW_TESTS=" + json.dumps(tests()), flush=True)
