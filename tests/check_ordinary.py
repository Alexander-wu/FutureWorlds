"""Independent uncached search verifies full-video ancestry, scores and gradients."""

import copy, io, json
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from futureworlds.ordinary_search import rollout_ordinary_beam
from futureworlds.memory import frame_logp
from futureworlds.objective import compute_policy_loss


def run_tests():
    torch.set_num_threads(2)
    torch.manual_seed(120)
    m = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=60,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    ctx = torch.tensor([[21, 22, 23, 24]])
    anchor = torch.tensor([[1, 2]])
    actions = torch.arange(12)[:, None] + 30
    width = 2
    visual = 7
    horizon = 10
    checks = []
    for beam in [8]:
        frames, report = rollout_ordinary_beam(
            m, ctx, anchor, actions, horizon, 0, visual=visual, width=width
        )
        # Python lists and independent prefix assembly, no KV cache and no tensor ancestry reuse.
        states = [([], [], 0.0)]
        with torch.no_grad():
            for t in range(2, horizon + 2):
                states = [(movie, [], score) for movie, _, score in states]
                for k in range(width):
                    expanded = []
                    for movie, current, score in states:
                        prompt = ctx[0].tolist()
                        for old_t in [1] + list(range(max(2, t - 6), t)):
                            prompt += (
                                anchor[0].tolist() if old_t == 1 else movie[old_t - 2]
                            ) + actions[old_t].tolist()
                        lp = (
                            m(
                                input_ids=torch.tensor([prompt + current]),
                                use_cache=False,
                            )
                            .logits[0, -1, :visual]
                            .log_softmax(-1)
                        )
                        for v in range(visual):
                            expanded.append(
                                (movie, current + [v], score + float(lp[v]))
                            )
                    states = sorted(expanded, key=lambda x: x[2], reverse=True)[:beam]
                states = [
                    (movie + [current], current, score)
                    for movie, current, score in states
                ]
        states = states[:4]
        expected = torch.tensor([x[0] for x in states])
        actual = torch.stack([f["response"] for f in frames], 1)
        assert torch.equal(expected, actual)
        error = max(
            float(
                (
                    frame_logp(m, f["prefix"], f["response"], visual=visual)
                    - f["model_logp"]
                )
                .abs()
                .max()
            )
            for f in frames
        )
        assert error < 2e-5
        assert (
            max(abs(a - b[2]) for a, b in zip(report["model_log_scores"], states))
            < 2e-4
        )
        checks.append(
            {
                "width": beam,
                "independent_search_and_causal_ancestry_exact": True,
                "max_logprob_error": error,
            }
        )
    m.zero_grad()
    f = frames[-1]
    old = frame_logp(m, f["prefix"], f["response"], visual=visual).detach()
    adv = torch.tensor([-1.5, -0.5, 0.5, 1.5])[:, None].expand_as(old)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
    loss = compute_policy_loss(
        old,
        frame_logp(m, f["prefix"], f["response"], visual=visual),
        adv,
        torch.ones_like(old),
        cliprange=0.2,
    )[0]
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    assert norm > 0
    optimizer.step()
    new = frame_logp(m, f["prefix"], f["response"], visual=visual).detach()
    direction = float(((new - old) * adv).mean())
    assert direction > 0
    return {
        "beam_tests": checks,
        "positive_advantage_update_direction": direction,
        "no_multinomial_sampling": True,
        "experimental_biased_surrogate_not_on_policy_grpo": True,
    }


if __name__ == "__main__":
    print("BEAM_TESTS=" + json.dumps(run_tests()), flush=True)
