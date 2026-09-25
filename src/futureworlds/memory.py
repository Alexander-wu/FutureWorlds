"""Causal ReCAP prompts shared by rollout and every policy log-probability pass."""

import torch


def retained(target, recent=6):
    # State 1 is the permanent dynamic anchor; state 0 is encoded in context.
    assert target >= 2 and recent >= 1
    return [1] + list(range(max(2, target - recent), target))


def make_prefix(context, history, actions, target, recent=6):
    # All actions are already tokenized. actions[t] follows retained state t.
    # The caller owns the dataset-specific raw-action/time-index conversion.
    kept = retained(target, recent)
    assert max(kept) == target - 1 and len(kept) == len(set(kept))
    blocks = [context]
    for t in kept:
        blocks += [history[t], actions[t].expand(context.shape[0], -1)]
    return torch.cat(blocks, dim=1), kept


@torch.no_grad()
def rollout(
    model,
    context,
    anchor,
    actions,
    horizon,
    group,
    generator,
    visual=4375,
    width=80,
    temperature=1.0,
):
    model.eval()
    context = context.expand(group, -1)
    history = {1: anchor.expand(group, -1)}
    frames = []
    traces = []
    for target in range(2, horizon + 2):
        prefix, kept = make_prefix(context, history, actions, target)
        pending = prefix
        kv = None
        codes = []
        lps = []
        for token in range(width):
            out = model(input_ids=pending, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            logp = torch.log_softmax(
                out.logits[:, -1, :visual].float() / temperature, -1
            )
            pending = torch.multinomial(logp.exp(), 1, generator=generator)
            codes.append(pending)
            lps.append(logp.gather(1, pending).squeeze(1))
        response = torch.cat(codes, 1)
        frames.append(
            {
                "prefix": prefix.detach(),
                "response": response.detach(),
                "sample_logp": torch.stack(lps, 1).detach(),
                "target": target,
            }
        )
        history[target] = response
        traces.append(
            {
                "target": target,
                "retained": kept,
                "prefix_length": prefix.shape[1],
                "positions_reset": True,
                "kv_reset": True,
                "future_observations": False,
            }
        )
    return frames, traces


def frame_logp(model, prefix, response, visual=4375, temperature=1.0):
    # Each target has its own rebased prefix. No discarded history is hidden in a cache.
    ids = torch.cat([prefix, response[:, :-1]], 1)
    out = model(input_ids=ids, use_cache=False)
    p = prefix.shape[1]
    logits = (
        out.logits[:, p - 1 : p + response.shape[1] - 1, :visual].float() / temperature
    )
    return torch.log_softmax(logits, -1).gather(-1, response.unsqueeze(-1)).squeeze(-1)
