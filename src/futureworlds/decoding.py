"""Fixed-length, constrained HF beam search and independent numerical checks."""

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList


class VisualTokensOnly(LogitsProcessor):
    def __init__(self, valid_size):
        self.valid_size = valid_size

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        scores[:, self.valid_size :] = -torch.inf
        return scores


@torch.inference_mode()
def beam_decode(model, prefix, width=4, steps=80, valid_size=4375):
    config = GenerationConfig(
        do_sample=False,
        num_beams=width,
        num_return_sequences=width,
        max_new_tokens=steps,
        min_new_tokens=steps,
        eos_token_id=None,
        bos_token_id=None,
        pad_token_id=0,
        length_penalty=0.0 if width > 1 else 1.0,
        early_stopping=False,
        renormalize_logits=True,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    result = model.generate(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        generation_config=config,
        logits_processor=LogitsProcessorList([VisualTokensOnly(valid_size)]),
        eos_token_id=None,
        bos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
    )
    candidates = result.sequences[:, prefix.shape[1] :]
    assert candidates.shape == (width, steps)
    assert int(candidates.min()) >= 0 and int(candidates.max()) < valid_size
    assert len({tuple(x) for x in candidates.tolist()}) == width
    transitions = model.compute_transition_scores(
        result.sequences,
        result.scores,
        getattr(result, "beam_indices", None),
        normalize_logits=False,
    )
    assert transitions.shape == (width, steps) and torch.isfinite(transitions).all()
    sums = transitions.sum(-1)
    scores = result.sequences_scores if width > 1 else sums
    error = float((sums - scores).abs().max())
    assert torch.allclose(sums, scores, rtol=0, atol=2e-4), error
    assert torch.all(scores[:-1] >= scores[1:])
    return (
        candidates,
        scores,
        {
            "beam_width": width,
            "tokens_per_frame": steps,
            "selected_index": 0,
            "candidate_scores": scores.tolist(),
            "score_reconstruction_max_error": error,
            "score_definition": "Sum of log probabilities renormalized over valid visual tokens only; no length penalty or GT/image reward.",
        },
    )


@torch.inference_mode()
def independent_check():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(3719)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=8,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=128,
                eos_token_id=2,
                bos_token_id=1,
                pad_token_id=0,
                attention_dropout=0.0,
            )
        )
        .float()
        .eval()
    )
    prefix = torch.tensor([[6, 1, 3]], dtype=torch.long)
    reports = []
    for width in [1, 4]:
        beams = [([], 0.0)]
        for _ in range(4):
            expanded = []
            for seq, score in beams:
                x = torch.tensor([prefix[0].tolist() + seq], dtype=torch.long)
                probs = (
                    model(input_ids=x, use_cache=False)
                    .logits[0, -1, :5]
                    .log_softmax(-1)
                )
                expanded.extend(
                    (seq + [token], score + float(probs[token])) for token in range(5)
                )
            beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:width]
        ids, scores, meta = beam_decode(
            model, prefix, width=width, steps=4, valid_size=5
        )
        expected = torch.tensor([seq for seq, _ in beams], dtype=torch.long)
        expected_scores = torch.tensor([score for _, score in beams])
        assert torch.equal(ids, expected), (width, ids, expected)
        assert torch.allclose(scores, expected_scores, rtol=0, atol=2e-5)
        reports.append(
            {
                "width": width,
                "matches_independent_uncached_search": True,
                "max_score_error": float((scores - expected_scores).abs().max()),
            }
        )
    return {
        "tests": reports,
        "eos_disabled": True,
        "valid_token_constraint_checked": True,
        "complete": True,
    }


if __name__ == "__main__":
    import json, transformers

    print(json.dumps({"transformers": transformers.__version__, **independent_check()}))
