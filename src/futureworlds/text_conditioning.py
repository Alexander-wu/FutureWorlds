"""Pretrained-text conditioning with zero-initialized residual cross-attention.

Frozen T5 features are inputs, not vocabulary IDs. Visual/action positions and
vocabulary stay unchanged. No-text requests bypass the added residual exactly.
"""

import torch

_UNSET = object()
from torch import nn
import torch.nn.functional as F


class TextResidual(nn.Module):
    def __init__(self, hidden=768, text_width=768, inner=256, heads=4):
        super().__init__()
        self.heads = heads
        self.inner = inner
        self.query_norm = nn.LayerNorm(hidden)
        self.text_norm = nn.LayerNorm(text_width)
        self.q = nn.Linear(hidden, inner, bias=False)
        self.k = nn.Linear(text_width, inner, bias=False)
        self.v = nn.Linear(text_width, inner, bias=False)
        self.out = nn.Linear(inner, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, hidden, text, mask):
        b, t, _ = hidden.shape
        if text.shape[0] != b:
            assert text.shape[0] == 1, (
                "Generation expansion is valid only within a single episode"
            )
            text = text.expand(b, -1, -1)
            mask = mask.expand(b, -1)
        dim = self.inner // self.heads
        q = self.q(self.query_norm(hidden)).view(b, t, self.heads, dim).transpose(1, 2)
        memory = self.text_norm(text)
        k = self.k(memory).view(b, -1, self.heads, dim).transpose(1, 2)
        v = self.v(memory).view(b, -1, self.heads, dim).transpose(1, 2)
        value = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None, None, :].bool(), dropout_p=0.0
        )
        value = value.transpose(1, 2).reshape(b, t, self.inner)
        # Explicitly suppress all-padding / deliberately dropped instruction.
        value = self.out(value) * mask.any(-1)[:, None, None].to(value.dtype)
        return hidden + value.to(hidden.dtype)


class TextWorldModel(nn.Module):
    def __init__(self, core, text_width=768, layers=(3, 7, 11), inner=256):
        super().__init__()
        self.core = core
        self.layers = list(layers)
        self.adapters = nn.ModuleDict(
            {
                str(i): TextResidual(core.config.hidden_size, text_width, inner)
                for i in layers
            }
        )
        self._text = None
        self._mask = None
        self._handles = []
        for i in layers:
            self._handles.append(
                core.model.layers[i].register_forward_pre_hook(
                    self._hook(i), with_kwargs=True
                )
            )

    def _hook(self, index):
        def hook(module, args, kwargs):
            if self._text is None:
                return args, kwargs
            adapter = self.adapters[str(index)]
            if args:
                return (adapter(args[0], self._text, self._mask),) + args[1:], kwargs
            kw = dict(kwargs)
            kw["hidden_states"] = adapter(kw["hidden_states"], self._text, self._mask)
            return args, kw

        return hook

    def set_text(self, text, mask):
        assert (text is None) == (mask is None)
        if text is not None:
            assert text.ndim == 3 and mask.shape == text.shape[:2]
            assert torch.isfinite(text).all()
        self._text = text
        self._mask = mask

    def forward(self, text_features=_UNSET, text_mask=_UNSET, **kwargs):
        # Generation and selected-token scoring use the explicitly bound episode.
        # No activation checkpointing: mutable conditioning cannot change during replay.
        if text_features is not _UNSET:
            assert text_mask is not _UNSET
            self.set_text(text_features, text_mask)
        else:
            assert text_mask is _UNSET
        return self.core(**kwargs)

    def generate(self, *args, **kwargs):
        # Caller sets the same episode instruction before every generated frame.
        return self.core.generate(*args, **kwargs)

    def compute_transition_scores(self, *args, **kwargs):
        return self.core.compute_transition_scores(*args, **kwargs)
