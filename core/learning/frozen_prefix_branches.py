"""Reuse frozen causal states across token-identical continuation branches."""

from __future__ import annotations

import copy
import hashlib
import json

import mlx.core as mx

from core.brain.llm.decoder_topology import decoder_backbone_owner, decoder_layer_masks
from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix


def native_source_anchor(sequences):
    """Use only the common, offset-bound source/template tokens of one request."""
    rows = tuple(sequences)
    if not rows or any(type(row.continuation_start) is not int
                       or not 0 < row.continuation_start < len(row.tokens) for row in rows):
        raise ValueError("native source anchor lacks a causal continuation boundary")
    boundary = min(row.continuation_start for row in rows)
    anchor = tuple(rows[0].tokens[:boundary])
    if any(tuple(row.tokens[:boundary]) != anchor for row in rows):
        raise ValueError("native source anchor differs between continuation branches")
    return anchor


class FrozenPrefixBranches:
    """One immutable source anchor, with an independently owned cache per branch.

    The loaded decoder constructs its own cache types. Every returned sequence
    includes the complete anchor hidden states and the causal continuation.
    No suffix layers, output decisions, or source tokens are skipped.
    """

    def __init__(self, model, *, split_at, anchor_tokens, max_tokens):
        from mlx_lm.models.cache import make_prompt_cache

        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("frozen branch token bound must be positive")
        self.max_tokens = max_tokens
        self.anchor_tokens = self._tokens(anchor_tokens)
        self.prefix = FrozenDecoderPrefix(model, split_at=split_at)
        self.cache = make_prompt_cache(decoder_backbone_owner(model))[:split_at]
        if len(self.cache) != split_at or any(entry is None for entry in self.cache):
            raise ValueError("frozen branches require the decoder's complete cache inventory")
        self.anchor_hidden = self._advance(self.anchor_tokens, self.cache)
        self.branches = self.reused_tokens = self.branch_tokens = 0

    def _tokens(self, tokens):
        if (not isinstance(tokens, tuple) or not tokens or len(tokens) > self.max_tokens
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("frozen branches require a nonempty bounded token tuple")
        return tokens

    def _advance(self, tokens, cache):
        self.prefix._assert_frozen()
        hidden = self.prefix.embedding(mx.array([tokens], dtype=mx.int32))
        masks = decoder_layer_masks(self.prefix.backbone, hidden, cache, end=self.prefix.split_at)
        for layer, mask, entry in zip(self.prefix.layers, masks, cache, strict=True):
            hidden = layer(hidden, mask=mask, cache=entry)
        hidden = mx.stop_gradient(hidden)
        mx.eval(hidden, [entry.state for entry in cache])
        return hidden

    def capture(self, tokens):
        tokens = self._tokens(tokens)
        self.prefix._assert_frozen()
        boundary = len(self.anchor_tokens)
        if tokens[:boundary] != self.anchor_tokens:
            raise ValueError("frozen branch does not share the exact token anchor")
        continuation = tokens[boundary:]
        # The runtime prompt cache uses the same deep-copy ownership contract.
        # Reusing mutable K/V capacity would let one alternative alter another.
        result = (mx.array(self.anchor_hidden) if not continuation else mx.concatenate((
            self.anchor_hidden, self._advance(continuation, copy.deepcopy(self.cache))), axis=1))
        mx.eval(result)
        self.branches += 1
        self.reused_tokens += boundary
        self.branch_tokens += len(continuation)
        return result

    def receipt(self):
        return {"schema": "aura.frozen_prefix_branches.v1",
                "anchor_token_sha256": hashlib.sha256(json.dumps(
                    self.anchor_tokens, separators=(",", ":")).encode("ascii")).hexdigest(),
                "anchor_tokens": len(self.anchor_tokens), "branches": self.branches,
                "reused_tokens": self.reused_tokens, "branch_tokens": self.branch_tokens,
                "executed_tokens": len(self.anchor_tokens) + self.branch_tokens,
                "uncached_tokens": self.reused_tokens + self.branch_tokens,
                "cache_types": [type(entry).__name__ for entry in self.cache],
                "suffix_computation_unchanged": True, "serving_authority": False}
