"""Full-shape causal sharing retains targets, ordering, and single-row arithmetic."""

from dataclasses import replace

import mlx.core as mx
import pytest

from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
from core.learning.semantic_native_causal_groups import (
    native_causal_groups,
    score_native_causal_groups,
)
from core.learning.semantic_native_program import NativeProgramSequence
from tests.test_frozen_decoder_prefix import _model
from tools.train_semantic_native_program import native_loss


def sequences():
    return (NativeProgramSequence((1, 2, 3, 4, 5), 2, semantic_positions=(3,)),
            NativeProgramSequence((1, 2, 3, 8, 9), 2, semantic_positions=(3,)),
            NativeProgramSequence((1, 2, 7, 4, 5), 2, semantic_positions=(3,)),
            NativeProgramSequence((1, 2, 3, 4, 5, 6), 2, semantic_positions=(3,)))


def test_grouping_retains_length_source_alignment_and_every_causal_predecessor():
    rows = sequences()
    groups = native_causal_groups(rows)
    assert [group.members for group in groups] == [(0, 1), (2,), (3,)]
    assert groups[0].prediction_positions == (2,)
    assert native_causal_groups((replace(rows[0], semantic_positions=(2, 3)), rows[1]))[0].members == (0,)
    assert native_causal_groups((rows[0], replace(rows[1], continuation_start=3)))[0].members == (0,)


@pytest.mark.parametrize("hybrid,tied", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("quantized", [False, True])
def test_shared_scores_match_each_original_full_sequence(hybrid, tied, quantized):
    import mlx.nn as nn

    model = _model(hybrid=hybrid, tied=tied, width=32)
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
    model.freeze()
    model.eval()
    split = len(model.layers) - 1
    prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
    rows = sequences()
    expected = tuple(-native_loss(suffix, prefix.capture(mx.array([row.tokens[:-1]])), row,
                                 summed=True, scope="semantic_decisions").item() for row in rows)
    observed, receipt = score_native_causal_groups(prefix, suffix, rows)
    assert observed == expected
    assert receipt["full_single_row_forwards"] == 3
    assert receipt["shared_forwards"] == 1
    assert receipt["numeric_equivalence_measured"] is False
    assert receipt["serving_authority"] is False
    reordered, _ = score_native_causal_groups(prefix, suffix, (rows[1], rows[0]))
    assert reordered == (expected[1], expected[0])


def test_multitoken_targets_share_only_when_all_previous_target_tokens_agree():
    rows = sequences()
    multi = tuple(replace(row, semantic_positions=(2, 3)) for row in rows[:3])
    assert [group.members for group in native_causal_groups(multi)] == [(0, 1), (2,)]
    changed_previous_target = replace(multi[1], tokens=(1, 2, 9, 8, 5))
    assert len(native_causal_groups((multi[0], changed_previous_target))) == 2


@pytest.mark.parametrize("last_target", [31, 32, 63, 64, 127])
def test_quantized_hybrid_keeps_causal_scores_across_recurrent_chunks(last_target):
    import mlx.nn as nn

    model = _model(hybrid=True, width=32, hybrid_layers=8)
    nn.quantize(model, group_size=32, bits=4)
    model.freeze()
    model.eval()
    prefix, suffix = FrozenDecoderPrefix(model, split_at=6), NativeDecoderSuffix(model, split_at=6)
    original = tuple((index * 7 + 3) % 32 for index in range(132))
    changed = list(original)
    changed[last_target] = (changed[last_target] + 1) % 32
    changed[-2] = (changed[-2] + 3) % 32
    rows = (NativeProgramSequence(original, 3, semantic_positions=(last_target,)),
            NativeProgramSequence(tuple(changed), 3, semantic_positions=(last_target,)))
    assert len(native_causal_groups(rows)) == 1
    expected = tuple(-native_loss(suffix, prefix.capture(mx.array([row.tokens[:-1]])), row,
                                 summed=True, scope="semantic_decisions").item() for row in rows)
    observed, receipt = score_native_causal_groups(prefix, suffix, rows)
    assert observed == expected
    assert receipt["full_single_row_forwards"] == 1


@pytest.mark.parametrize("change", [{"tokens": (1, True, 2, 3, 4)},
                                    {"semantic_positions": ()},
                                    {"semantic_positions": (3, 2)},
                                    {"semantic_positions": (1,)},
                                    {"semantic_positions": (5,)},
                                    {"continuation_start": True}])
def test_invalid_alignment_is_not_grouped(change):
    with pytest.raises(ValueError, match="boundary"):
        native_causal_groups((replace(sequences()[0], **change),))


def test_empty_or_unbounded_grouping_is_refused():
    for rows in ((), [], sequences() * 257):
        with pytest.raises(ValueError, match="bounded"):
            native_causal_groups(rows)


def test_suffix_training_cannot_share_a_stochastic_forward():
    model = _model()
    prefix, suffix = FrozenDecoderPrefix(model, split_at=2), NativeDecoderSuffix(model, split_at=2)
    suffix.layers[0].train()
    with pytest.raises(ValueError, match="evaluation"):
        score_native_causal_groups(prefix, suffix, sequences())


def test_nested_suffix_module_must_also_be_in_evaluation_mode():
    model = _model()
    prefix, suffix = FrozenDecoderPrefix(model, split_at=2), NativeDecoderSuffix(model, split_at=2)
    suffix.layers[0].mlp.down_proj.train()
    with pytest.raises(ValueError, match="evaluation"):
        score_native_causal_groups(prefix, suffix, sequences())
