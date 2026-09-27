"""Held-out forms preserve executable meaning and public source bindings."""

import pytest

from core.learning.semantic_public_inputs import semantic_public_character_inputs
from tools.semantic_native_operation_interventions import build_native_operation_interventions
from tools.semantic_native_paraphrase_interventions import (
    build_native_paraphrase_interventions,
    render_native_paraphrase,
)


@pytest.mark.parametrize("style", ["definition", "equation"])
def test_every_paraphrase_pair_preserves_program_values_and_source_intervention(style):
    base = build_native_operation_interventions(seed=2718283)
    pairs = build_native_paraphrase_interventions(seed=2718283, style=style)
    assert len(base) == len(pairs) == 24
    assert len({item.source_text for pair in pairs
                for item in (pair.original, pair.changed)}) == 48
    for prior, pair in zip(base, pairs, strict=True):
        assert pair.changed_instruction == prior.changed_instruction
        assert pair.original.program == prior.original.program
        assert pair.changed.program == prior.changed.program
        assert pair.original.inputs == pair.changed.inputs == prior.original.inputs
        assert pair.original.source_text != prior.original.source_text
        assert pair.changed.source_text != prior.changed.source_text
        assert pair.original.program.run(pair.original.inputs) != pair.changed.program.run(pair.changed.inputs)
        before, after = pair.original.instructions[-1], pair.changed.instructions[-1]
        assert pair.original.source_text[:before.operation_span.start] == (
            pair.changed.source_text[:after.operation_span.start])
        for example in (pair.original, pair.changed):
            recovered = semantic_public_character_inputs(example.source_text)
            assert recovered.values == example.inputs
            assert tuple(example.source_text[span.start:span.end]
                         for span in example.input_spans) == tuple(
                             example.source_text[item.character_start:item.character_end]
                             for item in recovered.literals)
            assert len(example.register_definition_spans) == (
                len(example.inputs) + example.program.depth)
            for annotated in example.instructions:
                assert all(example.source_text[span.start:span.end]
                           for span in annotated.argument_spans)


def test_noncommutative_roles_are_attributed_in_program_order():
    pairs = build_native_paraphrase_interventions(seed=2718283, style="definition")
    for pair in pairs:
        for example in (pair.original, pair.changed):
            for instruction in example.instructions:
                if instruction.instruction.op not in {"at", "count_of", "sub", "idiv"}:
                    continue
                arguments = tuple(example.source_text[span.start:span.end]
                                  for span in instruction.argument_spans)
                assert arguments[0] != arguments[1]
                if instruction.instruction.op in {"at", "count_of"}:
                    assert instruction.argument_spans[1].start < instruction.argument_spans[0].start


def test_renderer_rejects_unavailable_style():
    example = build_native_operation_interventions(seed=2718283)[0].original
    with pytest.raises(ValueError, match="style"):
        render_native_paraphrase(example, style="unknown")
