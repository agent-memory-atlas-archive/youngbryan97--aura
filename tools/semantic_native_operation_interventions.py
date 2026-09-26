"""Build source-only operation interventions for native program evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from core.learning.semantic_program_corpus import SemanticProgramExample
from core.learning.semantic_program_corpus_natural import (
    _natural_three_step_example,
    build_semantic_program_natural_request_corpus,
)
from core.learning.semantic_public_inputs import semantic_public_character_inputs

_FLIP = {"add": "sub", "sub": "add", "mul": "idiv", "idiv": "mul"}


@dataclass(frozen=True, slots=True)
class NativeOperationIntervention:
    original: SemanticProgramExample
    changed: SemanticProgramExample
    changed_instruction: int


def build_native_operation_interventions(*, seed: int) -> tuple[NativeOperationIntervention, ...]:
    """Change only the final requested operation, keeping public values fixed."""

    originals = build_semantic_program_natural_request_corpus(
        seed=seed, examples_per_schema_domain=1,
    )
    pairs = []
    for original in originals:
        operations = tuple(item.instruction.op for item in original.instructions)
        replacement = _FLIP[operations[-1]]
        changed = _natural_three_step_example(
            schema_kind=original.topology_id,
            domain_index=int(original.construction_id.rsplit("-", 1)[1]),
            sample_index=0,
            inputs=original.inputs,
            operations=(*operations[:-1], replacement),
        )
        old_span = original.instructions[-1].operation_span
        new_span = changed.instructions[-1].operation_span
        if (original.source_text[:old_span.start] != changed.source_text[:new_span.start]
                or not original.source_text.endswith(". What is the final value?")
                or not changed.source_text.endswith(". What is the final value?")
                or semantic_public_character_inputs(original.source_text).values != original.inputs
                or semantic_public_character_inputs(changed.source_text).values != changed.inputs
                or original.inputs != changed.inputs
                or original.program.run(original.inputs) == changed.program.run(changed.inputs)):
            raise ValueError("native operation intervention did not isolate a changed outcome")
        pairs.append(NativeOperationIntervention(original, changed, len(operations) - 1))
    return tuple(pairs)
