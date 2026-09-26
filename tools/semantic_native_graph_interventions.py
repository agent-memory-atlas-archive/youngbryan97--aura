"""Construct witnessed role and dependency interventions over existing graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_corpus import SemanticProgramExample
from core.learning.semantic_program_corpus_natural import build_semantic_program_natural_request_corpus
from core.learning.semantic_program_floor import (
    semantic_primitive_type_signature,
    semantic_program_structural_key,
)
from tools.semantic_native_paraphrase_interventions import render_native_paraphrase


@dataclass(frozen=True, slots=True)
class NativeGraphIntervention:
    original: SemanticProgramExample
    changed: SemanticProgramExample
    changed_instruction: int
    kind: str


def graph_intervention_candidate(program: Program, inputs, *, kind: str):
    """Keep primitive identities and change one type-correct causal binding."""
    if kind not in {"role", "dependency"}:
        raise ValueError("native graph intervention kind is unsupported")
    if (semantic_program_structural_key(program) is None or len(inputs) != program.n_inputs):
        raise ValueError("native graph intervention needs a complete source graph")
    types = ["integer_sequence" if isinstance(value, tuple) else "integer" for value in inputs]
    for step in program.instructions:
        signature = semantic_primitive_type_signature(step.op)
        if tuple(types[index] for index in step.args) != signature[0]:
            raise ValueError("native graph source is not type-correct")
        types.append(signature[1])
    value = program.run(inputs)
    if type(value) is not int:
        raise ValueError("native graph source does not return an exact integer")
    for ordinal in reversed(range(program.depth)):
        step = program.instructions[ordinal]
        if kind == "role":
            candidates = (tuple(reversed(step.args)),) if len(step.args) == 2 else ()
        else:
            candidates = tuple((*step.args[:role], reference, *step.args[role + 1:])
                for role, prior in enumerate(step.args)
                for reference in range(program.n_inputs, program.n_inputs + ordinal)
                if reference != prior and types[reference] == types[prior])
        for arguments in candidates:
            if (arguments == step.args or tuple(types[index] for index in arguments)
                    != semantic_primitive_type_signature(step.op)[0]):
                continue
            changed = Program(program.n_inputs, (*program.instructions[:ordinal],
                Instruction(step.op, arguments), *program.instructions[ordinal + 1:]))
            if semantic_program_structural_key(changed) is None:
                continue
            result = changed.run(inputs)
            if type(result) is int and result != value:
                return changed, ordinal
    return None


def build_native_graph_interventions(*, seed: int, kind: str,
                                    pairs_per_topology: int = 8) -> tuple[NativeGraphIntervention, ...]:
    """Freeze balanced generated controls before seeing any model decision."""
    if (type(seed) is not int or seed < 0 or type(pairs_per_topology) is not int
            or not 1 <= pairs_per_topology <= 8 or kind not in {"role", "dependency"}):
        raise ValueError("native graph intervention settings differ")
    groups, seen = {}, set()
    for attempt in range(16):
        for original in build_semantic_program_natural_request_corpus(
                seed=seed + attempt * 104729, examples_per_schema_domain=1):
            group = groups.setdefault(original.topology_id, [])
            if len(group) == pairs_per_topology:
                continue
            # A noncommutative terminal operation makes role changes observable.
            baseline = Program(original.program.n_inputs, (*original.program.instructions[:-1],
                Instruction("sub", original.program.instructions[-1].args)))
            changed = graph_intervention_candidate(baseline, original.inputs, kind=kind)
            if changed is None:
                continue
            program, ordinal = changed
            basis = replace(original, construction_id=f"native-{kind}-{original.construction_id}")
            before = render_native_paraphrase(basis, style="definition", program=baseline)
            after = render_native_paraphrase(basis, style="definition", program=program)
            if before.source_text in seen or after.source_text in seen:
                continue
            seen.update((before.source_text, after.source_text))
            group.append(NativeGraphIntervention(before, after, ordinal, kind))
        if groups and all(len(group) == pairs_per_topology for group in groups.values()):
            return tuple(pair for group in groups.values() for pair in group)
    raise ValueError("native graph intervention cannot fill every generated topology")
