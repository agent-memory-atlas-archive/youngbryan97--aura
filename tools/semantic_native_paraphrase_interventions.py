"""Render the same typed program in held-out definition and equation forms."""

from __future__ import annotations

import hashlib

from core.learning.semantic_program_corpus import (
    SemanticInstructionAnnotation,
    SemanticProgramExample,
    _AnnotatedText,
)
from core.learning.semantic_public_inputs import semantic_public_character_inputs
from tools.semantic_native_operation_interventions import (
    NativeOperationIntervention,
    build_native_operation_interventions,
)

_INPUT_NAMES = ("alpha", "beta", "gamma", "delta", "theta", "iota", "kappa", "lambda")
_RESULT_NAMES = ("epsilon", "zeta", "eta", "mu", "nu", "xi", "omicron", "rho")
_OPERATIONS = frozenset({"add", "sub", "mul", "idiv", "at", "count_of"})
_STYLES = frozenset({"definition", "equation"})


def _render_operation(builder: _AnnotatedText, *, style: str, op: str,
                      names: tuple[str, str], ordinal: int) -> None:
    def argument(position: int) -> None:
        builder.append(names[position], label=f"argument:{ordinal}:{position}")

    def operation(text: str) -> None:
        builder.append(text, label=f"operation:{ordinal}")

    if style == "equation":
        if op in {"add", "sub", "mul", "idiv"}:
            argument(0)
            builder.append(" ")
            operation({"add": "+", "sub": "-", "mul": "*", "idiv": "//"}[op])
            builder.append(" ")
            argument(1)
        elif op == "at":
            argument(0)
            operation("[")
            argument(1)
            builder.append("]")
        else:
            operation("frequency(")
            argument(1)
            builder.append(" in ")
            argument(0)
            builder.append(")")
        return

    if op in {"add", "sub", "mul", "idiv"}:
        opening, middle = {
            "add": ("the sum of ", " and "),
            "sub": ("the difference between ", " and "),
            "mul": ("the product of ", " and "),
            "idiv": ("the integer quotient of ", " divided by "),
        }[op]
        operation(opening.strip())
        builder.append(" ")
        argument(0)
        builder.append(middle)
        argument(1)
    elif op == "at":
        operation("the item at index")
        builder.append(" ")
        argument(1)
        builder.append(" of ")
        argument(0)
    else:
        operation("the frequency of")
        builder.append(" ")
        argument(1)
        builder.append(" in ")
        argument(0)


def render_native_paraphrase(original: SemanticProgramExample, *, style: str) -> SemanticProgramExample:
    """Change source expression only; retain the executable program and values."""
    if style not in _STYLES:
        raise ValueError("native paraphrase style is unsupported")
    program = original.program
    if (program.n_inputs > len(_INPUT_NAMES) or len(program.instructions) > len(_RESULT_NAMES)
            or any(item.op not in _OPERATIONS or len(item.args) != 2
                   for item in program.instructions)):
        raise ValueError("native paraphrase program exceeds its renderer")
    names = (*_INPUT_NAMES[:program.n_inputs], *_RESULT_NAMES[:len(program.instructions)])
    builder = _AnnotatedText()
    builder.append("Given these values: " if style == "definition" else "Inputs: ")
    for index, value in enumerate(original.inputs):
        if index:
            builder.append("; ")
        builder.begin(f"definition:{index}")
        builder.append(f"{names[index]} = ")
        literal = ("[" + ", ".join(str(item) for item in value) + "]"
                   if isinstance(value, tuple) else str(value))
        builder.append(literal, label=f"input:{index}")
        builder.finish(f"definition:{index}")
    builder.append(". Define the result in order: " if style == "definition"
                   else ". Evaluate these equalities in order: ")
    annotations = []
    for ordinal, instruction in enumerate(program.instructions):
        if ordinal:
            builder.append(". ")
        builder.begin(f"definition:{program.n_inputs + ordinal}")
        if style == "definition":
            builder.append("Let ")
        builder.append(names[program.n_inputs + ordinal])
        builder.append(" be " if style == "definition" else " = ")
        _render_operation(builder, style=style, op=instruction.op,
                          names=tuple(names[index] for index in instruction.args),
                          ordinal=ordinal)
        builder.finish(f"definition:{program.n_inputs + ordinal}")
        annotations.append(SemanticInstructionAnnotation(
            instruction=instruction,
            operation_span=builder.span(f"operation:{ordinal}"),
            argument_spans=tuple(builder.span(f"argument:{ordinal}:{position}")
                                 for position in range(2)),
            depends_on=tuple(sorted({index - program.n_inputs for index in instruction.args
                                     if index >= program.n_inputs})),
        ))
    builder.append(f". Return {names[-1]}.")
    text = builder.text
    if semantic_public_character_inputs(text).values != original.inputs:
        raise ValueError("native paraphrase changed source-grounded public values")
    identity = f"{style}|{original.example_id}|{text}"
    return SemanticProgramExample(
        example_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
        construction_id=f"native-{style}-{original.construction_id}",
        topology_id=original.topology_id,
        split=original.split,
        source_text=text,
        inputs=original.inputs,
        input_spans=tuple(builder.span(f"input:{index}") for index in range(program.n_inputs)),
        instructions=tuple(annotations),
        report_value=original.report_value,
        contrast_id=hashlib.sha256(f"{style}|{original.contrast_id}".encode()).hexdigest()[:24],
        register_definition_spans=tuple(builder.span(f"definition:{index}")
                                        for index in range(program.n_inputs + program.depth)),
    )


def build_native_paraphrase_interventions(*, seed: int,
                                          style: str) -> tuple[NativeOperationIntervention, ...]:
    pairs = []
    for original in build_native_operation_interventions(seed=seed):
        before = render_native_paraphrase(original.original, style=style)
        after = render_native_paraphrase(original.changed, style=style)
        if (before.program != original.original.program or after.program != original.changed.program
                or before.source_text == after.source_text or before.inputs != after.inputs
                or before.program.run(before.inputs) == after.program.run(after.inputs)):
            raise ValueError("native paraphrase failed to preserve the source intervention")
        pairs.append(NativeOperationIntervention(before, after, original.changed_instruction))
    return tuple(pairs)
