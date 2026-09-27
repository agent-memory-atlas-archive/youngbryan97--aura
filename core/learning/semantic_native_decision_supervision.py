"""Teacher forcing over the same typed alternatives used by native decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.learning.procedure_induction import Program
from core.learning.semantic_native_codec import native_surface_for_encoding
from core.learning.semantic_native_grammar import NativeGrammarDecision, decode_native_grammar

if TYPE_CHECKING:
    import mlx.core as mx

GRAMMAR_CHOICE_CONTRACT = {
    "basis": "native_typed_grammar_teacher_choices_v1",
    "alternatives": "all_type_admitted_inference_choices",
    "positive": "exact_source_procedure_next_decision",
    "loss": "mean_decision_conditional_negative_log_probability",
    "single_choice_loss": "zero",
    "finish_supervised": True,
    "target_depth_available_to_runtime": False,
    "alternative_semantic_incorrectness_claimed": False,
}


@dataclass(frozen=True)
class NativeTeacherDecision:
    kind: str
    choices: tuple[NativeGrammarDecision, ...]
    correct_index: int


def native_teacher_decisions(program: Program, input_types: tuple[str, ...], *,
                             register_encoding: str = "absolute_v1",
                             ) -> tuple[NativeTeacherDecision, ...]:
    """Recover a source target's decisions without constructing a second grammar.

    This function is training-only. The target chooses a path through the
    unchanged decoder; every alternative at that path remains in supervision.
    Other paths can compute equivalent results. Their exclusion from the exact
    requested procedure does not certify that their meanings are false.
    """
    if (not isinstance(program, Program) or program.n_inputs != len(input_types)
            or not 1 <= program.depth < 128):
        raise ValueError("native teacher needs matching public types and explicit finish headroom")
    target, _spans = native_surface_for_encoding(program, register_encoding=register_encoding)
    observed = []

    def teacher(choices: tuple[NativeGrammarDecision, ...]) -> tuple[float, ...]:
        positive = tuple(index for index, choice in enumerate(choices)
                         if target.startswith(choice.text)
                         and (type(choice.value) is not int
                              or target[len(choice.text):len(choice.text) + 1] in {",", "]"}))
        if len(positive) != 1:
            raise ValueError("native source procedure has no unique admitted next decision")
        observed.append((choices, positive[0]))
        return tuple(0. if index == positive[0] else -1. for index in range(len(choices)))

    result = decode_native_grammar(input_types, teacher, max_steps=program.depth + 1,
                                  register_encoding=register_encoding)
    if (result.program != program or result.bound_forced_completion
            or len(result.trace) != len(observed) or result.trace[-1]["chosen"] != "finish"):
        raise ValueError("native teacher path differs from its requested procedure")
    return tuple(NativeTeacherDecision(row["kind"], choices, correct)
                 for row, (choices, correct) in zip(result.trace, observed, strict=True))


def native_decision_choice_loss(scores: mx.array, correct_index: int) -> mx.array:
    """Normalize over admitted alternatives, including deterministic choices."""
    import mlx.core as mx

    from core.learning.semantic_native_program import native_choice_loss

    if (scores.ndim != 1 or scores.shape[0] < 1 or type(correct_index) is not int
            or not 0 <= correct_index < scores.shape[0]):
        raise ValueError("native decision objective needs one admitted source-positive choice")
    return mx.sum(scores) * 0. if scores.shape[0] == 1 else native_choice_loss(scores, (correct_index,))
