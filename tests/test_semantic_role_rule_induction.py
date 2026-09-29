"""Role rules are induced from relations, not numeric values or construction IDs."""

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_grammar import NativeGrammarDecision, decode_native_grammar
from core.learning.semantic_role_rule_induction import (
    RoleObservation,
    induce_role_rules,
    native_role_rule_adjustments,
    observations_from_examples,
)


def _observation(source: str, operation: str, first: str, second: str,
                 construction: str, name: str = "sub") -> RoleObservation:
    """Give semantic role zero the second surface operand."""
    op = source.index(operation)
    left = source.index(first, op + len(operation))
    right = source.index(second, left + len(first))
    return RoleObservation(source, (op, op + len(operation)),
        ((right, right + len(second)), (left, left + len(first))), construction, name)


def test_induced_rule_transfers_over_values_position_and_operation_word() -> None:
    bank = induce_role_rules((
        _observation("Subtract 4 from 32.", "Subtract", "4", "32", "one"),
        _observation("Begin by remove 17 from 90, then report it.",
                     "remove", "17", "90", "two"),
    ))
    assert len(bank.rules) == 1
    assert bank.rules[0].role_to_surface_slot == (1, 0)
    assert bank.first_input_binding("Subtract 8 from 41. Afterward, multiply by 3.", "sub") == (1, 0)
    assert bank.first_input_binding("Remove 7 from 103.", "sub") == (1, 0)


def test_conflict_unknown_relation_and_ambiguous_clause_abstain() -> None:
    one = _observation("Subtract 4 from 32.", "Subtract", "4", "32", "one")
    two = _observation("Remove 5 from 40.", "Remove", "5", "40", "two")
    inverse = RoleObservation(two.source, two.operation,
        tuple(reversed(two.arguments)), two.construction, two.operation_name)
    assert not induce_role_rules((one, inverse)).rules
    bank = induce_role_rules((one, two))
    assert bank.first_input_binding("Subtract 4 and 32.", "sub") is None
    assert bank.input_binding_hypotheses("Subtract 4 from 32; subtract 8 from 64.", "sub") == (
        (1, 0), (3, 2))
    assert bank.first_input_binding("Subtract 4 from 32; subtract 8 from 64.", "sub") is None
    assert bank.first_input_binding(
        "Before you subtract 3 from 4, first subtract 9 from 20.", "sub") is None
    assert bank.first_input_binding("Subtract 4. From 32.", "sub") is None
    assert induce_role_rules((one,)).first_input_binding(one.source, "sub") is None


def test_source_local_names_bind_inputs_and_reverse_with_the_relation() -> None:
    bank = induce_role_rules((
        _observation("Subtract 4 from 32.", "Subtract", "4", "32", "one"),
        _observation("Remove 5 from 40.", "Remove", "5", "40", "two"),
    ))
    declarations = "Inputs are arrivals 825, returns 32. "
    assert bank.first_input_binding(
        declarations + "First, subtract returns from arrivals.", "sub") == (0, 1)
    assert bank.first_input_binding(
        declarations + "First, subtract arrivals from returns.", "sub") == (1, 0)
    assert bank.first_input_binding(
        "Inputs are item 4, item 5. First, subtract item from item.", "sub") is None


def test_held_corpus_examples_cannot_enter_the_fit() -> None:
    from core.learning.semantic_program_corpus import build_semantic_program_corpus

    corpus = build_semantic_program_corpus(examples_per_operation_pair=1)
    training = tuple(example for example in corpus if example.split == "train")
    held = tuple(example for example in corpus if example.split != "train")
    assert observations_from_examples(training)
    with pytest.raises(ValueError, match="held construction"):
        observations_from_examples(held)


def test_held_constructions_preserve_the_first_input_role_when_unambiguous() -> None:
    from core.learning.semantic_program_corpus import (
        build_semantic_program_corpus,
        build_semantic_program_fork_join_corpus,
    )
    from core.learning.semantic_public_inputs import semantic_public_character_inputs

    corpus = (build_semantic_program_corpus(examples_per_operation_pair=1)
              + build_semantic_program_fork_join_corpus(examples_per_operation_triple=1))
    bank = induce_role_rules(observations_from_examples(
        tuple(example for example in corpus if example.split == "train")))
    counts = {"correct": 0, "wrong": 0, "abstain": 0}
    for example in corpus:
        if example.split == "train":
            continue
        literals = semantic_public_character_inputs(example.source_text).literals
        expected = tuple(next(index for index, literal in enumerate(literals)
                              if (literal.character_start, literal.character_end)
                              == (span.start, span.end))
                         for span in example.instructions[0].argument_spans)
        proposed = bank.first_input_binding(
            example.source_text, example.instructions[0].instruction.op)
        counts["abstain" if proposed is None else "correct" if proposed == expected
               else "wrong"] += 1
    assert counts == {"correct": 544, "wrong": 0, "abstain": 96}


def test_induced_role_evidence_is_callable_by_native_decoder_without_target() -> None:
    bank = induce_role_rules((
        _observation("Subtract 4 from 32.", "Subtract", "4", "32", "one"),
        _observation("Remove 5 from 40.", "Remove", "5", "40", "two"),
    ))
    source = "Subtract 7 from 31."

    def model_scores(choices: tuple[NativeGrammarDecision, ...]) -> tuple[float, ...]:
        if choices[0].role_index >= 0:
            return tuple(0. if item.value == 0 else -1. for item in choices)
        return tuple(0. if item.value == "sub" else -10. for item in choices)

    def with_evidence(choices: tuple[NativeGrammarDecision, ...]) -> tuple[float, ...]:
        base = model_scores(choices)
        prior = native_role_rule_adjustments(bank, source, choices,
                                             input_count=2, strength=2.)
        return tuple(score + delta for score, delta in zip(base, prior, strict=True))

    baseline = decode_native_grammar(("integer", "integer"), model_scores, max_steps=1)
    assisted = decode_native_grammar(("integer", "integer"), with_evidence, max_steps=1)
    assert baseline.program == Program(2, (Instruction("sub", (0, 0)),))
    assert assisted.program == Program(2, (Instruction("sub", (1, 0)),))
    assert assisted.program.run((7, 31)) == 24


@pytest.mark.parametrize(("instruction", "expected_args", "expected_value"), (
    ("subtract returns from arrivals", (0, 1), 793),
    ("subtract arrivals from returns", (1, 0), -793),
))
def test_native_decoder_binds_source_local_names_and_reversed_roles(
    instruction: str, expected_args: tuple[int, int], expected_value: int,
) -> None:
    bank = induce_role_rules((
        _observation("Subtract 4 from 32.", "Subtract", "4", "32", "one"),
        _observation("Remove 5 from 40.", "Remove", "5", "40", "two"),
    ))
    source = f"Inputs are arrivals 825, returns 32. First, {instruction}."

    def score(choices: tuple[NativeGrammarDecision, ...]) -> tuple[float, ...]:
        base = tuple(0. if item.value in (0, "sub") else -1. for item in choices)
        prior = native_role_rule_adjustments(bank, source, choices,
                                             input_count=2, strength=2.)
        return tuple(value + adjustment for value, adjustment in zip(base, prior, strict=True))

    decoded = decode_native_grammar(("integer", "integer"), score, max_steps=1)
    assert decoded.program.instructions == (Instruction("sub", expected_args),)
    assert decoded.program.run((825, 32)) == expected_value


def test_role_adjustments_abstain_when_source_or_decision_is_ambiguous() -> None:
    bank = induce_role_rules((
        _observation("Subtract 4 from 32.", "Subtract", "4", "32", "one"),
        _observation("Remove 5 from 40.", "Remove", "5", "40", "two"),
    ))
    choices = (NativeGrammarDecision("", (0, 0), 0, 0, 0, "sub"),
               NativeGrammarDecision("", (0, 0), 1, 0, 0, "sub"))
    assert native_role_rule_adjustments(bank, "Subtract 4 and 32.", choices,
                                        input_count=2, strength=3.) == (0., 0.)
    assert native_role_rule_adjustments(bank, "", choices,
                                        input_count=2, strength=3.) == (0., 0.)
    assert native_role_rule_adjustments(bank,
        "Subtract 4 from 32; subtract 8 from 64.", choices,
        input_count=4, strength=3.) == (0., 0.)
    assert native_role_rule_adjustments(bank, "Subtract 4 from 32.", tuple(
        NativeGrammarDecision(item.text, item.span, item.value, 1, 0, "sub")
        for item in choices), input_count=2, strength=3.) == (0., 0.)
    relative = (NativeGrammarDecision("", (0, 0), "input:0", 0, 0, "sub"),
                NativeGrammarDecision("", (0, 0), "input:1", 0, 0, "sub"))
    assert native_role_rule_adjustments(bank, "Subtract 4 from 32.", relative,
                                        input_count=2, strength=3.) == (-3., 0.)
    with pytest.raises(ValueError, match="finite strength"):
        native_role_rule_adjustments(bank, "Subtract 4 from 32.", choices,
                                     input_count=2, strength=float("nan"))


def test_ambiguous_operation_form_is_withheld_and_bank_round_trips() -> None:
    first = _observation("Transfer 4 from 32.", "Transfer", "4", "32", "one", "sub")
    second = _observation("Transfer 5 from 40.", "Transfer", "5", "40", "two", "add")
    bank = induce_role_rules((first, second))
    assert bank.operation_forms == ()
    assert bank.first_input_binding("Transfer 8 from 64.", "sub") is None
    assert type(bank).from_dict(bank.to_dict()) == bank
    damaged = bank.to_dict()
    damaged["rules"][0]["role_to_surface_slot"] = [0, 0]
    with pytest.raises(ValueError, match="canonical"):
        type(bank).from_dict(damaged)
    damaged = bank.to_dict()
    damaged["operation_forms"] = [["subtract", "sub", "unexpected"]]
    with pytest.raises(ValueError, match="canonical"):
        type(bank).from_dict(damaged)


def test_role_fit_receipt_is_bound_to_training_and_excludes_held_sources(tmp_path) -> None:
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.fit_semantic_role_rule_bank import SCHEMA, _implementation, read_role_rule_fit

    bank = induce_role_rules((
        _observation("Subtract 4 from 32.", "Subtract", "4", "32", "one"),
        _observation("Remove 5 from 40.", "Remove", "5", "40", "two"),
    ))
    body = {"schema": SCHEMA, "native_training_plan_sha256": "plan",
            "source_report_sha256": "report", "fit_source_sha256s": ["fit"],
            "fit_sources": 1, "observations": 2, "minimum_constructions": 2,
            "implementation": _implementation(),
            "bank": bank.to_dict(), "serving_authority": False}
    fit = {**body, "receipt_sha256": digest(body)}
    path = tmp_path / "role-fit.json"
    get_file_write_gateway().write_json(path, fit, schema_version=1,
        schema_name=SCHEMA, source="test_semantic_role_rule_induction")
    training = {"plan_sha256": "plan", "source_report_sha256": "report", "fit_ids": ["fit"]}
    assert read_role_rule_fit(path, training=training)["bank"] == bank.to_dict()
    with pytest.raises(ValueError, match="overlaps"):
        read_role_rule_fit(path, training=training, held_source_sha256s=("fit",))
    with pytest.raises(ValueError, match="ancestry"):
        read_role_rule_fit(path, training={**training, "fit_ids": ["other"]})
