"""Source erasure preserves positions, template, and every supervised token."""

import os
from pathlib import Path

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
from core.learning.semantic_native_program import (
    NativeProgramSequence,
    native_program_sequence,
    native_text_decision_sequence,
)
from core.learning.semantic_native_source_control import (
    SOURCE_ERASURE_CONTRACT,
    apply_native_source_evidence,
    erase_native_source_tokens,
    native_score_input_receipt,
    source_control_mode_from_plan,
)
from core.learning.semantic_native_source_pairs import SOURCE_PAIR_CONTRACT
from tests.test_semantic_native_program import Tokenizer


@pytest.mark.parametrize("source", ["Subtract 8 from 13.", "user", "<assistant> is a quoted word."])
def test_erasure_removes_only_user_content_without_shortening_the_computation(source):
    tokenizer = Tokenizer()
    target = Program(2, (Instruction("sub", (0, 1)),))
    original = native_program_sequence(source, target, tokenizer)
    erased, receipt = erase_native_source_tokens(original, source, tokenizer)
    positions = set(receipt["erased_source_positions"])
    assert positions == set(range(len("<user>"), len("<user>") + len(source)))
    assert erased.tokens == tuple(ord("?") if index in positions else token
                                  for index, token in enumerate(original.tokens))
    assert len(erased.tokens) == len(original.tokens)
    assert erased.tokens[original.continuation_start:] == original.tokens[original.continuation_start:]
    assert erased.semantic_positions == original.semantic_positions
    assert erased.continuation_start == original.continuation_start
    assert receipt["source_content_tokens_available"] is False
    assert receipt["retained_nuisances"] == (
        "source_token_length", "template_position", "native_output_prefix")


def test_control_refuses_source_or_template_drift():
    tokenizer = Tokenizer()
    row = native_program_sequence("Exact source", Program(2, (Instruction("add", (0, 1)),)), tokenizer)
    with pytest.raises(ValueError, match="boundary"):
        erase_native_source_tokens(row, "Other source", tokenizer)
    with pytest.raises(ValueError, match="complete source-bound"):
        erase_native_source_tokens(NativeProgramSequence((1, 2), 5), "Exact source", tokenizer)


def test_source_evidence_applies_the_same_intervention_to_any_native_sequence():
    tokenizer = Tokenizer()
    source = "Subtract 8 from 13."
    row = native_program_sequence(source, Program(2, (Instruction("sub", (0, 1)),)), tokenizer)
    original, receipt = apply_native_source_evidence(row, source, tokenizer, mode="source_text")
    assert original is row and receipt is None
    erased, receipt = apply_native_source_evidence(
        row, source, tokenizer, mode="source_token_erasure")
    assert erased != row
    assert receipt["source_content_tokens_available"] is False
    assert erased.tokens[row.continuation_start:] == row.tokens[row.continuation_start:]
    intact_input = native_score_input_receipt(row, None)
    erased_input = native_score_input_receipt(erased, receipt)
    assert intact_input["sequence_sha256"] != erased_input["sequence_sha256"]
    assert intact_input["source_control_sha256"] is None
    assert erased_input["source_control_sha256"] is not None
    assert erased_input["erased_source_tokens"] == receipt["erased_source_tokens"]
    with pytest.raises(ValueError, match="source-evidence mode"):
        apply_native_source_evidence(row, source, tokenizer, mode="unknown")


def test_plan_distinguishes_old_evidence_from_explicit_fit_only_control():
    assert source_control_mode_from_plan({"schema": "aura.semantic_native_fit_plan.v1"}) == "source_text"
    plan = {"schema": "aura.semantic_native_fit_plan.v2",
            "source_evidence_control": dict(SOURCE_ERASURE_CONTRACT)}
    assert source_control_mode_from_plan(plan) == "source_token_erasure"
    for invalid in ({"schema": "aura.semantic_native_fit_plan.v2"},
                    {**plan, "schema": "aura.semantic_native_fit_plan.v1"},
                    {**plan, "source_evidence_control": {**SOURCE_ERASURE_CONTRACT, "scope": "all"}}):
        with pytest.raises(ValueError, match="contract differs"):
            source_control_mode_from_plan(invalid)


def test_paired_grammar_plan_requires_intact_source_and_bound_interaction():
    plan = {"schema": "aura.semantic_native_fit_plan.v4",
            "objective": "grammar_source_pairs", "loss_scope": "semantic_decisions",
            "grammar_choice_contract": GRAMMAR_CHOICE_CONTRACT,
            "grammar_source_pair_contract": SOURCE_PAIR_CONTRACT,
            "grammar_source_pair_fit_partners": {"a": {"partner": "b"}}}
    assert source_control_mode_from_plan(plan) == "source_text"
    for invalid in ({**plan, "source_evidence_control": SOURCE_ERASURE_CONTRACT},
                    {**plan, "grammar_source_pair_contract": {**SOURCE_PAIR_CONTRACT, "weight": 0.}},
                    {**plan, "grammar_source_pair_fit_partners": {}}):
        with pytest.raises(ValueError, match="paired-source|contract differs"):
            source_control_mode_from_plan(invalid)


@pytest.mark.parametrize("paired", [False, True])
def test_path_objective_requires_its_own_loss_and_baseline_selection_contract(paired):
    from core.learning.semantic_native_path_objective import (
        GRAMMAR_PATH_CONTRACT,
        path_choice_contract,
    )
    from core.learning.semantic_native_path_selection import PATH_SELECTION_CONTRACT

    plan = {"schema": "aura.semantic_native_fit_plan.v5",
            "objective": "grammar_source_pairs" if paired else "grammar_choices",
            "loss_scope": "semantic_decisions", "grammar_choice_contract": path_choice_contract(),
            "grammar_path_objective_contract": GRAMMAR_PATH_CONTRACT,
            "path_checkpoint_selection_contract": PATH_SELECTION_CONTRACT,
            "selection": "baseline_preserving_complete_source_calibration_paths",
            "unfitted_checkpoint_eligible": True}
    if paired:
        plan.update(grammar_source_pair_contract=SOURCE_PAIR_CONTRACT,
                    grammar_source_pair_fit_partners={"a": {"partner": "b"}})
    assert source_control_mode_from_plan(plan) == "source_text"
    for invalid in ({**plan, "schema": "aura.semantic_native_fit_plan.v4"},
                    {**plan, "grammar_choice_contract": GRAMMAR_CHOICE_CONTRACT},
                    {**plan, "grammar_path_objective_contract": {}},
                    {**plan, "path_checkpoint_selection_contract": {}},
                    {**plan, "selection": "minimum_loss"},
                    {**plan, "unfitted_checkpoint_eligible": False},
                    {**plan, "source_evidence_control": SOURCE_ERASURE_CONTRACT}):
        with pytest.raises(ValueError):
            source_control_mode_from_plan(invalid)


def test_current_tokenizer_can_erase_source_but_preserves_the_private_channel():
    checkpoint = os.environ.get("AURA_NATIVE_TOKENIZER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("local resident tokenizer checkpoint not supplied")
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(checkpoint))
    source = "Subtract the second input from the first."
    row = native_program_sequence(source, Program(2, (Instruction("sub", (0, 1)),)), tokenizer)
    erased, receipt = erase_native_source_tokens(row, source, tokenizer)
    assert receipt["erased_source_tokens"] > 0
    assert len(erased.tokens) == len(row.tokens)
    assert erased.tokens[row.continuation_start:] == row.tokens[row.continuation_start:]
    assert "</think>" in tokenizer.decode(list(erased.tokens))


def typed_plan():
    from core.learning.semantic_native_path_objective import (
        GRAMMAR_PATH_CONTRACT,
        path_choice_contract,
    )
    from core.learning.semantic_native_path_selection import PATH_SELECTION_CONTRACT
    from core.learning.semantic_native_typed_source_pairs import (
        TYPED_SOURCE_PAIR_CONTRACT,
        typed_source_pair_inventory,
    )

    pairs = {"a": [{"kind": "reference", "partner": "b", "decision_index": 1}]}
    schedule = ["a", "b"]
    return {"schema": "aura.semantic_native_fit_plan.v6",
            "objective": "grammar_source_pairs", "loss_scope": "semantic_decisions",
            "grammar_choice_contract": path_choice_contract(),
            "grammar_path_objective_contract": GRAMMAR_PATH_CONTRACT,
            "path_checkpoint_selection_contract": PATH_SELECTION_CONTRACT,
            "selection": "baseline_preserving_complete_source_calibration_paths",
            "unfitted_checkpoint_eligible": True,
            "fit_ids": schedule, "scheduled_fit_ids": schedule,
            "grammar_source_pair_contract": TYPED_SOURCE_PAIR_CONTRACT,
            "grammar_source_pair_fit_partners": pairs, "grammar_source_pair_updates": 1,
            "grammar_source_pair_inventory": typed_source_pair_inventory(pairs, schedule)}


def test_joint_graph_partial_reuse_is_explicit_and_cannot_relabel_full_reuse():
    from core.learning.semantic_native_path_objective import JOINT_GRAPH_CONTRAST_CONTRACT
    from core.learning.semantic_native_path_selection import JOINT_GRAPH_SELECTION_CONTRACT

    plan = {**typed_plan(), "schema": "aura.semantic_native_fit_plan.v7",
            "path_checkpoint_selection_contract": JOINT_GRAPH_SELECTION_CONTRACT,
            "selection": "baseline_preserving_joint_source_calibration",
            "joint_graph_contrast_limit": 2,
            "graph_contrast_contract": JOINT_GRAPH_CONTRAST_CONTRACT,
            "prefix_storage_contract": {"mode": "source_shards"},
            "execution_contract": {"prefix_strategy": "trie"},
            "reused_prefix_contract": {
                "schema": "aura.native_frozen_prefix_reuse.v2",
                "reuse_scope": "grammar_rows_only_graph_rows_recaptured"}}
    assert source_control_mode_from_plan(plan) == "source_text"
    for invalid in (None, {}, {"schema": "aura.native_frozen_prefix_reuse.v1"},
                    {"schema": "aura.native_frozen_prefix_reuse.v2"}):
        with pytest.raises(ValueError, match="whole-graph"):
            source_control_mode_from_plan({**plan, "reused_prefix_contract": invalid})


def test_typed_plan_is_explicit_and_does_not_relabel_the_old_objective():
    plan = typed_plan()
    assert source_control_mode_from_plan(plan) == "source_text"
    for version in [1, 2, 3, 4, 5]:
        with pytest.raises(ValueError, match="historical"):
            source_control_mode_from_plan({**plan, "schema": f"aura.semantic_native_fit_plan.v{version}"})


@pytest.mark.parametrize("defect", ["inventory", "updates", "peer", "schedule", "loss", "control"])
def test_typed_plan_cannot_claim_unmeasured_contrast_coverage(defect):
    from copy import deepcopy

    plan = deepcopy(typed_plan())
    if defect == "inventory":
        plan["grammar_source_pair_inventory"]["sources_by_kind"]["reference"] += 1
    elif defect == "updates":
        plan["grammar_source_pair_updates"] += 1
    elif defect == "peer":
        plan["grammar_source_pair_fit_partners"]["a"][0]["partner"] = "held"
    elif defect == "schedule":
        plan["scheduled_fit_ids"] = ["b"]
    elif defect == "loss":
        plan["grammar_source_pair_contract"] = SOURCE_PAIR_CONTRACT
    else:
        plan["source_evidence_control"] = SOURCE_ERASURE_CONTRACT
    with pytest.raises(ValueError):
        source_control_mode_from_plan(plan)


def test_current_tokenizer_erases_source_before_an_incomplete_grammar_decision():
    checkpoint = os.environ.get("AURA_NATIVE_TOKENIZER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("local resident tokenizer checkpoint not supplied")
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(checkpoint))
    source = "Subtract 8 from 13."
    target = '{"inputs":2,"steps":[["sub"'
    start = target.index("sub")
    row = native_text_decision_sequence(source, target, ((start, start + 3),), tokenizer)
    erased, receipt = apply_native_source_evidence(
        row, source, tokenizer, mode="source_token_erasure")
    assert receipt["erased_source_tokens"] > 0
    assert erased.semantic_positions == row.semantic_positions
    assert erased.tokens[row.continuation_start:] == row.tokens[row.continuation_start:]
