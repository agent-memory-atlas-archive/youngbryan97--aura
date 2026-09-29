"""Build a small relation-invariance and intervention cohort without model scores."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from core.learning.semantic_program_corpus import SemanticProgramExample
from core.learning.semantic_program_corpus_natural import (
    build_semantic_program_natural_request_corpus,
)
from tools.semantic_native_graph_interventions import graph_intervention_candidate
from tools.semantic_native_paraphrase_interventions import render_native_paraphrase


@dataclass(frozen=True, slots=True)
class NativeRelationTransferCase:
    reference: SemanticProgramExample
    controlled: SemanticProgramExample
    kind: str


def build_native_relation_transfer(*, seed: int) -> tuple[NativeRelationTransferCase, ...]:
    """Hold values fixed while changing expression, argument role, or dependency."""
    if type(seed) is not int or seed < 0:
        raise ValueError("native relation transfer needs a nonnegative frozen seed")
    corpus = build_semantic_program_natural_request_corpus(
        seed=seed, examples_per_schema_domain=3)
    references = tuple(corpus[index] for index in (0, 24, 48))
    cases = []
    for reference in references:
        cases.append(NativeRelationTransferCase(reference,
            render_native_paraphrase(reference, style="definition"), "paraphrase"))
        for kind in ("role", "dependency"):
            changed = graph_intervention_candidate(reference.program, reference.inputs, kind=kind)
            if changed is None:
                raise ValueError(f"native relation transfer has no witnessed {kind} contrast")
            program, _ordinal = changed
            cases.append(NativeRelationTransferCase(reference,
                render_native_paraphrase(reference, style="definition", program=program), kind))
    sources = [case.controlled.source_text for case in cases]
    if len(set(sources)) != len(sources) or set(sources) & {
            reference.source_text for reference in references}:
        raise ValueError("native relation transfer duplicated a source")
    return tuple(cases)


def adjudicate_native_relation_transfer(reference_plan, controlled_plan,
                                        reference_verification, controlled_verification,
                                        reference_rows, controlled_rows):
    """Reuse the passed reference stage and require both invariance and sensitivity."""
    from core.learning.semantic_program_floor import semantic_program_structural_key

    additions = {"tools/semantic_native_relation_transfer.py",
                 "tools/semantic_native_graph_interventions.py",
                 "tools/semantic_native_paraphrase_interventions.py",
                 "tools/semantic_native_operation_interventions.py"}
    excluded = {"schema", "plan_sha256", "dataset", "sources", "implementation"}
    left_code = reference_plan.get("implementation", {})
    right_code = controlled_plan.get("implementation", {})
    if (reference_plan.get("dataset") != "natural_request"
            or controlled_plan.get("dataset") != "relation_transfer_controls"
            or reference_plan.get("weight_mode") not in {"fitted", "residual"}
            or reference_plan.get("schema", "").endswith(".v15")
                != controlled_plan.get("schema", "").endswith(".v15")
            or (reference_plan.get("weight_mode") == "residual"
                and (reference_plan.get("schema") not in {
                    "aura.semantic_native_grammar_plan.v13", "aura.semantic_native_grammar_plan.v15"}
                     or controlled_plan.get("schema") != reference_plan.get("schema")))
            or reference_plan.get("source_evidence") != "source_text"
            or reference_plan.get("search_completions", 0) < 1
            or {key: value for key, value in reference_plan.items() if key not in excluded}
                != {key: value for key, value in controlled_plan.items() if key not in excluded}
            or set(right_code) - set(left_code) != additions
            or any(right_code.get(path) != identity for path, identity in left_code.items())):
        raise ValueError("native relation transfer changed the frozen reference protocol")
    for plan, verification in ((reference_plan, reference_verification),
                               (controlled_plan, controlled_verification)):
        if (verification.get("artifacts_verified") is not True
                or verification.get("current_implementation_drift") != []
                or verification.get("plan_sha256") != plan["plan_sha256"]
                or verification.get("weight_mode") != plan["weight_mode"]
                or verification.get("training_plan_sha256") != plan["training_plan_sha256"]
                or verification.get("checkpoint_receipt_sha256")
                    != plan["checkpoint_receipt_sha256"]):
            raise ValueError("native relation transfer lacks unchanged independent verification")
    cases = build_native_relation_transfer(seed=reference_plan["seed"])
    def identity(example):
        return hashlib.sha256(example.source_text.encode()).hexdigest()
    references = tuple(dict.fromkeys(identity(case.reference) for case in cases))
    controls = tuple(identity(case.controlled) for case in cases)
    if (list(references) != reference_plan["sources"]
            or list(controls) != controlled_plan["sources"]
            or [row.get("source_sha256") for row in reference_rows] != list(references)
            or [row.get("source_sha256") for row in controlled_rows] != list(controls)):
        raise ValueError("native relation transfer source coverage differs")
    all_rows = (*reference_rows, *controlled_rows)
    if any(type(row.get(key)) is not bool for row in all_rows
           for key in ("program_equivalent", "answer_correct", "bound_forced_completion")):
        raise ValueError("native relation transfer outcomes were not measured")
    by_reference = {row["source_sha256"]: row for row in reference_rows}
    outcomes = []
    for case, row in zip(cases, controlled_rows, strict=True):
        reference = by_reference[identity(case.reference)]
        exact = (reference["program_equivalent"] and reference["answer_correct"]
                 and row["program_equivalent"] and row["answer_correct"]
                 and reference["decode_status"] == row["decode_status"] == "completed"
                 and not reference["bound_forced_completion"]
                 and not row["bound_forced_completion"])
        invariant_target = (semantic_program_structural_key(case.reference.program)
                            == semantic_program_structural_key(case.controlled.program))
        if invariant_target != (case.kind == "paraphrase"):
            raise ValueError("native relation transfer intervention changed its declared relation")
        outcomes.append({"reference_source_sha256": identity(case.reference),
            "controlled_source_sha256": identity(case.controlled), "kind": case.kind,
            "expected_relation_recovered": exact,
            "target_relation_preserved": invariant_target,
            "answer_contrast_witnessed": case.reference.program.run(case.reference.inputs)
                != case.controlled.program.run(case.controlled.inputs),
            "observed_program_reach": row["search"]["observed_program_reach"],
            "requested_top_k_proven": row["search"]["requested_top_k_proven"]})
    return {"reference_population": len(references), "controlled_population": len(controls),
            "relations_recovered": sum(row["expected_relation_recovered"] for row in outcomes),
            "by_kind": {kind: {"population": sum(row["kind"] == kind for row in outcomes),
                "recovered": sum(row["kind"] == kind and row["expected_relation_recovered"]
                                 for row in outcomes)}
                for kind in ("paraphrase", "role", "dependency")},
            "mechanism_micro_probe_passed": all(row["expected_relation_recovered"] for row in outcomes),
            "source_outcomes": outcomes, "reference_stage_reused_without_redecode": True,
            "general_transfer_proven": False, "broad_gain_proven": False,
            "serving_authority": False}
