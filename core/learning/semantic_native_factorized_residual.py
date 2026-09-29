"""Isolate native operation, binding, and stopping adaptations by grammar type."""

from __future__ import annotations

import math
from itertools import product
from typing import Any

from core.learning.procedure_induction import PRIMITIVES_BY_NAME
from core.learning.semantic_native_path_calibration import native_path_profile, native_path_totals
from core.learning.semantic_register_identity import RegisterIdentity
from core.verify.invariants import invariant

DECISION_KINDS = ("operation", "reference", "termination")
FACTORIZED_RESIDUAL_CONTRACT = {
    "schema": "aura.native_factorized_residual_contract.v1",
    "routing": "declared_native_grammar_choice_type",
    "selection": "complete_source_paths_without_lost_baseline_paths",
    "candidate_scales": "cartesian_product_of_measured_residual_scales",
    "unmeasured_score_interpolation": False,
    "source_or_construction_identity_used_at_runtime": False,
    "held_labels_used": False,
    "serving_authority": False,
}


def validated_kind_scales(scales: dict[str, float]) -> dict[str, float]:
    if (not isinstance(scales, dict) or set(scales) != set(DECISION_KINDS)
            or any(type(value) not in {int, float} or not math.isfinite(value)
                   or not 0. <= value <= 1. for value in scales.values())):
        raise ValueError("native factorization needs one finite measured scale per decision kind")
    return {kind: float(scales[kind]) for kind in DECISION_KINDS}


def native_competition_kind(choices: tuple[Any, ...]) -> str:
    """Read the existing grammar's disjoint atom types, never the source wording."""
    if not isinstance(choices, tuple) or not choices:
        raise ValueError("native factorization needs a nonempty typed competition")
    values = tuple(choice.value for choice in choices)
    if len(set(values)) != len(values):
        raise ValueError("native factorization repeats a grammar alternative")
    if all(isinstance(value, str) and value in PRIMITIVES_BY_NAME for value in values):
        return "operation"
    if all(isinstance(value, str) and value in {"finish", "continue"} for value in values):
        return "termination"
    if all(type(value) is int and value >= 0 for value in values):
        return "reference"
    if all(isinstance(value, str) for value in values):
        try:
            for value in values:
                RegisterIdentity.parse(value)
        # not a failure: a string that is not a register identity makes these
        # atoms something other than references, and the raise below says so.
        except ValueError:
            pass
        else:
            return "reference"
    raise ValueError("native factorization received mixed or undeclared grammar atoms")


def matched_residual_paths(rows_by_scale: dict[float, list[dict]], sources: list[str]) -> list[dict]:
    """Keep every source, alternative, and source-positive index identical."""
    if (not isinstance(rows_by_scale, dict) or len(rows_by_scale) < 2 or 0. not in rows_by_scale
            or any(type(scale) not in {int, float} or not math.isfinite(scale)
                   or not 0. <= scale <= 1. for scale in rows_by_scale)):
        raise ValueError("native factorization needs measured scales and an unfitted endpoint")
    for rows in rows_by_scale.values():
        native_path_totals(rows, sources)
    baseline = rows_by_scale[0.]
    for scale, rows in rows_by_scale.items():
        for row, reference in zip(rows, baseline, strict=True):
            if len(row["decisions"]) != len(reference["decisions"]):
                raise ValueError("native factorization path lengths differ across measured scales")
            for actual, expected in zip(row["decisions"], reference["decisions"], strict=True):
                if (any(actual.get(key) != expected.get(key)
                        for key in ("kind", "choices", "correct_index"))
                        or len(actual.get("choices", [])) != len(actual["scores"])):
                    raise ValueError("native factorization supervision differs across measured scales")
    return baseline


def factorized_paths(rows_by_scale: dict[float, list[dict]], sources: list[str],
                     scales: dict[str, float]) -> list[dict]:
    """Replay measured conditional scores; no numeric interpolation is performed."""
    scales = validated_kind_scales(scales)
    if any(scale not in rows_by_scale for scale in scales.values()):
        raise ValueError("native factorization requested an unmeasured scale")
    baseline = matched_residual_paths(rows_by_scale, sources)
    return [{"source": source, "decisions": [
        rows_by_scale[scales[decision["kind"]]][index]["decisions"][ordinal]
        for ordinal, decision in enumerate(row["decisions"])]}
        for index, (source, row) in enumerate(zip(sources, baseline, strict=True))]


def factorized_residual_admission(rows_by_scale: dict[float, list[dict]], sources: list[str]) -> dict:
    baseline = matched_residual_paths(rows_by_scale, sources)
    exact_base = {row["source"] for row in baseline
                  if native_path_profile(row["decisions"])["exact_teacher_path"]}
    adjudication = []
    for values in product(sorted(rows_by_scale), repeat=len(DECISION_KINDS)):
        scales = dict(zip(DECISION_KINDS, values, strict=True))
        rows = factorized_paths(rows_by_scale, sources, scales)
        exact = {row["source"] for row in rows
                 if native_path_profile(row["decisions"])["exact_teacher_path"]}
        adjudication.append({"scales": scales, "totals": native_path_totals(rows, sources),
                             "lost_baseline_sources": sorted(exact_base - exact),
                             "new_exact_sources": sorted(exact - exact_base),
                             "eligible": exact_base <= exact})
    selected = min((row for row in adjudication if row["eligible"]), key=lambda row: (
        -row["totals"]["exact_teacher_paths"], row["totals"]["mean_source_conditional_loss"],
        sum(row["scales"].values()), tuple(row["scales"][kind] for kind in DECISION_KINDS)))
    return {"selected_scales": selected["scales"], "selected_totals": selected["totals"],
            "adjudication": adjudication}


@invariant("learning.native_parameter_routes_are_disjoint_grammar_types", scope="learning",
           owner="core/learning/semantic_native_factorized_residual.py", observational=False)
def _native_parameter_route_types() -> tuple:
    from core.learning.semantic_native_grammar import NativeGrammarDecision

    atoms = (("add", "sub"), ("input:0", "result:0"), ("finish", "continue"))
    observed = tuple(native_competition_kind(tuple(NativeGrammarDecision("", (0, 0), value)
                                                 for value in values)) for values in atoms)
    assert observed == DECISION_KINDS
    try:
        native_competition_kind(tuple(NativeGrammarDecision("", (0, 0), value)
                                     for value in ("add", "finish")))
    except ValueError:
        return observed
    raise AssertionError("mixed native atom types acquired parameter-routing authority")
