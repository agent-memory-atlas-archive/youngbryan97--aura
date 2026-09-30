"""Let replayed scoped computations constrain the existing candidate portfolio."""

from dataclasses import dataclass

from core.evidence.candidate_portfolio import select_candidate_portfolio
from core.evidence.necessary_condition_selector import (
    NecessaryEvidenceCondition,
    build_necessary_condition_selector,
)
from core.evidence.packet import observe
from core.reasoning.computational_knowledge import QuantityBounds, _sha, verify_computed_knowledge
from typing import Any


@dataclass(frozen=True, slots=True)
class ComputedPortfolioSelection:
    decision: object
    receipt: dict


def select_computed_semantic_portfolio(
    portfolio: Any,
    consequences: tuple[Any, ...],
    *,
    now: Any,
    result_unit: str='count',
) -> Any:
    """Reject a computed contradiction; overlapping consequences retain uncertainty."""
    consequences = tuple(consequences)
    if not consequences or len(consequences) > 128:
        raise ValueError("computed selection requires a bounded nonempty evidence set")
    checked = tuple(verify_computed_knowledge(item, scope=portfolio.source_sha256, now=now)
                    for item in consequences)
    if len({item.request.identity for item in checked}) != len(checked):
        raise ValueError("computed selection repeats an evidence identity")
    hard = tuple(item for item in checked if item.hard_constraint)
    # Jointly contradictory hard intervals cannot justify picking any candidate.
    if hard and (len({item.result.dimension for item in hard}) != 1
                 or max(item.result.lower for item in hard) > min(item.result.upper for item in hard)):
        raise ValueError("scoped computed constraints contradict one another")
    measurements, provenance, rows = {}, {}, []
    for name, execution in portfolio.executions:
        value = execution.get("result")
        candidate = QuantityBounds.of(value, unit=result_unit) if type(value) is int else None
        comparisons = []
        for item in checked:
            if candidate is not None and candidate.dimension != item.result.dimension:
                raise ValueError("candidate and computed consequence use different dimensions")
            overlap = candidate is not None and not (candidate.upper < item.result.lower
                                                     or candidate.lower > item.result.upper)
            comparisons.append({"identity": item.request.identity, "overlaps": overlap,
                                "hard_constraint": item.hard_constraint,
                                "consequence_receipt_sha256": item.to_dict()["receipt_sha256"]})
        measurements[name] = {"executable_program": float(execution["completed"]),
                              "computed_consistency": float(all(row["overlaps"] for row in comparisons
                                                                if row["hard_constraint"]))}
        row = {"candidate": name, "comparisons": comparisons, "measurements": measurements[name]}
        rows.append(row)
        provenance[name] = observe(1., origin="scoped_computational_knowledge",
                                   ref=_sha(row), subject=f"computed_selection:{portfolio.source_sha256}")
    selector = build_necessary_condition_selector((
        NecessaryEvidenceCondition("executable_program", 1., "answer_requires_completed_floor_execution"),
        NecessaryEvidenceCondition("computed_consistency", 1., "answer_must_not_contradict_scoped_computed_consequences"),
    ))
    admissible = [name for name, values in measurements.items() if all(values.values())]
    decision = select_candidate_portfolio(selector, incumbent=portfolio.decision.selected,
                                          measurements=measurements, provenance=provenance) if admissible else None
    body = {"schema": "aura.semantic_computed_selection.v1", "source_sha256": portfolio.source_sha256,
            "rows": rows, "hard_constraint_count": len(hard), "admissible": admissible,
            "selected": decision.selected if decision is not None else None,
            "estimates_override_incumbent": False, "candidate_correctness_certified": False,
            "source_interpretation_proven": False, "general_transfer_proven": False}
    return ComputedPortfolioSelection(decision, {**body, "receipt_sha256": _sha(body)})
