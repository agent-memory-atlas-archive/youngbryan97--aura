"""Prospective capacity plans; never select hyperparameters from held targets."""

import math

from tools.semantic_native_adapters import adapter_contract


def capacity_candidates(*, ranks=(8, 16, 32, 64), depths=(1, 4, 8),
                        kinds=("lora", "dora", "product", "routed", "square", "dense"), alpha=16.):
    return tuple(adapter_contract(rank=rank, layers=depth, sites="native_topology_v1",
        scaling="alpha_over_sqrt_rank_v1", alpha=alpha, kind=kind, experts=4 if kind == "routed" else 1)
        for rank in ranks for depth in depths for kind in kinds)


def source_capacity_selection(rows, *, baseline_correct_ids, source_calibration_ids,
                              regression_tolerance=0.):
    """Require actual source-correctness/retention and select the smaller fit.

    A capacity sweep is not launched by this helper. Each row must carry its
    own measured checkpoint and candidate geometry; missing measurements are
    rejected, not counted as null or passing.
    """
    source_ids, baseline = set(source_calibration_ids), set(baseline_correct_ids)
    if not source_ids or not baseline <= source_ids or regression_tolerance != 0.:
        raise ValueError("capacity selection needs explicit calibration and exact source retention")
    decisions, admitted = [], []
    for index, row in enumerate(rows):
        correct = set(row.get("correct_source_ids", ()))
        reasons = []
        if set(row.get("measured_source_ids", ())) != source_ids or not correct <= source_ids:
            reasons.append("source_calibration_coverage_differs")
        if not baseline <= correct:
            reasons.append("baseline_source_regression")
        step, receipt = row.get("selected_step"), row.get("checkpoint_receipt_sha256")
        if (type(step) is not int or step < 1 or not isinstance(receipt, str) or len(receipt) != 64
                or any(part not in "0123456789abcdef" for part in receipt)):
            reasons.append("no_selected_positive_step_checkpoint")
        count, loss = row.get("trainable_parameters"), row.get("source_calibration_loss")
        if (type(count) is not int or count < 1 or type(loss) not in {int, float}
                or not math.isfinite(loss) or loss < 0):
            reasons.append("capacity_or_loss_unmeasured")
        if row.get("selection_uses_held_targets") is not False:
            reasons.append("held_selection_not_excluded")
        decision = {"index": index, "admitted": not reasons, "reasons": reasons,
                    "new_correct_ids": sorted(correct - baseline), "lost_correct_ids": sorted(baseline - correct)}
        decisions.append(decision)
        if not reasons:
            admitted.append((-len(correct), loss, count, index))
    selected = min(admitted)[-1] if admitted else None
    return {"selected_index": selected, "decisions": decisions,
            "scope": "source_calibration_only", "promotion_authority": False}
