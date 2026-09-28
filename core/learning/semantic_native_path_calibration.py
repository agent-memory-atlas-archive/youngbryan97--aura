"""Complete-path calibration from the native grammar's measured competitions."""

from __future__ import annotations

import math
from collections import Counter


def native_path_profile(decisions: list[dict]) -> dict:
    """A greedy teacher path is exact iff every visited target choice wins.

    This is conditional procedure evidence, not semantic equivalence. The
    implication requires the identical deterministic scorer and grammar at
    decode time; choices after the first error are counterfactual contexts.
    """
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("native path calibration needs a complete nonempty path")
    counts, correct_counts, losses, errors = Counter(), Counter(), [], []
    for ordinal, row in enumerate(decisions):
        kind, scores, correct = row.get("kind"), row.get("scores"), row.get("correct_index")
        if (kind not in {"operation", "reference", "termination"}
                or not isinstance(scores, (list, tuple)) or not scores
                or any(type(score) not in {int, float} or not math.isfinite(score) for score in scores)
                or type(correct) is not int or not 0 <= correct < len(scores)):
            raise ValueError("native path calibration competition differs")
        winner = max(range(len(scores)), key=scores.__getitem__)
        peak = max(scores)
        loss = peak + math.log(sum(math.exp(score - peak) for score in scores)) - scores[correct]
        losses.append(loss)
        counts[kind] += 1
        correct_counts[kind] += winner == correct
        if winner != correct:
            errors.append({"ordinal": ordinal, "kind": kind, "correct_index": correct,
                           "chosen_index": winner, "margin": scores[correct] - scores[winner]})
    if decisions[-1]["kind"] != "termination":
        raise ValueError("native path calibration lacks an explicit finish competition")
    return {"decision_count": len(decisions), "decision_counts": dict(counts),
            "correct_counts": dict(correct_counts), "exact_teacher_path": not errors,
            "first_error": errors[0] if errors else None, "errors": errors,
            "mean_conditional_loss": sum(losses) / len(losses),
            "joint_target_log_probability": -sum(losses),
            "free_decode_measured": False, "meaning_equivalence_proven": False}


def native_path_totals(rows: list[dict], sources: list[str]) -> dict:
    """Count complete source paths, without diluting one failure into many atoms."""
    if (not isinstance(sources, list) or not sources or len(set(sources)) != len(sources)
            or [row.get("source") for row in rows] != sources):
        raise ValueError("native path calibration source coverage differs")
    profiles = [native_path_profile(row["decisions"]) for row in rows]
    counts, correct, first_errors = Counter(), Counter(), Counter()
    for profile in profiles:
        counts.update(profile["decision_counts"])
        correct.update(profile["correct_counts"])
        if profile["first_error"] is not None:
            first_errors[profile["first_error"]["kind"]] += 1
    return {"population": len(rows), "exact_teacher_paths": sum(
                profile["exact_teacher_path"] for profile in profiles),
            "decision_counts": dict(counts), "correct_counts": dict(correct),
            "first_error_counts": dict(first_errors),
            "mean_source_conditional_loss": sum(
                profile["mean_conditional_loss"] for profile in profiles) / len(rows),
            "free_decode_measured": False, "general_transfer_proven": False}
