#!/usr/bin/env python3
"""Calculate source-only ranking obligations from immutable model measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learning.semantic_native_path_calibration import native_path_profile  # noqa: E402
from tools.evaluate_semantic_native_checkpoint import digest, verified_document  # noqa: E402


def ranking_obligation(scores, correct, *, margin):
    """Measure the exact relative score shift needed for a declared strict margin."""
    if (not isinstance(scores, (list, tuple)) or not scores
            or any(type(score) not in {int, float} or not math.isfinite(score) for score in scores)
            or type(correct) is not int or not 0 <= correct < len(scores)
            or type(margin) not in {int, float} or not math.isfinite(margin) or margin <= 0):
        raise ValueError("score obligation requires finite measured alternatives and a positive margin")
    rivals = [index for index in range(len(scores)) if index != correct]
    rival = max(rivals, key=scores.__getitem__) if rivals else None
    gap = None if rival is None else scores[correct] - scores[rival]
    return {"correct_index": correct, "strongest_rival_index": rival,
            "observed_margin": gap, "required_margin": margin,
            "relative_score_correction_needed": 0. if gap is None else max(0., margin - gap),
            "strict_margin_satisfied": gap is None or gap >= margin}


def source_obligations(row, *, margin):
    """Local greedy constraints and finite graph contrasts have different proof scopes."""
    profile = native_path_profile(row["decisions"])
    decisions = [{"ordinal": ordinal, "kind": item["kind"],
                  **ranking_obligation(item["scores"], item["correct_index"], margin=margin)}
                 for ordinal, item in enumerate(row["decisions"])]
    graph = row.get("whole_graph")
    graph_obligation = None
    if graph is not None:
        if (not isinstance(graph, dict) or type(graph.get("positive_index")) is not int
                or graph["positive_index"] != 0
                or not isinstance(graph.get("program_sha256s"), list)
                or len(graph["program_sha256s"]) < 2
                or len(set(graph["program_sha256s"])) != len(graph["program_sha256s"])
                or len(graph.get("scores", [])) != len(graph["program_sha256s"])):
            raise ValueError("score obligation lacks distinct measured whole-graph alternatives")
        graph_obligation = {**ranking_obligation(graph["scores"], 0, margin=margin),
                            "competitor_count": len(graph["scores"]) - 1,
                            "all_grammar_completions_covered": False}
    return {"source": row["source"], "teacher_path_exact": profile["exact_teacher_path"],
            "decisions": decisions, "whole_graph": graph_obligation,
            "greedy_strict_margin_satisfied": all(item["strict_margin_satisfied"] for item in decisions),
            "global_search_certified": False, "unseen_correctness_certified": False}


def audit_directory(training_directory, *, margin):
    """Read existing source calibration only; no hidden arrays or model load."""
    from tools.audit_semantic_native_path_calibration import calibration_competitions

    # Recorded scores remain auditable when an old training contract cannot be
    # selected by the current trainer. This path grants no fit qualification.
    training = verified_document(training_directory / "plan.json", "plan_sha256")
    report = verified_document(training_directory / "report.json")
    supervision = verified_document(training_directory / "supervision.json")
    if (report.get("plan_sha256") != training["plan_sha256"]
            or report.get("supervision_receipt_sha256") != supervision["receipt_sha256"]
            or training.get("held_labels_used_for_fit_or_selection") is not False):
        raise ValueError("score audit lacks source-only training custody")
    competitions = calibration_competitions(training, supervision)
    checkpoints = sorted(report["checkpoints"], key=lambda row: row["step"])
    if not checkpoints or len({row["step"] for row in checkpoints}) != len(checkpoints):
        raise ValueError("score audit checkpoint inventory differs")
    for checkpoint in checkpoints:
        step = checkpoint["step"]
        retained = verified_document(training_directory / f"checkpoint-{step}.json")
        weights = training_directory / f"checkpoint-{step}.safetensors"
        if (retained != checkpoint or retained.get("plan_sha256") != training["plan_sha256"]
                or hashlib.sha256(weights.read_bytes()).hexdigest() != retained["weights_sha256"]):
            raise ValueError("score audit checkpoint custody differs")
    sources = sorted(competitions)
    expected_graphs = {}
    for item in supervision.get("graph_rows", []):
        if item["source"] in competitions:
            expected_graphs.setdefault(item["source"], []).append(item["program_sha256"])
    results = []
    for checkpoint in checkpoints:
        step = checkpoint["step"]
        measured = verified_document(training_directory / f"calibration-paths-{step}.json")
        if (measured.get("plan_sha256") != training["plan_sha256"]
                or measured.get("step") != step
                or measured.get("receipt_sha256") != checkpoint.get("calibration_path_receipt_sha256")
                or [row.get("source") for row in measured["rows"]] != sources):
            raise ValueError("score audit calibration identity or population differs")
        rows = []
        for row in measured["rows"]:
            expected = competitions[row["source"]]
            if (len(row["decisions"]) != len(expected)
                    or any(actual["kind"] != alternatives[0]["kind"]
                        or actual["correct_index"] != alternatives[0]["correct_index"]
                        or actual["choices"] != [item["choice"] for item in alternatives]
                        or len(actual["scores"]) != len(alternatives)
                        for actual, alternatives in zip(row["decisions"], expected, strict=True))):
                raise ValueError("score audit measured teacher alternatives differ")
            graph = row.get("whole_graph")
            if (graph is None) != (row["source"] not in expected_graphs) or graph is not None and (
                    graph["program_sha256s"] != expected_graphs[row["source"]]):
                raise ValueError("score audit whole-graph inventory differs from supervision")
            rows.append(source_obligations(row, margin=margin))
        deficits = Counter(item["kind"] for row in rows for item in row["decisions"]
                           if not item["strict_margin_satisfied"])
        results.append({"step": step, "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
            "calibration_receipt_sha256": measured["receipt_sha256"], "population": len(rows),
            "teacher_paths_exact": sum(row["teacher_path_exact"] for row in rows),
            "greedy_margin_paths": sum(row["greedy_strict_margin_satisfied"] for row in rows),
            "decision_deficits_by_kind": dict(deficits),
            "graph_margin_deficits": sum(row["whole_graph"] is not None
                and not row["whole_graph"]["strict_margin_satisfied"] for row in rows), "rows": rows})
    body = {"schema": "aura.native_score_obligations.v1", "training_plan_sha256": training["plan_sha256"],
            "supervision_receipt_sha256": supervision["receipt_sha256"], "required_margin": margin,
            "partition": "source_calibration_only", "checkpoints": results,
            "model_scores_recomputed": False, "model_loaded": False, "held_labels_used": False,
            "current_training_contract_replayed": False, "fit_qualification_evidence": False,
            "training_selection_unchanged": True, "global_search_certified": False,
            "general_transfer_proven": False, "serving_authority": False}
    return {**body, "receipt_sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-margin", type=float, default=.1)
    args = parser.parse_args()
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    configure_refit_environment(args.output)
    result = audit_directory(args.training_directory.resolve(strict=True), margin=args.required_margin)
    _save_if_absent(args.output, result)
    print(json.dumps({"receipt_sha256": result["receipt_sha256"], "checkpoints": [
        {key: value for key, value in row.items() if key != "rows"} for row in result["checkpoints"]]},
        sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
