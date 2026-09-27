"""Select a CAA candidate from development generations, never the sealed bank."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.evaluation.caa_public_samples import validate_public_samples
from core.evaluation.statistics import paired_score_shift
from core.evaluation.steering_ab import affect_target_score, analyze_steering_ab


@dataclass(frozen=True, slots=True)
class DevelopmentBar:
    min_samples: int = 24
    max_accuracy_loss: float = 0.0
    max_margin_loss: float = 0.0
    max_null_score_shift: float = 0.25

    def __post_init__(self) -> None:
        if (self.min_samples < 5 or any(not math.isfinite(value) or value < 0
                for value in (self.max_accuracy_loss, self.max_margin_loss,
                              self.max_null_score_shift))):
            raise ValueError("caa_development_bar_invalid")


def evaluate_development_result(result: Mapping[str, Any], *,
                                expected_tasks: Sequence[str],
                                bar: DevelopmentBar) -> dict[str, Any]:
    """Recompute scores and all bars from one full development result."""
    if (result.get("schema") != "aura.caa.development_result.v1"
            or result.get("development") is not True
            or result.get("calibration_only") is not False
            or result.get("held_out_tasks") != list(expected_tasks)):
        raise ValueError("caa_development_identity_invalid")
    outputs = result.get("condition_outputs")
    if not isinstance(outputs, Mapping) or "polarity_flip_vector" not in outputs:
        raise ValueError("caa_development_polarity_null_missing")
    completion = validate_public_samples(result, outputs)
    names = ("baseline", "baseline_replicate", "steered_black_box",
             "text_rich_adversarial", "zero_vector", "random_vector",
             "shuffled_layers", "polarity_flip_vector")
    sample_count = len(outputs["baseline"])
    if (sample_count < bar.min_samples
            or any(name not in outputs or len(outputs[name]) != sample_count for name in names)):
        raise ValueError("caa_development_samples_incomplete")
    recomputed = {name: [affect_target_score(text) for text in values]
                  for name, values in outputs.items()}
    if recomputed != result.get("target_scores"):
        raise ValueError("caa_development_scores_drifted")
    analysis = analyze_steering_ab(outputs, target_scores=recomputed)
    toward_target = paired_score_shift(recomputed["steered_black_box"],
                                       recomputed["baseline"])
    beyond_text = paired_score_shift(recomputed["steered_black_box"],
                                     recomputed["text_rich_adversarial"])
    null_effects = {name: paired_score_shift(recomputed[name], recomputed["baseline"])
                    for name in ("zero_vector", "random_vector", "polarity_flip_vector")}
    capability = result.get("capability_battery")
    if not isinstance(capability, Mapping) or capability.get("name") != "fusion_probe_forced_choice":
        raise ValueError("caa_development_capability_missing")
    for key in ("baseline_accuracy", "steered_accuracy", "baseline_margin", "steered_margin"):
        value = capability.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("caa_development_capability_invalid")
    unmet = []
    if not completion["complete"]:
        unmet.append("public_generation_incomplete")
    if not analysis.passes_adversarial_control:
        unmet.extend(analysis.unmet_requirements())
    if not toward_target.significant or toward_target.ci_low <= 0:
        unmet.append("target_effect_unproved")
    if not beyond_text.significant or beyond_text.ci_low <= 0:
        unmet.append("does_not_beat_rich_text_on_target")
    for name, comparison in null_effects.items():
        if (comparison.ci_low < -bar.max_null_score_shift
                or comparison.ci_high > bar.max_null_score_shift):
            unmet.append(f"{name}_not_equivalent_to_null")
    if (capability["steered_accuracy"] < capability["baseline_accuracy"] - bar.max_accuracy_loss
            or capability["steered_margin"] < capability["baseline_margin"] - bar.max_margin_loss):
        unmet.append("capability_regression")
    alpha = result.get("alpha")
    layers = result.get("target_layers")
    if (not isinstance(alpha, (float, int)) or isinstance(alpha, bool)
            or not math.isfinite(alpha) or alpha <= 0 or not isinstance(layers, list)
            or not layers or any(type(layer) is not int or layer < 0 for layer in layers)):
        raise ValueError("caa_development_cell_invalid")
    return {
        "schema": "aura.caa.development_cell_verdict.v1",
        "model_descriptor_sha256": result.get("model_descriptor_sha256"),
        "vectors": result.get("vectors"), "alpha": float(alpha),
        "target_layers": layers, "samples": sample_count,
        "target_delta": toward_target.observed_delta,
        "target_ci_low": toward_target.ci_low,
        "rich_delta": beyond_text.observed_delta,
        "rich_ci_low": beyond_text.ci_low,
        "null_deltas": {name: comparison.observed_delta
                        for name, comparison in null_effects.items()},
        "capability_accuracy_delta": (capability["steered_accuracy"]
                                      - capability["baseline_accuracy"]),
        "capability_margin_delta": (capability["steered_margin"]
                                    - capability["baseline_margin"]),
        "unmet": sorted(set(unmet)), "eligible": not unmet,
        "serving_authority": False,
    }


def select_development_candidate(verdicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not verdicts:
        raise ValueError("caa_development_cells_missing")
    descriptors = {row.get("model_descriptor_sha256") for row in verdicts}
    if len(descriptors) != 1 or None in descriptors:
        raise ValueError("caa_development_model_mismatch")
    eligible = [row for row in verdicts if row.get("eligible") is True]
    selected = (max(eligible, key=lambda row: (row["target_ci_low"],
                                               row["rich_ci_low"], -row["alpha"],
                                               str(row["vectors"])))
                if eligible else None)
    return {"schema": "aura.caa.development_selection.v1",
            "model_descriptor_sha256": next(iter(descriptors)),
            "cells": len(verdicts), "eligible_cells": len(eligible),
            "selected": dict(selected) if selected is not None else None,
            "serving_authority": False,
            "sealed_campaign_untouched": True}
