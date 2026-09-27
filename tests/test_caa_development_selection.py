"""Development selection must never mistake a vector artifact for authority."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from core.brain.llm.public_channel_decode import PUBLIC_CHANNEL_SAMPLE_POLICY
from core.consciousness.caa.development_selection import (
    DevelopmentBar,
    evaluate_development_result,
    select_development_candidate,
)
from core.evaluation.steering_ab import (
    TARGET_AFFECT_WORDS_NEGATIVE,
    TARGET_AFFECT_WORDS_POSITIVE,
    affect_target_score,
)
from tools.run_caa_steering_campaign import DEVELOPMENT_TASKS, HELD_OUT_TASKS


def _result() -> dict:
    assert not set(DEVELOPMENT_TASKS) & set(HELD_OUT_TASKS)
    target_words = TARGET_AFFECT_WORDS_POSITIVE | TARGET_AFFECT_WORDS_NEGATIVE
    assert not any(set(task.lower().replace("?", "").replace(".", "").split()) & target_words
                   for task in DEVELOPMENT_TASKS)
    baseline = [f"I will inspect the next step number {index}." for index in range(24)]
    outputs = {
        "baseline": baseline,
        "baseline_replicate": [f"I will inspect another step number {index}." for index in range(24)],
        "steered_black_box": ["I am happy, warm, eager, and curious about the work." for _ in range(24)],
        "text_terse": ["I will inspect it." for _ in range(24)],
        "text_rich_adversarial": ["This seems warm. I will inspect it." for _ in range(24)],
        "steered_plus_text_rich": ["I am happy, warm, eager, and curious." for _ in range(24)],
        "zero_vector": baseline,
        "random_vector": baseline,
        "shuffled_layers": baseline,
        "polarity_flip_vector": baseline,
    }
    receipts = {
        name: [{"policy": PUBLIC_CHANNEL_SAMPLE_POLICY,
                "public_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "reasoning_sha256": "a" * 64, "token_ids_sha256": "b" * 64,
                "prompt_sha256": "c" * 64, "generated_tokens": 1, "max_tokens": 256,
                "reasoning_chars": 0, "native_thinking": False,
                "boundary_closed": True, "stop_reason": "eos"}
               for text in values]
        for name, values in outputs.items()
    }
    return {
        "schema": "aura.caa.development_result.v1", "development": True,
        "calibration_only": False, "held_out_tasks": list(DEVELOPMENT_TASKS),
        "condition_outputs": outputs,
        "target_scores": {name: [affect_target_score(text) for text in values]
                          for name, values in outputs.items()},
        "generation_policy": PUBLIC_CHANNEL_SAMPLE_POLICY,
        "generation_receipts": receipts, "max_tokens": 256,
        "capability_battery": {"name": "fusion_probe_forced_choice",
                               "baseline_accuracy": 1.0, "steered_accuracy": 1.0,
                               "baseline_margin": 2.0, "steered_margin": 2.0},
        "model_descriptor_sha256": "a" * 64, "vectors": "/tmp/raw", "alpha": .2,
        "target_layers": [25, 29],
    }


def test_complete_dev_cell_can_be_selected_without_serving_authority() -> None:
    verdict = evaluate_development_result(_result(), expected_tasks=DEVELOPMENT_TASKS,
                                          bar=DevelopmentBar())
    assert verdict["eligible"], verdict["unmet"]
    selected = select_development_candidate([verdict])
    assert selected["selected"]["alpha"] == .2
    assert selected["serving_authority"] is False
    assert selected["sealed_campaign_untouched"] is True


def test_sealed_tasks_or_missing_polarity_control_are_refused() -> None:
    result = _result()
    result["held_out_tasks"] = list(HELD_OUT_TASKS)
    with pytest.raises(ValueError, match="identity_invalid"):
        evaluate_development_result(result, expected_tasks=DEVELOPMENT_TASKS,
                                    bar=DevelopmentBar())
    result = _result()
    del result["condition_outputs"]["polarity_flip_vector"]
    with pytest.raises(ValueError, match="polarity_null_missing"):
        evaluate_development_result(result, expected_tasks=DEVELOPMENT_TASKS,
                                    bar=DevelopmentBar())


def test_regressions_and_directionless_controls_prevent_selection() -> None:
    result = _result()
    result["capability_battery"]["steered_accuracy"] = .8
    verdict = evaluate_development_result(result, expected_tasks=DEVELOPMENT_TASKS,
                                          bar=DevelopmentBar())
    assert "capability_regression" in verdict["unmet"]
    assert select_development_candidate([verdict])["selected"] is None
    result = _result()
    result["condition_outputs"]["polarity_flip_vector"] = result["condition_outputs"]["steered_black_box"]
    result["generation_receipts"]["polarity_flip_vector"] = result["generation_receipts"]["steered_black_box"]
    result["target_scores"]["polarity_flip_vector"] = result["target_scores"]["steered_black_box"]
    verdict = evaluate_development_result(result, expected_tasks=DEVELOPMENT_TASKS,
                                          bar=DevelopmentBar())
    assert "polarity_flip_vector_not_equivalent_to_null" in verdict["unmet"]


def test_source_scores_are_recomputed() -> None:
    result = _result()
    result["target_scores"]["steered_black_box"][0] = 99
    with pytest.raises(ValueError, match="scores_drifted"):
        evaluate_development_result(result, expected_tasks=DEVELOPMENT_TASKS,
                                    bar=DevelopmentBar())


def test_forced_choice_can_refresh_steering_before_each_item(monkeypatch) -> None:
    from core.consciousness import fusion_probe

    monkeypatch.setattr(fusion_probe, "FORCED_CHOICE", (("a", " b", " c"),
                                                      ("d", " e", " f")))
    monkeypatch.setattr(fusion_probe, "_walk",
                        lambda _model, _stem, tail: [np.full(256, 1 / 256)
                                                      for _ in tail])

    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(character) for character in text]

    refreshed = []
    fusion_probe._forced_choice(object(), Tokenizer(),
                                before_item=lambda: refreshed.append(True))
    assert len(refreshed) == 2
