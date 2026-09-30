"""A bounded pilot covers fit strata without borrowing held evidence."""

import hashlib
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

from core.learning.semantic_native_fit_sampling import (
    bounded_native_fit_schedule,
    native_depth_calibration_subset,
    scheduled_native_source_pairs,
)
from tools.verify_semantic_native_fit import verify_fit_sampling


def _examples(namespace="fit"):
    return tuple(SimpleNamespace(split="train", construction_id=construction,
        ir=SimpleNamespace(source_text_sha256=hashlib.sha256(
            f"{namespace}:{construction}:{depth}:{index}".encode()).hexdigest(),
            instructions=(None,) * depth))
        for construction, depth, count in (("old-a", 2, 3), ("old-b", 2, 2),
            ("new", 1, 24), ("new", 3, 24), ("new", 4, 24))
        for index in range(count))


def test_bounded_pilot_covers_all_construction_depth_strata_despite_cohort_imbalance():
    examples = _examples()
    ids = tuple(item.ir.source_text_sha256 for item in examples)
    schedule, receipt = bounded_native_fit_schedule(examples, ids, steps=17, seed=41)
    by_id = {item.ir.source_text_sha256: item for item in examples}
    exposure = Counter((by_id[source].construction_id, len(by_id[source].ir.instructions))
        for source in schedule)
    assert set(exposure) == {("old-a", 2), ("old-b", 2), ("new", 1), ("new", 3), ("new", 4)}
    assert max(exposure.values()) - min(exposure.values()) == 1
    assert receipt["primary_updates"] == 17
    assert receipt["eligible_sources"] == 77
    assert receipt["complete_primary_epoch"] is False
    assert receipt["held_labels_used"] is receipt["qualification_evidence"] is False
    assert bounded_native_fit_schedule(examples[::-1], ids[::-1], steps=17, seed=41) == (
        schedule, receipt)


@pytest.mark.parametrize("defect", ["too_short", "duplicate", "missing", "held", "boolean_steps"])
def test_bounded_pilot_rejects_incomplete_or_foreign_fit_metadata(defect):
    examples = list(_examples())
    ids = tuple(item.ir.source_text_sha256 for item in examples)
    steps = 17
    if defect == "too_short":
        steps = 4
    elif defect == "duplicate":
        ids += (ids[0],)
    elif defect == "missing":
        examples.pop()
    elif defect == "held":
        examples[0].split = "validation"
    else:
        steps = True
    with pytest.raises(ValueError):
        bounded_native_fit_schedule(examples, ids, steps=steps, seed=41)


def test_source_pair_filter_retains_unscheduled_fit_donors_but_excludes_their_primary_losses():
    pairs = {"a": [{"partner": "b", "kind": "termination"}],
             "b": [{"partner": "a", "kind": "termination"}]}
    assert scheduled_native_source_pairs(pairs, ("a", "a"), ("a", "b")) == {"a": pairs["a"]}
    with pytest.raises(ValueError, match="outside the fit"):
        scheduled_native_source_pairs({"a": [{"partner": "held"}]}, ("a",), ("a", "b"))
    with pytest.raises(ValueError, match="cross the fit"):
        scheduled_native_source_pairs(pairs, ("held",), ("a", "b"))
    with pytest.raises(ValueError, match="no witnessed"):
        scheduled_native_source_pairs({"b": pairs["b"]}, ("a",), ("a", "b"))


def test_depth_calibration_keeps_early_and_late_stops_within_the_same_construction():
    examples = _examples()
    ids = tuple(item.ir.source_text_sha256 for item in examples)
    selected = native_depth_calibration_subset(examples, ids, per_stratum=1)
    by_id = {item.ir.source_text_sha256: item for item in examples}
    assert len(selected) == 5
    assert {len(by_id[source].ir.instructions) for source in selected
            if by_id[source].construction_id == "new"} == {1, 3, 4}
    assert selected == native_depth_calibration_subset(examples[::-1], ids[::-1], per_stratum=1)
    with pytest.raises(ValueError, match="unique eligible"):
        native_depth_calibration_subset(examples, (*ids, "held"), per_stratum=1)
    with pytest.raises(ValueError, match="unique eligible"):
        native_depth_calibration_subset((*examples, examples[0]), ids, per_stratum=1)


def test_verifier_reconstructs_calibration_subset_from_the_independent_bank():
    examples = _examples()
    calibration_examples = _examples("calibration")
    items = {item.ir.source_text_sha256: item for item in (*examples, *calibration_examples)}
    ids = tuple(item.ir.source_text_sha256 for item in examples)
    calibration_ids = tuple(item.ir.source_text_sha256 for item in calibration_examples)
    schedule, receipt = bounded_native_fit_schedule(examples, ids, steps=17, seed=41)
    plan = {"fit_ids": list(ids), "steps": 17, "seed": 41,
        "scheduled_fit_ids": list(schedule), "fit_sampling_contract": receipt,
        "calibration_sampling_contract": {"policy": "construction_depth_lowest_source_sha256_v1",
            "per_stratum": 1, "source_limit": None},
        "calibration_ids": list(native_depth_calibration_subset(
            calibration_examples, calibration_ids, per_stratum=1))}
    bank = {"calibration_ids": list(calibration_ids)}
    assert not set(ids) & set(calibration_ids)
    assert verify_fit_sampling(plan, items, bank) == receipt
    plan["calibration_ids"].pop()
    with pytest.raises(ValueError, match="calibration differs"):
        verify_fit_sampling(plan, items, bank)
    with pytest.raises(ValueError, match="declared source bank"):
        verify_fit_sampling(plan, items)


@pytest.mark.parametrize("defect", [None, "order", "exposure", "depth", "seed"])
def test_independent_verifier_replays_sampler_and_detects_changed_training_exposure(defect):
    examples = _examples()
    items = {item.ir.source_text_sha256: item for item in examples}
    ids = tuple(items)
    schedule, receipt = bounded_native_fit_schedule(examples, ids, steps=17, seed=41)
    plan = {"fit_ids": list(ids), "steps": 17, "seed": 41,
        "scheduled_fit_ids": list(schedule), "fit_sampling_contract": deepcopy(receipt)}
    if defect is None:
        assert verify_fit_sampling(plan, items) == receipt
        assert verify_fit_sampling({}, {}) is None
        return
    if defect == "order":
        plan["scheduled_fit_ids"].reverse()
    elif defect == "exposure":
        plan["fit_sampling_contract"]["primary_updates"] += 1
    elif defect == "depth":
        examples[0].ir.instructions = (None,)
    else:
        plan["seed"] += 1
    with pytest.raises(ValueError, match="independent replay"):
        verify_fit_sampling(plan, items)
