"""The CAA extraction split and geometry cannot silently admit shortcuts."""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.caa.contrastive_design import (
    SCHEMA,
    ContrastDesign,
    construct_candidates,
    paired_direction,
    polarity_flip_nulls,
    remove_nuisance_subspace,
    unit_direction,
)
from tools.capture_27b_steering_vectors import capture_designed_vectors
from training.caa_contrastive_corpus import (
    CONTROL_DIMENSIONS,
    NUISANCE_BY_TARGET,
    TARGET_DIMENSIONS,
    build_contrastive_corpus,
)


def _design() -> dict:
    return {
        "schema": SCHEMA,
        "model_descriptor_sha256": "a" * 64,
        "pairs": [
            {"dimension": "curiosity", "split": split, "template_family": family,
             "topic": topic, "positive": f"{family} {topic} I want to examine it",
             "negative": f"{family} {topic} I do not want to examine it"}
            for split, families in (("train", ("statement", "reflection")),
                                    ("dev", ("dialogue",)))
            for family in families for topic in ("music", "arithmetic")
        ],
    }


def test_design_requires_disjoint_template_families_and_multiple_topics() -> None:
    design = ContrastDesign.from_dict(_design())
    assert len(design.partition("curiosity", "train")) == 4
    assert len(design.partition("curiosity", "dev")) == 2
    assert design.sha256 == ContrastDesign.from_dict(_design()).sha256


def test_design_refuses_template_or_text_leakage() -> None:
    raw = _design()
    raw["pairs"][-1]["template_family"] = "statement"
    with pytest.raises(ValueError, match="template_leakage"):
        ContrastDesign.from_dict(raw)
    raw = _design()
    raw["pairs"][-1]["positive"] = raw["pairs"][0]["positive"]
    with pytest.raises(ValueError, match="prompt_reused"):
        ContrastDesign.from_dict(raw)


def test_design_refuses_unbound_or_undersized_data() -> None:
    raw = _design()
    raw["model_descriptor_sha256"] = "32b"
    with pytest.raises(ValueError, match="model_identity"):
        ContrastDesign.from_dict(raw)
    raw = _design()
    raw["pairs"] = raw["pairs"][:4]
    with pytest.raises(ValueError, match="partition_too_small"):
        ContrastDesign.from_dict(raw)


def test_paired_direction_and_nuisance_projection() -> None:
    positive = np.array([[2., 1., 0.], [4., 3., 0.]])
    negative = np.array([[1., 0., 0.], [3., 2., 0.]])
    direction = paired_direction(positive, negative)
    assert np.allclose(direction, [1., 1., 0.])
    purified = remove_nuisance_subspace(direction, np.array([[1., 0., 0.]]))
    assert np.allclose(purified, [0., 1., 0.])
    assert np.allclose(unit_direction(purified), [0., 1., 0.])
    with pytest.raises(ValueError, match="unidentified"):
        unit_direction(remove_nuisance_subspace(direction, direction[None, :]))


def test_pair_order_is_not_a_null_but_polarity_flips_are() -> None:
    positive = np.ones((4, 4)) + 2 * np.eye(4)
    negative = np.zeros_like(positive)
    assert np.array_equal(paired_direction(positive, negative),
                          paired_direction(positive[::-1], negative[::-1]))
    nulls = polarity_flip_nulls(positive, negative, count=6, seed=3)
    assert len(nulls) == 6
    assert all(abs(float(np.dot(value, paired_direction(positive, negative)))) < 1e-8
               for value in nulls)
    assert all(value.shape == (4,) for value in nulls)
    assert all(np.array_equal(left, right) for left, right in zip(
        nulls, polarity_flip_nulls(positive, negative, count=6, seed=3), strict=True))


def test_polarity_controls_refuse_a_treatment_direction_disguised_as_null() -> None:
    positive = np.tile(np.array([1., 0.]), (8, 1))
    negative = np.zeros_like(positive)
    with pytest.raises(ValueError, match="geometry_unidentified"):
        polarity_flip_nulls(positive, negative, count=8, seed=25)
    with pytest.raises(ValueError, match="count_invalid"):
        polarity_flip_nulls(positive[:3], negative[:3], count=2, seed=25)


def test_provisional_corpus_has_disjoint_dev_for_every_target_and_control() -> None:
    design = build_contrastive_corpus("b" * 64)
    assert {row.dimension for row in design.pairs} == set(TARGET_DIMENSIONS) | set(CONTROL_DIMENSIONS)
    assert all(len(design.partition(dimension, "train")) == 8
               and len(design.partition(dimension, "dev")) == 2
               for dimension in (*TARGET_DIMENSIONS, *CONTROL_DIMENSIONS))
    assert all(set(NUISANCE_BY_TARGET[target]) <= set(CONTROL_DIMENSIONS)
               and target not in NUISANCE_BY_TARGET[target]
               for target in TARGET_DIMENSIONS)


def test_raw_and_purified_are_distinct_candidates_with_declared_nulls() -> None:
    positive = np.concatenate((np.tile([2., 1.], (4, 1)), np.eye(4) - .25), axis=1)
    negative = np.zeros_like(positive)
    controls = {"length": (np.array([[1., 0., 0., 0., 0., 0.],
                                      [2., 0., 0., 0., 0., 0.]]),
                           np.zeros((2, 6)))}
    candidates = construct_candidates(positive, negative, controls, null_count=6, seed=9)
    assert candidates["nuisance_dimensions"] == ("length",)
    assert candidates["raw"][0] > 0
    assert np.allclose(candidates["purified"], [0., 1., 0., 0., 0., 0.])
    assert len(candidates["polarity_flip_nulls"]) == 6
    assert all(abs(cosine) < 1e-8 for cosine in candidates["polarity_null_target_cosines"])


def test_capture_consumes_train_pairs_only_and_requires_complete_geometry() -> None:
    design = build_contrastive_corpus("c" * 64)
    activations = {}
    for index, pair in enumerate(design.pairs):
        base = np.array([float(index % 7), 0., *([0.] * 8)])
        difference = np.array([0., 1., *(np.eye(8)[index % 8] - .125)])
        activations[pair.positive] = {25: base + difference}
        activations[pair.negative] = {25: base}
    called = []

    def read(prompt):
        called.append(prompt)
        return activations[prompt]

    candidates = capture_designed_vectors(design, read, [25], 10, NUISANCE_BY_TARGET)
    assert set(candidates) == set(TARGET_DIMENSIONS)
    assert all(candidates[name][25]["raw"].shape == (10,) for name in candidates)
    assert len(called) == sum(2 * len(design.partition(name, "train"))
                              for name in (*TARGET_DIMENSIONS, *CONTROL_DIMENSIONS))
    assert not set(called) & {prompt for pair in design.pairs if pair.split == "dev"
                             for prompt in (pair.positive, pair.negative)}

    def missing(_prompt):
        return None

    with pytest.raises(ValueError, match="capture_incomplete"):
        capture_designed_vectors(design, missing, [25], 3, NUISANCE_BY_TARGET)

    def collinear(prompt):
        side = 1.0 if prompt in {pair.positive for pair in design.pairs} else 0.0
        return {25: np.array([side, *([0.] * 9)])}

    with pytest.raises(ValueError, match="polarity_null_geometry_unidentified"):
        capture_designed_vectors(design, collinear, [25], 10, NUISANCE_BY_TARGET)
