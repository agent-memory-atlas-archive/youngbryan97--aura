"""Contrast and vector contracts for model-bound activation steering.

Training pairs and development prompts have separate template families. The
sealed campaign is deliberately outside this module: it cannot be used for
vector construction or hyperparameter choice.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA = "aura.caa.contrastive_design.v1"


@dataclass(frozen=True, slots=True)
class ContrastPair:
    dimension: str
    split: str
    template_family: str
    topic: str
    positive: str
    negative: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ContrastPair:
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("contrast_pair_fields_invalid")
        if any(not isinstance(value, str) or not value.strip() for value in raw.values()):
            raise ValueError("contrast_pair_text_invalid")
        pair = cls(**raw)
        if pair.split not in {"train", "dev"} or pair.positive == pair.negative:
            raise ValueError("contrast_pair_polarity_or_split_invalid")
        return pair


@dataclass(frozen=True, slots=True)
class ContrastDesign:
    pairs: tuple[ContrastPair, ...]
    model_descriptor_sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ContrastDesign:
        if not isinstance(raw, dict) or set(raw) != {"schema", "model_descriptor_sha256", "pairs"}:
            raise ValueError("contrast_design_fields_invalid")
        if raw["schema"] != SCHEMA:
            raise ValueError("contrast_design_schema_invalid")
        descriptor = raw["model_descriptor_sha256"]
        if not isinstance(descriptor, str) or len(descriptor) != 64 or any(
            character not in "0123456789abcdef" for character in descriptor
        ):
            raise ValueError("contrast_design_model_identity_invalid")
        rows = raw["pairs"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("contrast_design_pairs_missing")
        design = cls(tuple(ContrastPair.from_dict(row) for row in rows), descriptor)
        design.validate()
        return design

    def validate(self) -> None:
        by_dimension: dict[str, list[ContrastPair]] = defaultdict(list)
        prompt_owner: dict[str, tuple[str, str]] = {}
        for pair in self.pairs:
            by_dimension[pair.dimension].append(pair)
            for prompt in (pair.positive, pair.negative):
                normalized = " ".join(prompt.casefold().split())
                if normalized in prompt_owner:
                    raise ValueError("contrast_design_prompt_reused")
                prompt_owner[normalized] = (pair.dimension, pair.split)
        for rows in by_dimension.values():
            train = [row for row in rows if row.split == "train"]
            dev = [row for row in rows if row.split == "dev"]
            if len(train) < 4 or len(dev) < 2:
                raise ValueError("contrast_design_partition_too_small")
            train_families = {row.template_family for row in train}
            dev_families = {row.template_family for row in dev}
            if len(train_families) < 2 or not dev_families or train_families & dev_families:
                raise ValueError("contrast_design_template_leakage")
            if len({row.topic for row in train}) < 2 or len({row.topic for row in dev}) < 2:
                raise ValueError("contrast_design_topic_coverage_missing")
            keys = [(row.split, row.template_family, row.topic) for row in rows]
            if len(keys) != len(set(keys)):
                raise ValueError("contrast_design_context_reused")

    @property
    def sha256(self) -> str:
        body = {"schema": SCHEMA, "model_descriptor_sha256": self.model_descriptor_sha256,
                "pairs": [{name: getattr(pair, name) for name in pair.__dataclass_fields__}
                          for pair in self.pairs]}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")
                                   ).encode("utf-8")).hexdigest()

    def partition(self, dimension: str, split: str) -> tuple[ContrastPair, ...]:
        if split not in {"train", "dev"}:
            raise ValueError("contrast_design_split_invalid")
        return tuple(row for row in self.pairs if row.dimension == dimension and row.split == split)


def paired_direction(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Mean of matched differences, preserving each pair as the unit of data."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if (positive.shape != negative.shape or positive.ndim != 2
            or positive.shape[0] < 2 or positive.shape[1] < 1
            or not np.isfinite(positive).all() or not np.isfinite(negative).all()):
        raise ValueError("contrast_activations_invalid")
    return np.mean(positive - negative, axis=0)


def remove_nuisance_subspace(direction: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Remove only measured control directions; never presume positional PCs."""
    direction = np.asarray(direction, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if (direction.ndim != 1 or controls.ndim != 2
            or controls.shape[1] != direction.size
            or not np.isfinite(direction).all() or not np.isfinite(controls).all()):
        raise ValueError("nuisance_geometry_invalid")
    if not controls.size:
        return direction.copy()
    _, singular, vh = np.linalg.svd(controls, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * max(controls.shape) * np.finfo(float).eps))
    basis = vh[:rank]
    return direction - basis.T @ (basis @ direction)


def unit_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    if direction.ndim != 1 or not np.isfinite(direction).all():
        raise ValueError("steering_direction_invalid")
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("steering_direction_unidentified")
    return (direction / norm).astype(np.float32)


def polarity_flip_nulls(positive: np.ndarray, negative: np.ndarray, *,
                        count: int, seed: int) -> tuple[np.ndarray, ...]:
    """Flip labels within pairs; reordering pairs alone cannot null a mean."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    paired_direction(positive, negative)
    if count < 1:
        raise ValueError("polarity_null_count_invalid")
    rng = np.random.default_rng(seed)
    differences = positive - negative
    return tuple(np.mean(differences * rng.choice((-1., 1.), size=(len(differences), 1)), axis=0)
                 for _ in range(count))


def construct_candidates(
    target_positive: np.ndarray,
    target_negative: np.ndarray,
    nuisance_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    null_count: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Return competing raw/purified vectors and label-flip null controls.

    Purification is a candidate, never an automatic improvement. Its effect
    and any loss of target signal must be tested on disjoint development data.
    """
    target = paired_direction(target_positive, target_negative)
    controls = []
    for name in sorted(nuisance_pairs):
        positive, negative = nuisance_pairs[name]
        controls.append(paired_direction(positive, negative))
    nuisance = np.stack(controls) if controls else np.empty((0, target.size))
    purified = remove_nuisance_subspace(target, nuisance)
    nulls = polarity_flip_nulls(target_positive, target_negative,
                                count=null_count, seed=seed)
    return {
        "raw": unit_direction(target),
        "purified": unit_direction(purified) if np.linalg.norm(purified) > 1e-8 else None,
        "polarity_flip_nulls": tuple(unit_direction(vector) if np.linalg.norm(vector) > 1e-8
                                      else None for vector in nulls),
        "raw_norm": float(np.linalg.norm(target)),
        "purified_norm": float(np.linalg.norm(purified)),
        "nuisance_dimensions": tuple(sorted(nuisance_pairs)),
    }


def summarize_design(design: ContrastDesign) -> dict[str, Any]:
    return {"schema": SCHEMA, "design_sha256": design.sha256,
            "model_descriptor_sha256": design.model_descriptor_sha256,
            "counts": {dimension: dict(Counter(row.split for row in design.pairs
                                                if row.dimension == dimension))
                       for dimension in sorted({row.dimension for row in design.pairs})}}
