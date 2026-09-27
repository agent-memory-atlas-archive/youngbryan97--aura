#!/usr/bin/env python3
"""tools/capture_27b_steering_vectors.py — CAA vectors for the resident checkpoint.

Her affective steering is causal: a vector added to the residual stream at
chosen layers changes the tokens. That is exactly why it cannot survive a model
swap. The 45 vectors retained from the 32B are 5120 wide and so is this
checkpoint's residual stream, so every one of them loads and every one of them
names a direction in a space that no longer exists. The migration contract says
so itself, with the steering component marked deferred, and a deferred steering
component is why no hooks install, why her substrate cannot reach the resident
model's forward pass, and why a fusion certificate over that channel would be a
measurement of nothing.

This captures the replacements. One forward pass per prompt with every target
layer hooked at once, rather than one pass per layer, because the difference of
means at layer 30 and the difference of means at layer 40 are two readings of
the same run.

Capture grants nothing. The plan's four evidence requirements are a separate
job, and `steering_regeneration.may_serve` fails closed without them.

    tools/capture_27b_steering_vectors.py
    tools/capture_27b_steering_vectors.py --plan artifacts/.../steering_plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_steering_capture")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan.json"


def capture_designed_vectors(design, read_prompt, layers_wanted, hidden, nuisance_by_target):
    """Capture paired train activations and retain raw, purified and null arms."""
    import numpy as np

    from core.consciousness.caa.contrastive_design import construct_candidates

    train = {}
    for dimension in sorted({row.dimension for row in design.pairs}):
        rows = design.partition(dimension, "train")
        by_layer = {layer: {"positive": [], "negative": []} for layer in layers_wanted}
        for pair in rows:
            for polarity in ("positive", "negative"):
                reading = read_prompt(getattr(pair, polarity))
                if reading is None or set(reading) != set(layers_wanted):
                    raise ValueError("contrast_capture_incomplete")
                for layer in layers_wanted:
                    vector = np.asarray(reading[layer], dtype=np.float32)
                    if vector.shape != (hidden,) or not np.isfinite(vector).all():
                        raise ValueError("contrast_capture_geometry_invalid")
                    by_layer[layer][polarity].append(vector)
        train[dimension] = {layer: (np.stack(values["positive"]),
                                    np.stack(values["negative"]))
                            for layer, values in by_layer.items()}
    candidates = {}
    for dimension, nuisances in nuisance_by_target.items():
        if dimension not in train or any(name not in train for name in nuisances):
            raise ValueError("contrast_capture_dimension_missing")
        candidates[dimension] = {}
        for layer in layers_wanted:
            candidates[dimension][layer] = construct_candidates(
                *train[dimension][layer],
                {name: train[name][layer] for name in nuisances},
                seed=layer,
            )
    return candidates


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "training/vectors",
        help="where the per-layer .npz files land",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--contrastive-corpus", action="store_true",
        help="capture paired, template-held-out train stimuli as separate raw, "
             "purified and polarity-null candidates; never grants serving authority",
    )
    arguments = parser.parse_args(argv)

    import numpy as np

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    model_path = Path(plan["model_path"])
    layers_wanted = [int(row["index"]) for row in plan["target_layers"]]
    kind_by_layer = {int(row["index"]): str(row["kind"]) for row in plan["target_layers"]}
    hidden = int(plan["hidden_size"])

    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None:
        print("no active cortex; nothing to bind a capture to", file=sys.stderr)
        return 1
    descriptor = str(spec.descriptor_sha256)
    # Two identifiers name this checkpoint and they are not the same string.
    # The plan fingerprints the weights; the registry keeps its own descriptor,
    # and the descriptor is the one a cached vector is checked against when the
    # steering engine decides whether to load it. So the capture binds on the
    # path, carries both, and stamps each file with the registry's.
    if Path(str(getattr(spec, "model_path", ""))).resolve() != model_path.resolve():
        print(
            f"the plan targets {model_path} and the active cortex is "
            f"{getattr(spec, 'model_path', None)}; refusing to bind a capture to "
            "a checkpoint the plan did not describe",
            file=sys.stderr,
        )
        return 1

    design = None
    if arguments.contrastive_corpus:
        from training.caa_contrastive_corpus import build_contrastive_corpus
        design = build_contrastive_corpus(descriptor)
        if arguments.out.exists():
            print("contrastive output must be a fresh directory", file=sys.stderr)
            return 1
    pairs = (sum(2 * len(design.partition(dimension, "train"))
                 for dimension in {row.dimension for row in design.pairs})
             if design is not None else
             sum(len(d["positive"]) + len(d["negative"]) for d in AFFECTIVE_DIMENSIONS))
    print(
        f"checkpoint      {model_path.name}\n"
        f"descriptor      {descriptor[:16]}\n"
        f"target layers   {len(layers_wanted)} "
        f"({sum(1 for k in kind_by_layer.values() if k == 'full_attention')} attention, "
        f"{sum(1 for k in kind_by_layer.values() if k != 'full_attention')} linear)\n"
        f"dimensions      {', '.join(d['key'] for d in AFFECTIVE_DIMENSIONS)}\n"
        f"forward passes  {pairs} (one per prompt, every layer read from each)\n"
        f"vectors out     {len(layers_wanted) * len(AFFECTIVE_DIMENSIONS)}",
        flush=True,
    )
    if arguments.dry_run:
        return 0

    # One 27B at a time. The live instance holds ~20GB wired on a 64GB host,
    # so a second checkpoint loaded beside it is what takes the machine down
    # rather than what measures it. The lane is the thing that knows.
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"caa-capture:{model_path.name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "capture_27b_steering_vectors"},
    ):
        import mlx.core as mx
        from mlx_lm import load

        started = time.time()
        print("loading the checkpoint", flush=True)
        model, tokenizer = load(str(model_path))
        print(f"loaded in {time.time() - started:.1f}s", flush=True)

        from core.brain.llm.decoder_topology import resolve_language_model
        decoder = resolve_language_model(model)
        blocks = list(getattr(decoder, "layers", None) or getattr(decoder.model, "layers", []))
        if not blocks:
            print("the loaded object publishes no decoder layers", file=sys.stderr)
            return 1
        if max(layers_wanted) >= len(blocks):
            print(
                f"the plan names layer {max(layers_wanted)} and the checkpoint has "
                f"{len(blocks)}",
                file=sys.stderr,
            )
            return 1

        captured: dict[int, object] = {}

        # One subclass per target block, each closing over its own index, so a
        # single pass reads every layer the plan asks for. Reading them one layer
        # at a time would run the same prompt sixteen times for sixteen numbers
        # that a single run already contains.
        for index in layers_wanted:
            block = blocks[index]
            base = block.__class__

            def make(base_class, layer_index):
                class Capturing(base_class):
                    def __call__(self, x, *args, **kwargs):
                        out = super().__call__(x, *args, **kwargs)
                        hidden_states = out[0] if isinstance(out, tuple) else out
                        if hidden_states is not None:
                            captured[layer_index] = hidden_states[0, -1, :].astype(mx.float32)
                        return out
                return Capturing

            block.__class__ = make(base, index)

        def read_prompt(text: str):
            captured.clear()
            tokens = tokenizer.encode(text)
            ids = getattr(tokens, "input_ids", tokens)
            try:
                out = model(mx.array([ids]))
                mx.eval(out, *[v for v in captured.values() if v is not None])
            except (RuntimeError, ValueError, TypeError) as exc:
                print(f"  prompt discarded: {type(exc).__name__}: {exc}", flush=True)
                return None
            return {
                index: np.array(value, dtype=np.float32)
                for index, value in captured.items()
                if value is not None
            }

        config_path = model_path / "config.json"
        config_sha = _sha256_file(config_path) if config_path.exists() else ""
        if design is not None:
            from training.caa_contrastive_corpus import NUISANCE_BY_TARGET

            candidates = capture_designed_vectors(design, read_prompt,
                                                    layers_wanted, hidden,
                                                    NUISANCE_BY_TARGET)
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".caa-capture-",
                                             dir=arguments.out.parent) as temporary:
                staging = Path(temporary) / "generation"
                staging.mkdir()
                entries = []
                for dimension, by_layer in sorted(candidates.items()):
                    for layer, variants in sorted(by_layer.items()):
                        for kind in ("raw", "purified"):
                            vector = variants[kind]
                            if vector is None:
                                continue
                            directory = staging / kind
                            directory.mkdir(exist_ok=True)
                            path = directory / f"{dimension}_layer{layer}.npz"
                            np.savez(path, v=vector, source="extracted_caa",
                                     extracted=True, dimension=dimension, layer=layer,
                                     layer_kind=kind_by_layer[layer], model=str(model_path),
                                     model_path=str(model_path), model_config_sha256=config_sha,
                                     model_config_path=str(config_path),
                                     model_descriptor_sha256=descriptor,
                                     plan_descriptor_fingerprint=str(plan["descriptor_fingerprint"]),
                                     contrast_design_sha256=design.sha256,
                                     purification=kind,
                                     derived_at=time.time())
                            entries.append({"path": str(path.relative_to(staging)),
                                            "sha256": _sha256_file(path),
                                            "raw_norm": variants["raw_norm"],
                                            "purified_norm": variants["purified_norm"],
                                            "nuisance_dimensions": variants["nuisance_dimensions"]})
                        for ordinal, vector in enumerate(variants["polarity_flip_nulls"]):
                            if vector is None:
                                continue
                            directory = staging / "polarity_nulls" / str(ordinal)
                            directory.mkdir(parents=True, exist_ok=True)
                            path = directory / f"{dimension}_layer{layer}.npz"
                            np.savez(path, v=vector, source="extracted_caa_null",
                                     extracted=True, dimension=dimension, layer=layer,
                                     model_descriptor_sha256=descriptor,
                                     contrast_design_sha256=design.sha256,
                                     control="within_pair_polarity_flip")
                            entries.append({"path": str(path.relative_to(staging)),
                                            "sha256": _sha256_file(path)})
                metadata = {"schema": "aura.caa.contrastive_capture.v1",
                            "model_descriptor_sha256": descriptor,
                            "contrast_design_sha256": design.sha256,
                            "plan_descriptor_fingerprint": plan["descriptor_fingerprint"],
                            "train_pairs": {name: len(design.partition(name, "train"))
                                            for name in sorted({row.dimension for row in design.pairs})},
                            "dev_prompts_used_for_extraction": False,
                            "serving_authority": False,
                            "vectors": entries}
                for directory in (staging / "raw", staging / "purified",
                                  *(staging / "polarity_nulls" / str(index)
                                    for index in range(8))):
                    if directory.exists():
                        relative = directory.relative_to(staging)
                        (directory / "metadata.json").write_text(
                            json.dumps({"schema": "aura.caa.contrastive_vector_arm.v1",
                                        "model_descriptor_sha256": descriptor,
                                        "contrast_design_sha256": design.sha256,
                                        "arm": str(relative),
                                        "serving_authority": False,
                                        "vectors": [entry for entry in entries
                                                    if (staging / entry["path"]).parent == directory]},
                                       sort_keys=True, indent=2) + "\n", encoding="utf-8")
                (staging / "metadata.json").write_text(
                    json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                if arguments.out.exists():
                    raise ValueError("contrast_output_appeared_during_capture")
                staging.rename(arguments.out)
            print(f"captured {len(entries)} candidate/control vectors into {arguments.out}; "
                  "no serving authority", flush=True)
            return 0

        arguments.out.mkdir(parents=True, exist_ok=True)
        written = 0
        for dimension in AFFECTIVE_DIMENSIONS:
            key = str(dimension["key"])
            sums: dict[str, dict] = {"positive": {}, "negative": {}}
            counts = {"positive": 0, "negative": 0}
            for side in ("positive", "negative"):
                for prompt in dimension[side]:
                    reading = read_prompt(str(prompt))
                    if not reading:
                        continue
                    counts[side] += 1
                    for index, vector in reading.items():
                        running = sums[side].get(index)
                        sums[side][index] = vector if running is None else running + vector
            if not counts["positive"] or not counts["negative"]:
                print(f"  {key}: no usable prompts on one side; skipped", flush=True)
                continue
            for index in layers_wanted:
                positive = sums["positive"].get(index)
                negative = sums["negative"].get(index)
                if positive is None or negative is None:
                    continue
                vector = (positive / counts["positive"]) - (negative / counts["negative"])
                if vector.shape[0] != hidden:
                    print(
                        f"  {key} layer {index}: width {vector.shape[0]} against "
                        f"{hidden}; skipped",
                        flush=True,
                    )
                    continue
                out = arguments.out / f"{key}_layer{index}.npz"
                np.savez(
                    out,
                    v=vector.astype(np.float32),
                    source="extracted_caa",
                    extracted=True,
                    dimension=key,
                    layer=index,
                    layer_kind=kind_by_layer.get(index, ""),
                    model=str(model_path),
                    model_path=str(model_path),
                    model_config_sha256=config_sha,
                    model_config_path=str(config_path),
                    model_descriptor_sha256=descriptor,
                    plan_descriptor_fingerprint=str(plan["descriptor_fingerprint"]),
                    positive_prompts=counts["positive"],
                    negative_prompts=counts["negative"],
                    derived_at=time.time(),
                )
                written += 1
            print(
                f"  {key}: {counts['positive']} positive, {counts['negative']} negative, "
                f"{len(layers_wanted)} layers",
                flush=True,
            )

        print(
            f"\nwrote {written} vectors to {arguments.out} in "
            f"{time.time() - started:.1f}s, bound to {descriptor[:16]}\n"
            "capture grants nothing: the four evidence requirements are a separate run",
            flush=True,
        )
        return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
