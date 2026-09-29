#!/usr/bin/env python3
"""Compare causal sharing with original single-row scores on one frozen model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learning.semantic_native_causal_groups import native_causal_groups  # noqa: E402
from tools.evaluate_semantic_native_checkpoint import (  # noqa: E402
    digest,
    selected_checkpoint,
    verified_document,
)
from tools.probe_semantic_proposer_crossfit import _save_if_absent  # noqa: E402


def replay_case_inventory(generation_directory, training, sources):
    """Freeze the first decision group of each shape in saved target-blind searches."""
    from mlx_lm.utils import load_tokenizer

    from core.learning.semantic_native_program import native_text_decision_sequence
    from core.learning.semantic_native_search import search_native_grammar
    from core.learning.semantic_native_source_control import native_score_input_receipt
    from tools.evaluate_semantic_native_grammar import grammar_examples, source_input_types
    from tools.verify_semantic_native_grammar import verify_grammar_row

    plan = verified_document(generation_directory / "plan.json", "plan_sha256")
    if (not sources or len(set(sources)) != len(sources)
            or plan.get("weight_mode") != "residual" or plan.get("source_evidence") != "source_text"
            or plan.get("search_score_mode") != "native_nonpositive"
            or not plan.get("search_completions") or plan.get("sources")[:len(sources)] != sources
            or plan.get("training_plan_sha256") != training["plan_sha256"]):
        raise ValueError("causal probe requires frozen residual source-text searches")
    tokenizer = load_tokenizer(Path(training["model_path"]))
    examples = grammar_examples(dataset=plan["dataset"], seed=plan["seed"], count=len(plan["sources"]))
    cases, seen_shapes = [], set()
    for example, source in zip(examples, plan["sources"], strict=True):
        if source not in sources:
            continue
        row = verified_document(generation_directory / "rows" / f"{source}.json")
        verify_grammar_row(row, example=example, identity=source, plan_sha256=plan["plan_sha256"])
        transcript = row["search"]["score_transcript"]
        receipts = row["score_input_receipts"]
        index = 0

        def scorer(choices, *, example=example, transcript=transcript,
                   receipts=receipts, source=source):
            nonlocal index
            if index >= len(transcript):
                raise ValueError("causal probe transcript ended before the search")
            entry = transcript[index]
            sequences = tuple(native_text_decision_sequence(
                example.source_text, choice.text, (choice.span,), tokenizer,
                max_tokens=training["max_sequence_tokens"]) for choice in choices)
            if ([choice.value for choice in choices] != entry["choices"]
                    or [native_score_input_receipt(sequence, None) for sequence in sequences]
                    != receipts[index]):
                raise ValueError("causal probe score inputs differ from the generated row")
            groups = native_causal_groups(sequences)
            shape = (len(choices), len(groups))
            if len(groups) < len(choices) and shape not in seen_shapes:
                seen_shapes.add(shape)
                cases.append({"source_sha256": source, "decision_index": index,
                              "shape": list(shape), "values": list(entry["choices"]),
                              "recorded_scores": list(entry["scores"]),
                              "token_sha256": [digest(sequence.tokens) for sequence in sequences]})
            index += 1
            return tuple(entry["scores"])

        result = search_native_grammar(source_input_types(example.source_text)[1], scorer,
            max_steps=plan["max_steps"], max_nodes=plan["search_nodes"],
            completions=plan["search_completions"], register_encoding=plan["register_encoding"],
            score_mode=plan["search_score_mode"])
        if (index != len(transcript) or result.expanded_nodes != row["search"]["expanded_nodes"]
                or result.scored_alternatives != row["search"]["scored_alternatives"]):
            raise ValueError("causal probe changed the recorded search inventory")
    if not cases:
        raise ValueError("causal probe has no shared full-shape decisions")
    return plan, cases


def score_probe_cases(cases, generation_directory, training, prefix, suffix, tokenizer):
    """Compute each declared group twice without using a task target."""
    import mlx.core as mx

    from core.learning.semantic_native_causal_groups import score_native_causal_groups
    from core.learning.semantic_native_program import native_text_decision_sequence
    from core.learning.semantic_native_search import search_native_grammar
    from tools.evaluate_semantic_native_grammar import grammar_examples, source_input_types
    from tools.train_semantic_native_program import native_loss

    plan = verified_document(generation_directory / "plan.json", "plan_sha256")
    examples = grammar_examples(dataset=plan["dataset"], seed=plan["seed"], count=len(plan["sources"]))
    by_source = {source: example for source, example in zip(plan["sources"], examples, strict=True)}
    selected = {(case["source_sha256"], case["decision_index"]): case for case in cases}
    measured = []
    for source in dict.fromkeys(case["source_sha256"] for case in cases):
        example = by_source[source]
        row = verified_document(generation_directory / "rows" / f"{source}.json")
        index = 0

        def scorer(choices, *, example=example, row=row, source=source):
            nonlocal index
            case = selected.get((source, index))
            entry = row["search"]["score_transcript"][index]
            if case is not None:
                sequences = tuple(native_text_decision_sequence(
                    example.source_text, choice.text, (choice.span,), tokenizer,
                    max_tokens=training["max_sequence_tokens"]) for choice in choices)
                if ([choice.value for choice in choices] != case["values"]
                        or [digest(sequence.tokens) for sequence in sequences] != case["token_sha256"]):
                    raise ValueError("causal probe frozen decision changed")
                direct = tuple(-native_loss(suffix, prefix.capture(
                    mx.array([sequence.tokens[:-1]], dtype=mx.int32)), sequence,
                    summed=True, scope="semantic_decisions").item() for sequence in sequences)
                shared, receipt = score_native_causal_groups(prefix, suffix, sequences)
                if list(case["shape"]) != [len(sequences), receipt["full_single_row_forwards"]]:
                    raise ValueError("causal probe frozen group shape changed")
                measured.append({"source_sha256": source, "decision_index": index,
                                 "shape": case["shape"], "direct": direct, "shared": shared,
                                 "recorded_scores": case["recorded_scores"],
                                 "shared_forwards": receipt["shared_forwards"]})
                print(json.dumps({"stage": "causal_probe", "source_sha256": source,
                                  "decision_index": index, "exact": direct == shared,
                                  "shared_forwards": receipt["shared_forwards"]}), flush=True)
            index += 1
            return tuple(entry["scores"])

        search_native_grammar(source_input_types(example.source_text)[1], scorer,
            max_steps=plan["max_steps"], max_nodes=plan["search_nodes"],
            completions=plan["search_completions"], register_encoding=plan["register_encoding"],
            score_mode=plan["search_score_mode"])
    if len(measured) != len(cases):
        raise ValueError("causal probe did not execute its complete frozen inventory")
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--generation-directory", type=Path, required=True)
    parser.add_argument("--residual-calibration", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    from tools.verify_semantic_native_residual import verify_residual

    training, _selected = selected_checkpoint(args.training_directory)
    residual = verify_residual(args.residual_calibration, args.training_directory)
    generation, cases = replay_case_inventory(args.generation_directory, training, args.sources)
    if (generation["residual_calibration"]["report_receipt_sha256"] != residual["report_receipt_sha256"]
            or generation["residual_calibration"]["selected_scale"] != residual["selected_scale"]):
        raise ValueError("causal probe residual differs from generated search")
    paths = ("core/learning/semantic_native_causal_groups.py",
             "tools/probe_semantic_native_causal_groups.py")
    body = {"schema": "aura.native_causal_group_probe_plan.v1",
            "training_plan_sha256": training["plan_sha256"],
            "generation_plan_sha256": generation["plan_sha256"],
            "residual_report_receipt_sha256": residual["report_receipt_sha256"],
            "implementation": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                               for path in paths}, "cases": cases,
            "serving_authority": False, "qualification_evidence": False}
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "cases": len(cases),
                          "plan_sha256": plan["plan_sha256"]}), flush=True)
        return

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.calibrate_semantic_native_residual import native_lora_sites
    from tools.semantic_native_execution import apply_execution

    if (verified_document(args.directory / "plan.json", "plan_sha256") != plan
            or any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha
                   for path, sha in plan["implementation"].items())):
        raise ValueError("causal probe plan or implementation drifted")
    with (standalone_model_lane(owner_id=f"native-causal-probe:{args.directory.name}",
                                model_path=training["model_path"], purpose="evaluation",
                                require_exclusive=True, allow_owner_eviction=False,
                                preemptible=False, metadata={"production_effect": False}),
          mlx_memory_envelope(fraction=.80)):
        model, tokenizer = load(training["model_path"])
        model.freeze()
        model.eval()
        mx.random.seed(training["seed"])
        linear_to_lora_layers(model, training["suffix_layers"], {
            "rank": training["rank"], "scale": 16., "dropout": 0.,
            "keys": training["adapter_keys"]})
        model.load_weights(str(args.training_directory /
                               f"checkpoint-{residual['candidate_step']}.safetensors"), strict=False)
        apply_execution(model, training)
        sites = native_lora_sites(model, training)
        for site in sites:
            site.scale = 16. * residual["selected_scale"]
        split = len(model.layers) - training["suffix_layers"]
        prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
        measured = score_probe_cases(cases, args.generation_directory, training,
                                     prefix, suffix, tokenizer)
    errors = [abs(left - right) for case in measured
              for left, right in zip(case["direct"], case["shared"], strict=True)]
    history_errors = [abs(left - right) for case in measured
                      for left, right in zip(case["direct"], case["recorded_scores"], strict=True)]
    result = {"schema": "aura.native_causal_group_probe.v1",
              "plan_sha256": plan["plan_sha256"], "population": len(measured),
              "alternatives": sum(case["shape"][0] for case in measured),
              "full_single_row_forwards": sum(case["shape"][1] for case in measured),
              "max_direct_shared_score_error": max(errors),
              "max_history_score_error": max(history_errors),
              "exact_direct_shared_scores": all(error == 0. for error in errors),
              "model_scores_independently_recomputed": True,
              "qualifies_other_sources": False, "serving_authority": False,
              "cases": measured}
    _save_if_absent(args.directory / "report.json", {**result, "receipt_sha256": digest(result)})
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)


if __name__ == "__main__":
    main()
