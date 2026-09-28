#!/usr/bin/env python3
"""Generate typed graphs from native semantic choices, without a target bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INTERVENTION_DATASETS = frozenset({
    "operation_intervention", "definition_intervention", "equation_intervention",
    "role_intervention", "dependency_intervention",
})
RETAINED_DATASETS = frozenset({"retained_validation", "retained_test"})


def capture_grammar_choices(prefix, branches, model, sequences, *, split_at, max_tokens,
                            strategy):
    """Return full causal states, reusing only an identical frozen source anchor."""
    import mlx.core as mx

    if strategy == "full":
        return tuple(prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                     for sequence in sequences), None
    if strategy != "trie" or not sequences:
        raise ValueError("native grammar prefix strategy or choice set differs")
    from core.learning.frozen_prefix_branches import FrozenPrefixBranches, native_source_anchor

    anchor = native_source_anchor(sequences)
    if branches is None:
        branches = FrozenPrefixBranches(model, split_at=split_at,
            anchor_tokens=anchor, max_tokens=max_tokens)
    elif branches.anchor_tokens != anchor:
        raise ValueError("native grammar source anchor changed within one request")
    states = branches.capture_many(tuple(tuple(sequence.tokens[:-1]) for sequence in sequences))
    return states, branches


def select_search_proposal(search, scorer):
    """Rank every admitted graph without receiving its target or correctness."""
    scores = tuple(scorer(candidate.result.program) for candidate in search.candidates)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in scores):
        raise ValueError("native proposal selection needs every finite graph score")
    return (max(range(len(scores)), key=scores.__getitem__) if scores else None), scores


def grammar_examples(*, dataset, seed, count):
    from core.learning.semantic_program_corpus_natural import (
        build_semantic_program_natural_request_corpus,
    )

    if dataset == "natural_request":
        examples = build_semantic_program_natural_request_corpus(
            seed=seed, examples_per_schema_domain=3,
        )
        ordered = tuple(examples[index] for sample in range(24) for index in
                        (sample, sample + 24, sample + 48))
    elif dataset in INTERVENTION_DATASETS:
        if dataset == "operation_intervention":
            from tools.semantic_native_operation_interventions import (
                build_native_operation_interventions,
            )

            pairs = build_native_operation_interventions(seed=seed)
        elif dataset in {"role_intervention", "dependency_intervention"}:
            from tools.semantic_native_graph_interventions import build_native_graph_interventions

            pairs = build_native_graph_interventions(seed=seed, kind=dataset.removesuffix("_intervention"))
        else:
            from tools.semantic_native_paraphrase_interventions import (
                build_native_paraphrase_interventions,
            )

            pairs = build_native_paraphrase_interventions(
                seed=seed, style=dataset.removesuffix("_intervention"))
        groups = {}
        for pair in pairs:
            groups.setdefault(pair.original.topology_id, []).append(pair)
        ordered = tuple(example for cohort in zip(*groups.values(), strict=True)
                        for pair in cohort for example in (pair.original, pair.changed))
        if count % 2:
            raise ValueError("operation interventions require complete source pairs")
    else:
        raise ValueError("unknown native grammar dataset")
    if not 1 <= count <= len(ordered):
        raise ValueError("native grammar population exceeds the frozen source inventory")
    return ordered[:count]


def source_input_types(source_text):
    from core.learning.semantic_public_inputs import semantic_public_character_inputs

    public = semantic_public_character_inputs(source_text)
    types = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                  for value in public.values)
    if not types:
        raise ValueError("native grammar source has no supported public values")
    return public, types


def validated_source_pair_map(examples, *, dataset, require_contrast=False):
    """Bind adjacent intervention sources without using a target during scoring."""
    if dataset not in INTERVENTION_DATASETS:
        if require_contrast:
            raise ValueError("source-pair swap needs an intervention dataset")
        return {}
    if len(examples) % 2:
        raise ValueError("source-pair intervention population is incomplete")
    sources = [hashlib.sha256(example.source_text.encode()).hexdigest() for example in examples]
    if len(set(sources)) != len(sources):
        raise ValueError("source-pair intervention sources are not unique")
    if require_contrast:
        from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent

        for index in range(0, len(examples), 2):
            left, right = examples[index:index + 2]
            left_public, left_types = source_input_types(left.source_text)
            right_public, right_types = source_input_types(right.source_text)
            if (left_public.values != right_public.values or left_types != right_types
                    or tuple(left.inputs) != tuple(right.inputs)):
                raise ValueError("source-pair swap needs matched public inputs and types")
            if semantic_programs_structurally_equivalent(left.program, right.program):
                raise ValueError("source-pair swap needs distinct programs")
            if left.program.run(left.inputs) == right.program.run(right.inputs):
                raise ValueError("source-pair swap needs distinct public answers")
    return {sources[index]: sources[index ^ 1] for index in range(len(sources))}


def grammar_pair_totals(rows, *, dataset):
    if dataset not in INTERVENTION_DATASETS:
        return {"pair_count": 0, "pair_exact": 0, "source_responsive": 0}
    if len(rows) % 2:
        raise ValueError("operation intervention outcomes split a source pair")
    pair_exact = source_responsive = 0
    for original, changed in zip(rows[::2], rows[1::2], strict=True):
        pair_exact += all(row["program_equivalent"] and row["answer_correct"]
                          for row in (original, changed))
        if original["decode_status"] != "completed" or changed["decode_status"] != "completed":
            continue
        left, right = original["program"]["instructions"], changed["program"]["instructions"]
        if dataset in {"role_intervention", "dependency_intervention"}:
            if len(left) != len(right) or any(a[0] != b[0] for a, b in zip(left, right, strict=True)):
                continue
            differences = [(a[1], b[1]) for a, b in zip(left, right, strict=True) if a[1] != b[1]]
            if len(differences) != 1:
                continue
            before, after = differences[0]
            if dataset == "role_intervention":
                source_responsive += len(before) == 2 and before == after[::-1]
            else:
                source_responsive += (len(before) == len(after)
                    and sum(a != b for a, b in zip(before, after, strict=True)) == 1)
        else:
            source_responsive += (left[:-1] == right[:-1] and left[-1][1] == right[-1][1]
                                  and left[-1][0] != right[-1][0])
    return {"pair_count": len(rows) // 2, "pair_exact": pair_exact,
            "source_responsive": source_responsive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--canary", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=3600.)
    parser.add_argument("--search-completions", type=int, default=0)
    parser.add_argument("--search-nodes", type=int, default=256)
    parser.add_argument("--search-score-mode", choices=("normalized_choices", "native_nonpositive"),
                        default="native_nonpositive")
    parser.add_argument("--weight-mode", choices=("fitted", "base"), default="fitted")
    parser.add_argument("--prefix-strategy", choices=("full", "trie"), default="full")
    parser.add_argument("--source-evidence", choices=("source_text", "source_token_erasure",
                                                      "source_pair_swap"),
                        default="source_text")
    parser.add_argument("--dataset", choices=("natural_request", *sorted(INTERVENTION_DATASETS | RETAINED_DATASETS)),
                        default="natural_request")
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--bundle", action="append")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    population_bound = 500 if args.dataset in RETAINED_DATASETS else 72
    if not 1 <= args.max_steps <= 128 or not 1 <= args.canary <= population_bound or not 0 < args.max_seconds <= 14400:
        parser.error("finite depth, population, and runtime bounds required")
    if not 0 <= args.search_completions <= 128 or not 1 <= args.search_nodes <= 100000:
        parser.error("finite search node and completion bounds required")
    if args.dataset in INTERVENTION_DATASETS and args.seed is None:
        parser.error("operation interventions require an explicit frozen seed")
    if args.dataset in RETAINED_DATASETS:
        if args.source_report is None or not args.bundle or args.seed is not None:
            parser.error("retained development sources need their report and bundles, without a new seed")
    elif args.source_report is not None or args.bundle is not None:
        parser.error("source report and bundles belong to retained development evaluation")
    if args.prefix_strategy == "trie" and (args.dataset not in INTERVENTION_DATASETS
            or args.search_completions):
        parser.error("native grammar trie requires intervention sources and greedy decode")
    seed = 0 if args.dataset in RETAINED_DATASETS else (3141592 if args.seed is None else args.seed)
    if seed < 0:
        parser.error("native grammar seed must be nonnegative")
    from tools.evaluate_semantic_native_checkpoint import digest, selected_checkpoint
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    configure_refit_environment(args.directory / "report.json")
    from core.brain.llm.model_registry import get_active_cortex_spec
    training, selected = selected_checkpoint(args.training_directory)
    from core.learning.semantic_native_codec import register_encoding_from_plan
    register_encoding = register_encoding_from_plan(training)
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("native grammar model identity differs from training")
    source_basis = None
    if args.dataset in RETAINED_DATASETS:
        from tools.semantic_native_retained_sources import load_retained_native_sources

        examples, source_basis = load_retained_native_sources(args.source_report, args.bundle,
            split=args.dataset.removeprefix("retained_"), count=args.canary)
        if source_basis["source_report_sha256"] != training["source_report_sha256"]:
            raise ValueError("retained native evaluation differs from the trained source basis")
    else:
        examples = grammar_examples(dataset=args.dataset, seed=seed, count=args.canary)
    sources = [hashlib.sha256(example.source_text.encode()).hexdigest() for example in examples]
    source_pair_map = validated_source_pair_map(
        examples, dataset=args.dataset, require_contrast=args.source_evidence == "source_pair_swap")
    source_text_by_sha256 = dict(zip(sources, (example.source_text for example in examples), strict=True))
    forbidden = set(training["fit_ids"]) | set(training["calibration_ids"]) | set(training["held_ids"])
    if forbidden & set(sources) or len(set(sources)) != len(sources):
        raise ValueError("native grammar evaluation sources overlap training")
    paths = ("tools/evaluate_semantic_native_grammar.py",
             "core/learning/semantic_native_grammar.py",
             "core/learning/semantic_native_search.py",
             "core/learning/semantic_native_program.py",
             "core/learning/semantic_native_codec.py",
             "core/learning/semantic_native_relative_program.py",
             "core/learning/semantic_native_source_control.py",
             "core/learning/semantic_register_identity.py",
             "core/learning/frozen_decoder_prefix.py",
             "core/learning/semantic_program_floor.py",
             "core/learning/semantic_program_corpus_natural.py",
             "core/learning/semantic_public_inputs.py",
             "tools/train_semantic_native_program.py")
    if args.dataset in INTERVENTION_DATASETS:
        paths += ("tools/semantic_native_operation_interventions.py",)
    if args.dataset in {"definition_intervention", "equation_intervention"}:
        paths += ("tools/semantic_native_paraphrase_interventions.py",)
    if args.dataset in {"role_intervention", "dependency_intervention"}:
        paths += ("tools/semantic_native_graph_interventions.py",
                  "tools/semantic_native_paraphrase_interventions.py")
    if args.dataset in RETAINED_DATASETS:
        paths += ("tools/semantic_native_retained_sources.py",
                  "core/learning/semantic_program_feature_materialization.py")
    from tools.semantic_native_execution import EXECUTION_PATHS, execution_from_plan
    paths += EXECUTION_PATHS
    execution = execution_from_plan(training, check_installed=True)
    if args.prefix_strategy == "trie" and (execution is None
            or execution["precision"] != "float32"):
        raise ValueError("native grammar trie requires measured float32 training arithmetic")
    if args.prefix_strategy == "trie":
        paths += ("core/learning/frozen_prefix_branches.py",)
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    schema_version = "v4" if args.dataset in {"definition_intervention", "equation_intervention"} else "v3"
    if args.dataset in {"role_intervention", "dependency_intervention"}:
        schema_version = "v5"
    if args.dataset in RETAINED_DATASETS:
        schema_version = "v6"
    if args.prefix_strategy == "trie":
        schema_version = "v7"
    body = {"schema": f"aura.semantic_native_grammar_plan.{schema_version}",
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": selected["receipt_sha256"],
            "weight_mode": args.weight_mode,
            "source_evidence": args.source_evidence,
            "source_pair_map": source_pair_map,
            "dataset": args.dataset, "seed": seed,
            "model_descriptor_sha256": spec.descriptor_sha256,
            "pointer_sha256": spec.pointer_sha256, "implementation": implementation,
            "sources": sources, "max_steps": args.max_steps, "max_seconds": args.max_seconds,
            "register_encoding": register_encoding,
            "search_mode": "best_first_then_complete_graph_score" if args.search_completions else "greedy",
            "search_nodes": args.search_nodes, "search_completions": args.search_completions,
            "search_score_mode": args.search_score_mode,
            "candidate_inventory": "none", "input_grounding": "semantic_public_character_inputs.v1",
            "target_available_to_scorer": False, "held_labels_used_for_fit_or_selection": False,
            "serving_authority": False, "qualification_evidence": False}
    if args.prefix_strategy == "trie":
        body["prefix_strategy"] = "trie"
    if source_basis is not None:
        body["source_cohort_basis"] = source_basis
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "population": len(examples),
                          "plan_sha256": plan["plan_sha256"]}), flush=True)
        return

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.semantic_native_codec import native_sequence_for_encoding
    from core.learning.semantic_native_grammar import (
        NativeGrammarIncompleteError,
        decode_native_grammar,
    )
    from core.learning.semantic_native_program import native_text_decision_sequence
    from core.learning.semantic_native_search import search_native_grammar
    from core.learning.semantic_native_source_control import (
        apply_native_source_evidence,
        native_score_input_receipt,
    )
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.train_semantic_native_program import native_loss

    started, rows = time.monotonic(), []
    with (standalone_model_lane(owner_id=f"semantic-native-grammar:{args.directory.name}",
                                model_path=str(spec.model_path), purpose="evaluation",
                                preemptible=False, metadata={"production_effect": False}),
          mlx_memory_envelope(fraction=.80)):
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        split = len(model.layers) - training["suffix_layers"]
        prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
        if args.weight_mode == "fitted":
            mx.random.seed(training["seed"])
            linear_to_lora_layers(model, training["suffix_layers"], {
                "rank": training["rank"], "scale": 16., "dropout": 0., "keys": training["adapter_keys"]})
            model.load_weights(str(args.training_directory /
                                   f"checkpoint-{selected['step']}.safetensors"), strict=False)
        from tools.semantic_native_execution import apply_execution
        apply_execution(model, training)
        for example, identity in zip(examples, sources, strict=True):
            public_inputs, types = source_input_types(example.source_text)
            scored_source = (source_text_by_sha256[source_pair_map[identity]]
                             if args.source_evidence == "source_pair_swap"
                             else example.source_text)
            scored_source_sha256 = hashlib.sha256(scored_source.encode()).hexdigest()
            scored = 0
            score_input_receipts = []
            branches = None
            def score(choices, *, source=scored_source, source_identity=identity,
                      receipts=score_input_receipts):
                nonlocal scored, branches
                sequences = []
                input_receipts = []
                for choice in choices:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError("native grammar run reached its finite bound")
                    sequence = native_text_decision_sequence(
                        source, choice.text, (choice.span,), tokenizer,
                        max_tokens=training["max_sequence_tokens"])
                    sequence, control = apply_native_source_evidence(
                        sequence, source, tokenizer,
                        mode="source_text" if args.source_evidence == "source_pair_swap"
                        else args.source_evidence)
                    sequences.append(sequence)
                    input_receipts.append(native_score_input_receipt(sequence, control))
                    scored += 1
                states, branches = capture_grammar_choices(prefix, branches, model,
                    sequences, split_at=split, max_tokens=training["max_sequence_tokens"],
                    strategy=args.prefix_strategy)
                scores = tuple(-native_loss(suffix, hidden, sequence, summed=True,
                    scope="semantic_decisions").item()
                    for hidden, sequence in zip(states, sequences, strict=True))
                receipts.append(input_receipts)
                print(json.dumps({"stage": "decision", "source_sha256": source_identity,
                                  "scored_prefixes": scored, "choices": len(choices)}), flush=True)
                return tuple(scores)

            search_evidence = None
            try:
                if args.search_completions:
                    searched = search_native_grammar(types, score, max_steps=args.max_steps,
                        max_nodes=args.search_nodes, completions=args.search_completions,
                        register_encoding=register_encoding, score_mode=args.search_score_mode)
                    def whole_graph_score(program, *, source=scored_source):
                        if time.monotonic() - started > args.max_seconds:
                            raise TimeoutError("native grammar run reached its finite bound")
                        sequence = native_sequence_for_encoding(source, program, tokenizer,
                            max_tokens=training["max_sequence_tokens"], register_encoding=register_encoding,
                            decision_basis=training.get("semantic_decision_basis", "program_atoms_v1"))
                        sequence, _control = apply_native_source_evidence(
                            sequence, source, tokenizer,
                            mode="source_text" if args.source_evidence == "source_pair_swap"
                            else args.source_evidence)
                        hidden = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                        return -native_loss(suffix, hidden, sequence, summed=True,
                                           scope="semantic_decisions").item()
                    chosen, graph_scores = select_search_proposal(searched, whole_graph_score)
                    search_evidence = {"expanded_nodes": searched.expanded_nodes,
                        "scored_decisions": searched.scored_decisions,
                        "scored_alternatives": searched.scored_alternatives,
                        "disconnected_leaves": searched.disconnected_leaves,
                        "frontier_nodes": searched.frontier_nodes,
                        "frontier_log_probability_bound": searched.frontier_log_probability_bound,
                        "halt_reason": searched.halt_reason,
                        "requested_top_k_proven": searched.requested_top_k_proven,
                        "selected_index": chosen, "complete_graph_scores": graph_scores,
                        "proposals": [{"program": candidate.result.program.to_dict(),
                            "log_probability": candidate.log_probability,
                            "decision_trace": candidate.result.trace,
                            "bound_forced_completion": candidate.result.bound_forced_completion}
                            for candidate in searched.candidates]}
                    generated = None if chosen is None else searched.candidates[chosen].result
                    # Grade reach only after target-blind generation and selection.
                    search_evidence["observed_program_reach"] = any(
                        semantic_programs_structurally_equivalent(candidate.result.program, example.program)
                        for candidate in searched.candidates)
                else:
                    generated = decode_native_grammar(types, score, max_steps=args.max_steps,
                                                       register_encoding=register_encoding)
                program = None if generated is None else generated.program
                trace = () if generated is None else generated.trace
                forced = False if generated is None else generated.bound_forced_completion
                status = "search_without_completion" if generated is None else "completed"
            except NativeGrammarIncompleteError as failure:
                program, trace = failure.program, failure.trace
                forced = False
                status = "disconnected_at_depth_bound"
            equivalent = status == "completed" and semantic_programs_structurally_equivalent(program, example.program)
            answer_correct = False
            if status == "completed":
                try:
                    answer_correct = program.run(public_inputs.values) == example.program.run(example.inputs)
                except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
                    answer_correct = False
            else:
                answer_correct = False
            row_body = {"plan_sha256": plan["plan_sha256"], "source_sha256": identity,
                        "source_evidence": args.source_evidence,
                        "scored_source_sha256": scored_source_sha256,
                        "construction": example.construction_id,
                        "public_input_receipt_sha256": public_inputs.receipt()["receipt_sha256"],
                        "program": None if program is None else program.to_dict(), "decode_status": status,
                        "program_equivalent": equivalent, "answer_correct": answer_correct,
                        "bound_forced_completion": forced,
                        "depth_bound_reached": forced or status == "disconnected_at_depth_bound",
                        "decision_trace": trace, "search": search_evidence,
                        "score_input_receipts": score_input_receipts,
                        "target_available_to_scorer": False}
            if args.prefix_strategy == "trie":
                row_body["prefix_execution"] = branches.receipt() if branches is not None else None
            row = {**row_body, "receipt_sha256": digest(row_body)}
            _save_if_absent(args.directory / "rows" / f"{identity}.json", row)
            rows.append(row)
            print(json.dumps({"stage": "grammar", "observed": len(rows), "population": len(examples),
                              "program_equivalent": equivalent,
                              "depth": 0 if program is None else program.depth, "decode_status": status}), flush=True)
        current = get_active_cortex_spec(force_refresh=True)
        execution_from_plan(training, check_installed=True)
        if (current is None or current.descriptor_sha256 != spec.descriptor_sha256
                or current.pointer_sha256 != spec.pointer_sha256
                or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
                       for name, sha in implementation.items())
                or selected_checkpoint(args.training_directory) != (training, selected)):
            raise ValueError("native grammar implementation, model, or checkpoint drifted")
        pair_totals = grammar_pair_totals(rows, dataset=args.dataset)
        result = {"schema": f"aura.semantic_native_grammar.{schema_version}",
                  "plan_sha256": plan["plan_sha256"],
                  "weight_mode": args.weight_mode, "dataset": args.dataset, "seed": seed,
                  "source_evidence": args.source_evidence,
                  "population": len(rows), "program_equivalent": sum(row["program_equivalent"] for row in rows),
                  "answer_correct": sum(row["answer_correct"] for row in rows),
                  **pair_totals,
                  "bound_forced_completion": sum(row["bound_forced_completion"] for row in rows),
                  "depth_bound_reached": sum(row["depth_bound_reached"] for row in rows),
                  "row_receipts": {row["source_sha256"]: row["receipt_sha256"] for row in rows},
                  "candidate_inventory": "none", "input_grounding": "semantic_public_character_inputs.v1",
                  "target_available_to_scorer": False, "serving_authority": False,
                  "qualification_evidence": False, "elapsed_seconds": time.monotonic() - started}
        if args.prefix_strategy == "trie":
            result["prefix_strategy"] = "trie"
        if source_basis is not None:
            result["source_cohort_basis"] = source_basis
        _save_if_absent(args.directory / "report.json", {**result, "receipt_sha256": digest(result)})
        print(json.dumps({key: value for key, value in result.items() if key != "row_receipts"}), flush=True)


if __name__ == "__main__":
    main()
