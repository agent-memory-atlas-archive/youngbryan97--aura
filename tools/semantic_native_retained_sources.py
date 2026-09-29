"""Recover exposed development requests with public-order grading coordinates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import zip_longest
from pathlib import Path

from core.learning.procedure_induction import Instruction
from core.learning.semantic_public_inputs import semantic_public_character_inputs


def public_order_example(example):
    """Relabel annotated input registers; never interpret a request for the scorer."""
    public = semantic_public_character_inputs(example.source_text)
    positions = {(literal.character_start, literal.character_end): index
                 for index, literal in enumerate(public.literals)}
    if len(public.values) != len(example.inputs):
        raise ValueError("retained source public literal population differs")
    mapping = tuple(positions.get((span.start, span.end)) for span in example.input_spans)
    if (len(mapping) != len(example.inputs) or any(index is None for index in mapping)
            or len(set(mapping)) != len(mapping)
            or any(public.values[index] != example.inputs[old] for old, index in enumerate(mapping))):
        raise ValueError("retained source public literal identity differs")
    inverse = tuple(mapping.index(index) for index in range(len(mapping)))
    instructions = tuple(replace(annotation, instruction=Instruction(annotation.instruction.op,
        tuple(mapping[index] if index < len(mapping) else index for index in annotation.instruction.args)))
        for annotation in example.instructions)
    definitions = example.register_definition_spans
    if definitions:
        definitions = tuple(definitions[index] for index in inverse) + definitions[len(mapping):]
    aligned = replace(example, inputs=public.values,
        input_spans=tuple(example.input_spans[index] for index in inverse),
        instructions=instructions, register_definition_spans=definitions)
    if aligned.program.run(aligned.inputs) != example.program.run(example.inputs):
        raise ValueError("retained source coordinate relabeling changed executable meaning")
    return aligned


def load_retained_native_sources(source_report_path, bundles, *, split, count):
    """Rebuild pinned public corpora, without loading model or hidden-state arrays."""
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_feature_materialization import rebuild_semantic_feature_selection
    from tools.refit_semantic_argument_proposals import source_bundle_arguments

    if split not in {"validation", "test"} or type(count) is not int or not 1 <= count <= 500:
        raise ValueError("retained native evaluation needs a bounded development split")
    path = Path(source_report_path)
    raw = path.read_bytes()
    source = json.loads(raw)
    if (source.get("schema") not in {"aura.compositional_source_training.v1",
                                     "aura.compositional_source_training.v2"}
            or source.get("report_sha256") != _sha({key: value for key, value in source.items()
                                            if key != "report_sha256"})
            or source.get("fit_complete") is not True):
        raise ValueError("retained native source report is not complete or source-bound")
    expected = source["representation_compatibility"]["source_feature_manifest_sha256s"]
    paths = source_bundle_arguments(source, bundles=bundles)
    groups, manifest_paths, manifests = {}, {}, {}
    for argument in paths:
        name, _, directory = argument.partition("=")
        manifest = json.loads((Path(directory) / "manifest.json").read_bytes())
        if manifest.get("manifest_sha256") != expected[name]:
            raise ValueError("retained native source manifest differs from the report")
        _, examples = rebuild_semantic_feature_selection(manifest)
        group = [public_order_example(example) for example in examples if example.split == split]
        if group:
            groups[name] = sorted(group, key=lambda example: hashlib.sha256(example.source_text.encode()).hexdigest())
        manifest_paths[name] = str(Path(directory).resolve())
        manifests[name] = manifest["manifest_sha256"]
    if set(manifests) != set(expected):
        raise ValueError("retained native source inventory is incomplete")
    ordered = tuple(example for cohort in zip_longest(*(groups[key] for key in sorted(groups)))
                    for example in cohort if example is not None)
    identities = [hashlib.sha256(example.source_text.encode()).hexdigest() for example in ordered]
    if count > len(ordered) or len(set(identities)) != len(identities):
        raise ValueError("retained native source population is insufficient or duplicated")
    basis = {"source_report_path": str(path.resolve()),
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_paths": manifest_paths, "source_manifest_sha256s": manifests,
        "split": split, "available_population": len(ordered),
        "cohort_counts": {key: len(group) for key, group in sorted(groups.items())},
        "population_order": "cohort_round_robin_source_digest",
        "target_coordinates": "public_literal_character_order_v1",
        "exposure": "previously_exposed_development"}
    return ordered[:count], basis


def retained_native_source_window(source_report_path, bundles, *, split, offset, count):
    """Partition all 500 ordered requests without selecting by measured outcomes."""
    if (type(offset) is not int or type(count) is not int
            or offset < 0 or count < 1 or offset + count > 500):
        raise ValueError("retained native source window exceeds the complete population")
    examples, basis = load_retained_native_sources(source_report_path, bundles,
                                                   split=split, count=500)
    identities = [hashlib.sha256(example.source_text.encode()).hexdigest() for example in examples]
    from tools.evaluate_semantic_native_checkpoint import digest

    window = {"offset": offset, "count": count, "population": 500,
              "ordered_sources_sha256": digest(identities)}
    return examples[offset:offset + count], basis, window
