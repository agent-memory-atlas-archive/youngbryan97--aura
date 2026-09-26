"""Public source-coordinate relabeling preserves bindings, including equal values."""

import hashlib
import json
from dataclasses import replace

import pytest

from core.learning.semantic_program_corpus import CharacterSpan
from core.learning.semantic_program_corpus_natural import build_semantic_program_natural_request_corpus
from tools.semantic_native_paraphrase_interventions import render_native_paraphrase
from tools.semantic_native_retained_sources import load_retained_native_sources, public_order_example


def example():
    return render_native_paraphrase(build_semantic_program_natural_request_corpus(
        seed=1618033, examples_per_schema_domain=1)[0], style="definition")


def test_source_order_identity_does_not_change_an_already_aligned_example():
    original = example()
    assert public_order_example(original) == original


def test_reversed_annotation_coordinates_preserve_causal_binding_and_definitions():
    from core.learning.procedure_induction import Instruction

    original = example()
    n = len(original.inputs)
    reverse = {index: n - index - 1 for index in range(n)}
    annotations = tuple(replace(row, instruction=Instruction(row.instruction.op,
        tuple(reverse.get(index, index) for index in row.instruction.args))) for row in original.instructions)
    permuted = replace(original, inputs=original.inputs[::-1], input_spans=original.input_spans[::-1],
        instructions=annotations, register_definition_spans=(
            original.register_definition_spans[:n][::-1] + original.register_definition_spans[n:]))
    assert public_order_example(permuted) == original
    assert permuted.program.run(permuted.inputs) == original.program.run(original.inputs)


def test_literal_identity_uses_source_span_not_a_value_lookup():
    original = example()
    corrupted = replace(original, input_spans=(original.input_spans[0],) * len(original.inputs))
    with pytest.raises(ValueError, match="literal identity"):
        public_order_example(corrupted)
    corrupted = replace(original, input_spans=(CharacterSpan(0, 1), *original.input_spans[1:]))
    with pytest.raises(ValueError, match="literal identity"):
        public_order_example(corrupted)
    with pytest.raises(ValueError, match="envelope"):
        public_order_example(replace(original, input_spans=original.input_spans[:-1]))


def test_equal_valued_literals_still_have_distinct_roles():
    from core.learning.procedure_induction import Instruction

    raw = build_semantic_program_natural_request_corpus(
        seed=1618033, examples_per_schema_domain=1)[0]
    original = render_native_paraphrase(replace(raw, inputs=(7,) * len(raw.inputs)), style="definition")
    n = len(original.inputs)
    annotations = tuple(replace(row, instruction=Instruction(row.instruction.op,
        tuple(n - index - 1 if index < n else index for index in row.instruction.args)))
        for row in original.instructions)
    permuted = replace(original, input_spans=original.input_spans[::-1], instructions=annotations,
        register_definition_spans=original.register_definition_spans[:n][::-1]
        + original.register_definition_spans[n:])
    assert public_order_example(permuted) == original


def retained_fixture(tmp_path, monkeypatch, *, duplicate=False):
    from core.learning.semantic_program_campaign import _sha

    original = example()
    corpora = {
        "a": (replace(original, split="validation"), replace(original, split="train")),
        "b": (replace(original, split="validation", source_text=original.source_text + " "),
              replace(original, split="test", source_text=original.source_text + "  ")),
        "c": (replace(original, split="train", source_text=original.source_text + "   "),),
    }
    if duplicate:
        corpora["b"] = (replace(original, split="validation"),)
    manifests, paths = {}, []
    for name in corpora:
        directory = tmp_path / name
        directory.mkdir()
        manifests[name] = f"manifest-{name}"
        (directory / "manifest.json").write_text(json.dumps({
            "manifest_sha256": manifests[name], "cohort": name}))
        paths.append(f"{name}={directory}")
    monkeypatch.setattr(
        "core.learning.semantic_program_feature_materialization.rebuild_semantic_feature_selection",
        lambda manifest: (None, corpora[manifest["cohort"]]))
    body = {"schema": "aura.compositional_source_training.v2", "fit_complete": True,
            "representation_compatibility": {"source_feature_manifest_sha256s": manifests}}
    report = {**body, "report_sha256": _sha(body)}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    return report_path, paths, report


def test_retained_source_loader_binds_all_manifests_and_exposed_split(tmp_path, monkeypatch):
    report_path, paths, _ = retained_fixture(tmp_path, monkeypatch)
    examples, basis = load_retained_native_sources(report_path, paths, split="validation", count=2)
    assert len(examples) == 2
    assert all(item.split == "validation" for item in examples)
    assert basis["source_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert basis["available_population"] == 2
    assert basis["cohort_counts"] == {"a": 1, "b": 1}
    assert basis["exposure"] == "previously_exposed_development"
    assert load_retained_native_sources(report_path, paths[::-1], split="validation", count=1)[0] == examples[:1]
    assert load_retained_native_sources(report_path, paths, split="test", count=1)[1]["cohort_counts"] == {"b": 1}
    with pytest.raises(ValueError, match="at least three"):
        load_retained_native_sources(report_path, paths[:-1], split="validation", count=1)
    (tmp_path / "a" / "manifest.json").write_text(json.dumps({"manifest_sha256": "other"}))
    with pytest.raises(ValueError, match="manifest differs"):
        load_retained_native_sources(report_path, paths, split="validation", count=1)


def test_retained_source_loader_rejects_incomplete_forged_and_repeated_populations(tmp_path, monkeypatch):
    from core.learning.semantic_program_campaign import _sha

    report_path, paths, report = retained_fixture(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match="insufficient or duplicated"):
        load_retained_native_sources(report_path, paths, split="validation", count=2)
    for changes in ({"fit_complete": False}, {"schema": "arbitrary"}, {"report_sha256": "forged"}):
        body = {key: value for key, value in {**report, **changes}.items() if key != "report_sha256"}
        updated = {**body, "report_sha256": _sha(body) if "report_sha256" not in changes else "forged"}
        report_path.write_text(json.dumps(updated))
        with pytest.raises(ValueError, match="not complete or source-bound"):
            load_retained_native_sources(report_path, paths, split="validation", count=1)


@pytest.mark.parametrize("split,count", [("train", 1), ("validation", True), ("test", 0), ("test", 501)])
def test_retained_source_loader_rejects_fit_data_and_unbounded_populations(split, count):
    with pytest.raises(ValueError, match="bounded development split"):
        load_retained_native_sources("unused", (), split=split, count=count)
