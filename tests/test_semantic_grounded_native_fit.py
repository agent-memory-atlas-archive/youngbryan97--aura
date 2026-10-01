"""Source-only joint native fitting uses bounded real prefixes and one lane."""

import hashlib
import json
import weakref
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from core.learning.semantic_grounded_binding_acquisition import (
    grounded_supervision_from_source_example,
)
from tests.test_semantic_program_shared_transducer import _shared_example
from tools.semantic_grounded_native_fit import fit_native_grounded_sources, native_capture_template


def sources():
    items = []
    for variant in (0, 2):
        item = _shared_example(three_steps=False, variant=variant, split="train")
        text = "".join(chr(token) for token in item.ir.source_token_ids)
        items.append(replace(item, ir=replace(item.ir, source_text_sha256=hashlib.sha256(text.encode()).hexdigest())))
    return tuple(items)


def test_native_capture_span_contract_is_blind_to_teacher_references():
    item = sources()[0]
    evidence = grounded_supervision_from_source_example(item).evidence
    changed = replace(item, ir=replace(item.ir, instructions=tuple(replace(instruction,
        args=tuple(reversed(instruction.args))) for instruction in item.ir.instructions)))
    assert native_capture_template(item, evidence, (0, 1)) == native_capture_template(changed, evidence, (0, 1))


@pytest.mark.parametrize("interrupted", [False, "update", "completion"])
def test_joint_cli_engine_shards_actual_prefixes_and_drops_model_before_lane_release(monkeypatch, tmp_path, interrupted):
    import mlx_lm
    from mlx_lm.models.qwen2 import Model, ModelArgs

    import core.runtime.model_lane_control as lane
    import tools.train_semantic_native_program as custody

    config = {"model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "vocab_size": 1024, "rms_norm_eps": 1e-6}
    model_path = tmp_path / "fixture-model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config))
    spec = SimpleNamespace(model_path=model_path, descriptor_sha256="a" * 64, pointer_sha256="b" * 64)
    monkeypatch.setattr(custody, "require_native_cortex_spec", lambda: spec)
    calls = []
    residents = []
    @contextmanager
    def owned(**kwargs):
        calls.append(("enter", kwargs))
        try:
            yield
        finally:
            assert all(reference() is None for reference in residents)
            calls.append(("exit", kwargs))
    monkeypatch.setattr(lane, "standalone_model_lane", owned)
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(map(ord, text))
        def decode(self, tokens, **kwargs):
            return "".join(map(chr, tokens))
    loads = []
    def load(path):
        loads.append(path)
        mx.random.seed(111)
        model = Model(ModelArgs.from_dict(config))
        residents.append(weakref.ref(model))
        return model, Tokenizer()
    monkeypatch.setattr(mlx_lm, "load", load)
    items = sources()
    examples = tuple(grounded_supervision_from_source_example(item) for item in items)
    options = dict(spec=spec, rank=2, layers=2, max_tokens=64, cache_bytes=8192, relation_width=8,
        fit_options={"steps": 4, "save_every": 2, "learning_rate": .01, "max_seconds": 30., "role_margin": .25})
    if interrupted == "update":
        _unused, uninterrupted = fit_native_grounded_sources(examples[:1], examples[1:], items,
            tmp_path / "uninterrupted", **options)
        import core.learning.semantic_grounded_binding_engine as module
        original = module._save_grounded_restart

        def interrupt(directory, state, *args):
            original(directory, state, *args)
            if state["step"] == 2:
                raise RuntimeError("native interruption")

        monkeypatch.setattr(module, "_save_grounded_restart", interrupt)
        with pytest.raises(RuntimeError, match="native interruption"):
            fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit", **options)
        monkeypatch.setattr(module, "_save_grounded_restart", original)
        engine, report = fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit",
            resume=True, **options)
        assert report["resume_from_step"] == 2
        assert report["history"] == uninterrupted["history"]
        assert report["selected_step"] == uninterrupted["selected_step"]
        for name in ("checkpoint-4.safetensors", "selected.safetensors"):
            expected = mx.load(str(tmp_path / "uninterrupted" / name))
            actual = mx.load(str(tmp_path / "fit" / name))
            assert set(actual) == set(expected)
            assert all(mx.array_equal(actual[key], value).item() for key, value in expected.items())
    elif interrupted == "completion":
        from core.runtime.file_write_gateway import get_file_write_gateway
        gateway = get_file_write_gateway()
        original = gateway.write_bytes_if_absent

        def interrupt(path, *args, **kwargs):
            if path.name == "completion.json":
                raise OSError("completion publication interrupted")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(gateway, "write_bytes_if_absent", interrupt)
        with pytest.raises(OSError, match="completion publication"):
            fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit", **options)
        monkeypatch.setattr(gateway, "write_bytes_if_absent", original)
        with pytest.raises(ValueError, match="source supervision"):
            fit_native_grounded_sources((replace(examples[0], environment="changed"),), examples[1:], items,
                tmp_path / "fit", resume=True, **options)
        assert len(loads) == 1
        engine, report = fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit",
            resume=True, **options)
    else:
        engine, report = fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit", **options)
    assert engine is None and report["joint_native_adapter_training"] and report["semantic_success"] is None
    assert len(loads) == (3 if interrupted == "update" else 1)
    assert [event for event, _ in calls] == (["enter", "exit"] * 3 if interrupted == "update" else ["enter", "exit"])
    assert calls[0][1]["require_exclusive"] and not calls[0][1]["allow_owner_eviction"]
    receipt = json.loads((tmp_path / "fit-native-custody" / "completion.json").read_text())
    assert receipt["state_store"]["sources"] == 2 and receipt["state_store"]["peak_resident_bytes"] <= 8192
    assert receipt["observed_adapter_parameters"] == receipt["adapter_projection"]["trainable_parameters"]
    assert not receipt["held_sources_scored"] and not receipt["serving_authority"]
    if interrupted == "completion":
        assert receipt["completion_recovery"] == "verified_saved_fit_without_model_loading"
        assert receipt["memory_envelope"] is None and receipt["peak_memory_bytes"] is None
    assert any(key.startswith("native_suffix.") for key in mx.load(str(tmp_path / "fit" / "selected.safetensors")))
    from tools.verify_semantic_grounded_fit import verify
    checked = verify(tmp_path / "fit")
    assert checked["native_acquisition_sha256"] and checked["artifacts_verified"]
    assert not checked["model_weights_loaded"] and not checked["held_sources_scored"]
    with pytest.raises(ValueError, match="already complete"):
        fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit", resume=True, **options)
    assert len(loads) == (3 if interrupted == "update" else 1)


def test_bad_source_population_or_token_bound_fails_before_model_loading(tmp_path):
    item = sources()[0]
    example = grounded_supervision_from_source_example(item)
    spec = SimpleNamespace(model_path=tmp_path, descriptor_sha256="a" * 64, pointer_sha256="b" * 64)
    with pytest.raises(ValueError, match="custody"):
        fit_native_grounded_sources((example,), (example,), (item,), tmp_path / "unused", spec=spec)
    assert not (tmp_path / "unused-native-custody").exists()
