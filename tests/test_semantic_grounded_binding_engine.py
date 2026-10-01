"""Source-only fit custody and native adapter/pointer gradient integration."""

import hashlib
import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent
from core.learning.semantic_grounded_binding_engine import (
    GroundedBindingEngine,
    GroundedBindingEvidence,
    GroundedBindingSupervision,
    NativeGroundedCapture,
    fit_grounded_binding,
    project_grounded_evidence,
)
from core.learning.semantic_relational_pointer import RelationalBindingPointer


def source(identity, *, environment="source"):
    records = tuple(ContextReferent(identity, str(index), "Number", "observed:" + str(index)) for index in range(2))
    context = BindingContext(records, {"Number": None})
    role = BindingRole("operand", "value", "Number")
    evidence = GroundedBindingEvidence(identity, context, (role,),
        {role.identity: mx.array([[1., 0., 1., 0.]])},
        {role.identity: mx.array([[.5, 0., 1., 0.]])},
        {records[0].key: mx.array([[0., 1., 0., 1.]]), records[1].key: mx.array([[1., 0., 1., 0.]])})
    return GroundedBindingSupervision(evidence, ((records[1].key,),),
        graphs=((records[0].key,), (records[1].key,)), positive_graphs=(1,), environment=environment)


def test_fitted_actual_evidence_reloads_and_reaches_the_contextual_solver(tmp_path):
    mx.random.seed(3)
    pointer = RelationalBindingPointer(4, relation_width=8)
    fit, calibration = source("fit"), source("calibration")
    engine, report = fit_grounded_binding(pointer, (fit,), (calibration,), tmp_path / "fit",
        steps=16, save_every=4, learning_rate=.02, max_seconds=30.)
    assert report["selected_step"] > 0 and report["semantic_success"] is None
    assert report["serving_authority"] is False
    restored = GroundedBindingEngine.load(tmp_path / "fit")
    assert restored.costs(calibration.evidence) == engine.costs(calibration.evidence)
    result = restored.resolve(calibration.evidence)
    assert result.executable and result.bindings == (("operand", ("calibration", "1")),)
    receipt = restored.binding_receipt(calibration.evidence, result)
    assert receipt["roles"][0]["selected_source"] == ("calibration", "1")
    assert len(receipt["roles"][0]["alternatives"]) == 2 and not receipt["margin_is_probability"]
    saved = tmp_path / "fit" / "selected.safetensors"
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == report["weights_sha256"]
    saved.write_bytes(saved.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="weights differ"):
        GroundedBindingEngine.load(tmp_path / "fit")


def test_fit_rejects_overlap_inadmissible_positives_and_nonfit_nuisance_basis(tmp_path):
    from core.learning.semantic_binding_invariance import BindingNuisanceProjection

    pointer = RelationalBindingPointer(4)
    fit = source("fit")
    for training, calibration, options in (
        ((fit,), (fit,), {}),
        ((replace(fit, positives=((("absent", "entity"),),)),), (source("cal"),), {}),
        ((fit,), (source("cal"),), {"nuisance_projection": BindingNuisanceProjection(np.eye(4)[:, :1], ("cal",))}),
    ):
        with pytest.raises(ValueError):
            fit_grounded_binding(pointer, training, calibration, tmp_path / "unused", steps=2, save_every=1, **options)
    assert not (tmp_path / "unused").exists()


def test_actual_fit_schedule_is_fit_only_and_partial_epochs_are_not_hidden(tmp_path):
    with pytest.raises(ValueError, match="fit-only"):
        fit_grounded_binding(RelationalBindingPointer(4), (source("fit"),), (source("cal"),),
            tmp_path / "invalid", steps=2, save_every=1, training_schedule=("fit", "cal"))
    assert not (tmp_path / "invalid").exists()
    _engine, report = fit_grounded_binding(RelationalBindingPointer(4, relation_width=4),
        (source("fit1"), source("fit2"), source("fit3")), (source("cal"),),
        tmp_path / "partial", steps=2, save_every=1, training_schedule=("fit3", "fit1"), max_seconds=30.)
    assert report["training_schedule"] == ("fit3", "fit1")
    assert report["primary_sources_visited"] == 2 and not report["complete_primary_epoch"]


def test_sampling_receipt_must_describe_the_schedule_not_a_claimed_full_epoch(tmp_path):
    with pytest.raises(ValueError, match="sampling receipt"):
        fit_grounded_binding(RelationalBindingPointer(4), (source("fit"),), (source("cal"),),
            tmp_path / "invalid", steps=2, save_every=1, training_schedule=("fit", "fit"),
            sampling_receipt={"primary_updates": 2})
    assert not (tmp_path / "invalid").exists()


def test_nuisance_projection_keeps_native_state_gradients():
    from core.learning.semantic_binding_invariance import BindingNuisanceProjection

    example = source("fit").evidence
    basis = BindingNuisanceProjection(np.eye(4)[:, :1], ("fit",))
    value = example.operations["operand"]
    gradient = mx.grad(lambda x: mx.sum(project_grounded_evidence(
        replace(example, operations={"operand": x}), basis).operations["operand"]))(value)
    assert mx.array_equal(gradient, mx.array([[0., 1., 1., 1.]])).item()


def test_domain_adversarial_fit_keeps_environment_labels_out_of_served_parameters(tmp_path):
    mx.random.seed(6)
    pointer = RelationalBindingPointer(4, relation_width=4)
    _engine, report = fit_grounded_binding(pointer,
        (source("fit1", environment="a"), source("fit2", environment="b")), (source("cal"),),
        tmp_path / "fit", steps=4, save_every=2, group_eta=.1, domain_reversal=.3, max_seconds=30.)
    assert report["training_nuisance_sha256"] and not report["inference_family_labels"]
    assert all("nuisance" not in name for name in mx.load(str(tmp_path / "fit" / "selected.safetensors")))
    assert GroundedBindingEngine.load(tmp_path / "fit").pointer.to_contract() == pointer.to_contract()


def test_joint_native_fit_changes_actual_suffix_adapters_and_loads_one_combined_state(tmp_path):
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.semantic_program_ir import TokenSpan
    from tools.semantic_native_adapters import adapter_contract, install_native_adapters

    mx.random.seed(17)
    config = {"model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
              "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
              "vocab_size": 32, "rms_norm_eps": 1e-6}
    model = Model(ModelArgs.from_dict(config))
    model.freeze()
    model.eval()
    prefix, suffix = FrozenDecoderPrefix(model, split_at=1), NativeDecoderSuffix(model, split_at=1)
    contract = adapter_contract(rank=2, layers=2, sites="native_topology_v1")
    plan = {"rank": 2, "suffix_layers": 2, "adapter_keys": contract["keys"], "adapter_contract": contract}
    install_native_adapters(model, plan)
    hidden = prefix.capture(mx.array([[1, 7, 3, 9]], dtype=mx.int32))
    examples, captures = [], {}
    for identity in ("fit", "cal"):
        evidence = source(identity).evidence
        records, roles = evidence.context.referents, evidence.roles
        capture = NativeGroundedCapture(identity, hidden, evidence.context, roles, (0, 1),
            {"operand": TokenSpan(1, 2)}, {"operand": TokenSpan(2, 3)},
            {records[0].key: TokenSpan(0, 1), records[1].key: TokenSpan(3, 4)})
        captures[identity] = capture
        examples.append(GroundedBindingSupervision(capture.capture(suffix), ((records[1].key,),)))
    baseline = {key: mx.array(value) for key, value in tree_flatten(suffix.trainable_parameters())}
    pointer = RelationalBindingPointer(16, depths=2, relation_width=8)
    engine, report = fit_grounded_binding(pointer, (examples[0],), (examples[1],), tmp_path / "fit",
        steps=8, save_every=4, learning_rate=.01, max_seconds=30., native_suffix=suffix,
        native_captures=captures, native_contract=plan)
    last = mx.load(str(tmp_path / "fit" / "checkpoint-8.safetensors"))
    assert any(not mx.array_equal(value, last["native_suffix." + key]).item() for key, value in baseline.items())
    assert report["joint_native_adapter_training"] and len(report["native_captures"]) == 2
    with pytest.raises(ValueError, match="caller-owned"):
        GroundedBindingEngine.load(tmp_path / "fit")
    restored = GroundedBindingEngine.load(tmp_path / "fit", native_suffix=suffix, native_contract=plan)
    expected = engine.resolve_native(captures["cal"])
    assert restored.resolve_native(captures["cal"]).bindings == expected.bindings
    assert all("lora_" in key for key in last)
    payload = json.loads((tmp_path / "fit" / "report.json").read_text())
    assert payload["semantic_success"] is None and not payload["serving_authority"]


def test_equal_denotation_and_valid_same_source_roles_remain_distinct_queries():
    from core.learning.semantic_grounded_binding_engine import grounded_source_loss
    from core.learning.semantic_relational_pointer import semantic_role_features

    item = source("equal")
    records = tuple(replace(record, attributes={"number": "5"}) for record in item.evidence.context.referents)
    context = BindingContext(records, {"Number": None})
    roles = (BindingRole("first", "minuend", "Number"), BindingRole("second", "subtrahend", "Number"))
    evidence = replace(item.evidence, context=context, roles=roles,
        operations={role.identity: item.evidence.operations["operand"] for role in roles},
        mentions={role.identity: item.evidence.mentions["operand"] for role in roles})
    supervision = GroundedBindingSupervision(evidence, ((records[0].key,), (records[1].key,)))
    pointer = RelationalBindingPointer(4, relation_width=16, rounds=0)
    import mlx.nn as nn
    import mlx.optimizers as optim
    optimizer = optim.Adam(learning_rate=.025)
    for _ in range(32):
        loss, gradients = nn.value_and_grad(pointer, lambda owner:
            grounded_source_loss(owner, supervision, role_margin=.5))(pointer)
        optimizer.update(pointer, gradients)
        mx.eval(pointer.parameters(), loss)
    engine = GroundedBindingEngine(pointer)
    result = engine.resolve(evidence)
    assert result.executable and dict(result.bindings) == {"first": records[0].key, "second": records[1].key}
    # Role conditioning supplies evidence, not an invalid all-different rule.
    same = replace(evidence, roles=tuple(replace(role, referents=(records[0].key,)) for role in roles))
    assert {key for _, key in engine.resolve(same).bindings} == {records[0].key}
    queries = semantic_role_features(roles)
    arrays, mask = evidence.arrays()
    scores = pointer(*arrays, allowed=mask, role_features=queries)
    lesioned = pointer(*arrays, allowed=mask, role_features=mx.zeros_like(queries))
    assert mx.array_equal(lesioned[0], lesioned[1]).item()
    assert not mx.array_equal(scores[0], scores[1]).item()


def test_research_artifact_writer_rejects_unowned_namespaces_and_publication_names(tmp_path):
    from core.learning.semantic_grounded_binding_engine import _write_binding_artifact

    with pytest.raises(ValueError, match="no claimed fresh run"):
        _write_binding_artifact(tmp_path / "selected.safetensors", b"unowned")
    for name in ("serving.json", "production.safetensors", "checkpoint--1.safetensors", "checkpoint-01.safetensors"):
        with pytest.raises(ValueError, match="fixed schema namespace"):
            _write_binding_artifact(tmp_path / name, b"unowned")
    assert not tuple(tmp_path.iterdir())


def test_exact_restart_matches_uninterrupted_adam_group_and_adversary_updates(tmp_path, monkeypatch):
    from mlx.utils import tree_flatten

    import core.learning.semantic_grounded_binding_engine as module

    training = (source("fit1", environment="a"), source("fit2", environment="b"))
    calibration = (source("cal"),)
    options = dict(steps=8, save_every=2, learning_rate=.02, max_seconds=30., group_eta=.1, domain_reversal=.2)
    mx.random.seed(49)
    _engine, complete = fit_grounded_binding(RelationalBindingPointer(4, relation_width=4),
        training, calibration, tmp_path / "complete", **options)
    original = module._save_grounded_restart

    def interrupt(directory, state, *args):
        original(directory, state, *args)
        if state["step"] == 4:
            raise RuntimeError("simulated process loss after durable update")

    monkeypatch.setattr(module, "_save_grounded_restart", interrupt)
    mx.random.seed(49)
    with pytest.raises(RuntimeError, match="process loss"):
        fit_grounded_binding(RelationalBindingPointer(4, relation_width=4), training, calibration,
                             tmp_path / "recovered", **options)
    monkeypatch.setattr(module, "_save_grounded_restart", original)
    mx.random.seed(900)  # Recovery must restore the actual RNG, not use this initialization.
    engine, recovered = fit_grounded_binding(RelationalBindingPointer(4, relation_width=4), training,
        calibration, tmp_path / "recovered", resume=True, **options)
    assert recovered["resume_from_step"] == 4 and recovered["history"] == complete["history"]
    assert recovered["selected_step"] == complete["selected_step"]
    expected = mx.load(str(tmp_path / "complete" / "checkpoint-8.safetensors"))
    actual = mx.load(str(tmp_path / "recovered" / "checkpoint-8.safetensors"))
    assert all(mx.array_equal(actual[key], value).item() for key, value in expected.items())
    selected = GroundedBindingEngine.load(tmp_path / "recovered")
    assert all(mx.array_equal(value, dict(tree_flatten(selected.pointer.parameters()))[key]).item()
               for key, value in tree_flatten(engine.pointer.parameters()))


def test_restart_rejects_changed_source_schedule_and_corrupted_generation(tmp_path, monkeypatch):
    import core.learning.semantic_grounded_binding_engine as module

    training, calibration = (source("fit"),), (source("cal"),)
    options = dict(steps=4, save_every=2, learning_rate=.02, max_seconds=30.)
    original = module._save_grounded_restart

    def interrupt(directory, state, *args):
        original(directory, state, *args)
        if state["step"] == 2:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "_save_grounded_restart", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        fit_grounded_binding(RelationalBindingPointer(4, relation_width=4), training, calibration,
                             tmp_path / "fit", **options)
    monkeypatch.setattr(module, "_save_grounded_restart", original)
    with pytest.raises(ValueError, match="owner or source"):
        fit_grounded_binding(RelationalBindingPointer(4, relation_width=4), training, calibration,
            tmp_path / "fit", resume=True, **{**options, "learning_rate": .01})
    with pytest.raises(ValueError, match="existing complete"):
        fit_grounded_binding(RelationalBindingPointer(4, relation_width=4), training, calibration,
                             tmp_path / "absent", resume=True, **options)
    pointer = json.loads((tmp_path / "fit" / "resume.json").read_bytes())
    generation = tmp_path / "fit" / pointer["file"]
    generation.write_bytes(generation.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        fit_grounded_binding(RelationalBindingPointer(4, relation_width=4), training, calibration,
                             tmp_path / "fit", resume=True, **options)


def test_restart_uses_previous_complete_generation_after_pointer_publication_failure(tmp_path, monkeypatch):
    import core.learning.semantic_grounded_binding_engine as module

    training, calibration = (source("fit"),), (source("cal"),)
    options = dict(steps=4, save_every=2, learning_rate=.02, max_seconds=30.)
    mx.random.seed(71)
    _engine, expected = fit_grounded_binding(RelationalBindingPointer(4, relation_width=4),
        training, calibration, tmp_path / "complete", **options)
    original = module._write_binding_artifact
    publications = []

    def fail_publication(path, payload):
        if path.name == "resume.json":
            publications.append(json.loads(payload)["step"])
            if publications[-1] == 2:
                raise OSError("lost before pointer publication")
        return original(path, payload)

    monkeypatch.setattr(module, "_write_binding_artifact", fail_publication)
    mx.random.seed(71)
    with pytest.raises(OSError, match="pointer publication"):
        fit_grounded_binding(RelationalBindingPointer(4, relation_width=4), training, calibration,
                             tmp_path / "fit", **options)
    assert json.loads((tmp_path / "fit" / "resume.json").read_bytes())["step"] == 0
    assert len(tuple((tmp_path / "fit").glob("resume-*.safetensors"))) == 2
    # The mutable selection may already have changed; the generation owns it.
    (tmp_path / "fit" / "selected.safetensors").write_bytes(b"partial selection")
    monkeypatch.setattr(module, "_write_binding_artifact", original)
    _engine, recovered = fit_grounded_binding(RelationalBindingPointer(4, relation_width=4),
        training, calibration, tmp_path / "fit", resume=True, **options)
    assert recovered["resume_from_step"] == 0 and recovered["history"] == expected["history"]
    assert recovered["weights_sha256"] == expected["weights_sha256"]
    assert module.verify_grounded_fit_checkpoint(tmp_path / "fit")["receipt_sha256"] == recovered["receipt_sha256"]
    from tools.verify_semantic_grounded_fit import verify
    receipt = verify(tmp_path / "fit")
    assert receipt["artifacts_verified"] and receipt["completed_updates"] == 4
    assert receipt["source_fit_count"] == receipt["source_calibration_count"] == 1
    assert not receipt["model_weights_loaded"] and not receipt["qualification_evidence"]
    checkpoint = tmp_path / "fit" / f"checkpoint-{recovered['selected_step']}.safetensors"
    checkpoint.write_bytes((tmp_path / "fit" / "checkpoint-0.safetensors").read_bytes())
    if recovered["selected_step"] > 0:
        with pytest.raises(ValueError, match="selected checkpoint"):
            module.verify_grounded_fit_checkpoint(tmp_path / "fit")
    with pytest.raises(ValueError, match="already complete"):
        fit_grounded_binding(RelationalBindingPointer(4, relation_width=4), training, calibration,
                             tmp_path / "fit", resume=True, **options)


def test_restart_refuses_symlinked_generation_before_model_update(tmp_path):
    import mlx.nn as nn
    import mlx.optimizers as optim

    import core.learning.semantic_grounded_binding_engine as module
    from core.runtime.file_read_gateway import StableFileReadError

    model = nn.Module()
    model.pointer = RelationalBindingPointer(4, relation_width=4)
    optimizer = optim.Adam(learning_rate=.02)
    optimizer.init(model.trainable_parameters())
    owner = tmp_path / "fit"
    owner.mkdir()
    (owner / "fit-owner.json").write_text("{}")
    from mlx.utils import tree_flatten
    module._save_grounded_restart(owner, {"identity": "a" * 64, "step": 0}, model, optimizer,
                                  dict(tree_flatten(model.pointer.trainable_parameters())))
    receipt = json.loads((owner / "resume.json").read_bytes())
    path = owner / receipt["file"]
    external = tmp_path / "external.safetensors"
    path.rename(external)
    path.symlink_to(external)
    with pytest.raises(StableFileReadError, match="symlink_rejected"):
        module._load_grounded_restart(owner, "a" * 64, model, optimizer, model.pointer)
