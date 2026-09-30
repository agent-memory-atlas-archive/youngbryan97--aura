from types import SimpleNamespace

import pytest

from core.learning.semantic_program_campaign import _sha
from tools.freeze_semantic_source_folds import prepare_source_folds, source_folds


def _example(identity, construction, split):
    return SimpleNamespace(ir=SimpleNamespace(source_text_sha256=identity),
                           construction_id=construction, contrast_id=None, split=split)


def test_source_folds_exclude_withheld_rows_and_replay_deterministically():
    examples = [_example("a", "one", "train"), _example("b", "two", "train"),
                _example("c", "three", "train"), _example("v", "held", "validation")]
    folds = source_folds(examples)
    assert set(folds["assignments"]) == {"a", "b", "c"}
    assert folds["validation_used"] is False and folds["test_used"] is False
    assert folds["receipt_sha256"] == _sha({key: value for key, value in folds.items()
                                             if key != "receipt_sha256"})
    assert folds == source_folds(tuple(reversed(examples)))


def test_source_folds_require_train_and_withheld_population():
    train = [_example(str(index), str(index), "train") for index in range(3)]
    with pytest.raises(ValueError, match="train and withheld"):
        source_folds(train)
    with pytest.raises(ValueError, match="train and withheld"):
        source_folds([_example("v", "held", "validation")])
    with pytest.raises(ValueError, match="unsupported source fold axis"):
        source_folds([*train, _example("v", "held", "validation")], axis="other")


def _source_report(model):
    body = {"schema": "aura.compositional_source_training.v2",
            "transducer_receipt_sha256": model.receipt_sha256,
            "fit_complete": True,
            "representation_compatibility": {"source_feature_manifest_sha256s": {"one": "manifest"}}}
    return {**body, "report_sha256": _sha(body)}


def test_source_fit_preparation_binds_model_cohort_and_folds_without_measuring_a_candidate(monkeypatch):
    import tools.refit_semantic_argument_proposals as source

    model = SimpleNamespace(receipt_sha256="candidate")
    report = _source_report(model)
    examples = [_example("a", "one", "train"), _example("b", "two", "train"),
                _example("c", "three", "train"), _example("v", "held", "validation")]
    seen = []
    def load(candidate, training_report, bundles):
        seen.append((candidate, training_report, bundles))
        return examples
    monkeypatch.setattr(source, "load_source_examples", load)
    folds, receipt = prepare_source_folds(model, report, ["one=bundle"], source_fit_only=True)
    assert seen == [(model, report, ["one=bundle"])]
    assert receipt["candidate"] == model.receipt_sha256
    assert receipt["source_report_sha256"] == report["report_sha256"]
    assert receipt["source_feature_manifest_sha256s"] == {"one": "manifest"}
    assert receipt["folds_receipt_sha256"] == folds["receipt_sha256"]
    assert receipt["candidate_report_sha256"] is None and receipt["source_fit_only"] is True
    assert all(receipt[key] is False for key in (
        "validation_used", "test_used", "candidate_evaluation_performed", "qualification_evidence", "serving_authority"))
    assert receipt["receipt_sha256"] == _sha({key: value for key, value in receipt.items()
                                               if key != "receipt_sha256"})
    assert set(folds["assignments"]) == {"a", "b", "c"}


@pytest.mark.parametrize("options", [{}, {"source_fit_only": True, "candidate_report": {}},
                                     {"source_fit_only": 1}])
def test_source_fold_preparation_requires_one_explicit_authority_mode(options):
    model = SimpleNamespace(receipt_sha256="candidate")
    with pytest.raises(ValueError, match="choose either"):
        prepare_source_folds(model, _source_report(model), ["one=bundle"], **options)


def test_source_fit_preparation_still_rejects_tampered_parent_and_incompatible_bundles(monkeypatch):
    import tools.refit_semantic_argument_proposals as source

    model = SimpleNamespace(receipt_sha256="candidate")
    report = _source_report(model)
    with pytest.raises(ValueError, match="report identity"):
        prepare_source_folds(model, {**report, "transducer_receipt_sha256": "other"}, [], source_fit_only=True)
    def incompatible(*_args):
        raise ValueError("source manifest differs")
    monkeypatch.setattr(source, "load_source_examples", incompatible)
    with pytest.raises(ValueError, match="manifest differs"):
        prepare_source_folds(model, report, ["one=wrong"], source_fit_only=True)


def test_default_preparation_still_requires_a_verified_measured_candidate_report():
    model = SimpleNamespace(receipt_sha256="candidate")
    with pytest.raises(ValueError, match="candidate report identity"):
        prepare_source_folds(model, _source_report(model), [], candidate_report={})


@pytest.mark.parametrize("completion", [None, False, 1])
def test_source_fit_preparation_refuses_incomplete_or_unmeasured_fit(completion):
    model = SimpleNamespace(receipt_sha256="candidate")
    body = {key: value for key, value in _source_report(model).items() if key != "report_sha256"}
    body["fit_complete"] = completion
    report = {**body, "report_sha256": _sha(body)}
    with pytest.raises(ValueError, match="completed source fit"):
        prepare_source_folds(model, report, [], source_fit_only=True)
