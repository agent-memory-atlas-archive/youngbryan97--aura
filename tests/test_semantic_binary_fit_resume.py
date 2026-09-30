"""Saved binary iterates preserve work, never bypass fresh convergence."""

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_bounded_binary_fit import (
    BinaryFeatureRows,
    binary_fit_checkpoint_receipt,
    binary_fit_checkpoint_scope,
    fit_bounded_binary_head,
)
from core.learning.semantic_fit_checkpoint import fit_identity, read_fit_archive


def fixture():
    rng = np.random.default_rng(271828)
    x = rng.normal(size=(160, 12)).astype(np.float32)
    y = (x[:, 0] + .2 * x[:, 1] + rng.normal(size=160) > .8).astype(np.int8)
    return x, y, rng.uniform(.2, 2., len(y))


def test_iteration_bound_retains_a_warm_start_and_next_attempt_must_converge(tmp_path):
    x, y, weights = fixture()
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        with pytest.raises(ValueError, match="did not converge"):
            fit_bounded_binary_head(BinaryFeatureRows(x), y, sample_weight=weights,
                                    tolerance=1e-9, max_iter=1)
    path, = tmp_path.glob("*.npz")
    body, arrays = read_fit_archive(path, ("weight", "center"))
    assert body["status"] == "iterating" and body["iterations"] == 1
    assert np.any(arrays["weight"] != arrays["center"])
    progress = []
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        actual, bias = fit_bounded_binary_head(BinaryFeatureRows(x), y, sample_weight=weights,
                                             tolerance=1e-9, progress=progress.append)
        receipt = binary_fit_checkpoint_receipt()
    expected, expected_bias = fit_bounded_binary_head(BinaryFeatureRows(x), y,
                                                     sample_weight=weights, tolerance=1e-9)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert bias == pytest.approx(expected_bias, abs=3e-5)
    assert progress[0]["resumed_iterations"] == 1
    assert progress[-1]["stage"] == "binary_fit_complete"
    assert receipt["cached_convergence_is_authority"] is False
    assert receipt["records"][0]["converged"] is True
    assert receipt["records"][0]["resumed_iterations"] == 1
    assert receipt["receipt_sha256"] == fit_identity({key: value for key, value in receipt.items()
                                                      if key != "receipt_sha256"})
    assert binary_fit_checkpoint_receipt() is None


def test_even_a_converged_archive_cannot_replace_a_new_solver_result(tmp_path, monkeypatch):
    x, y, weights = fixture()
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        fit_bounded_binary_head(x, y, sample_weight=weights)
    calls = []
    def unfinished(objective, start, **kwargs):
        calls.append(start.copy())
        value, _gradient = objective(start)
        return SimpleNamespace(x=start, fun=value, success=False, nit=0, message="unfinished")
    monkeypatch.setattr("scipy.optimize.minimize", unfinished)
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        with pytest.raises(ValueError, match="did not converge"):
            fit_bounded_binary_head(x, y, sample_weight=weights)
    assert len(calls) == 1 and np.any(calls[0] != 0)


@pytest.mark.parametrize("changed", ["features", "order", "labels", "weights", "context", "tolerance"])
def test_warm_starts_never_cross_objective_or_partition_identity(tmp_path, changed):
    x, y, weights = fixture()
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        fit_bounded_binary_head(x, y, sample_weight=weights)
    context, tolerance = "a" * 64, 1e-4
    if changed == "features":
        x[0, 0] += .1
    elif changed == "order":
        x = x[::-1]
    elif changed == "labels":
        y[0] = 1 - y[0]
    elif changed == "weights":
        weights[0] += .1
    elif changed == "context":
        context = "b" * 64
    else:
        tolerance = 1e-5
    progress = []
    with binary_fit_checkpoint_scope(tmp_path, context):
        fit_bounded_binary_head(x, y, sample_weight=weights,
                                tolerance=tolerance, progress=progress.append)
    assert progress[0]["resumed_iterations"] == 0
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_modified_checkpoint_is_rejected_before_fitting(tmp_path):
    x, y, weights = fixture()
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        fit_bounded_binary_head(x, y, sample_weight=weights)
    path, = tmp_path.glob("*.npz")
    with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["weight"][0] += 1
    np.savez(path, **arrays)
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        with pytest.raises(ValueError, match="checksum"):
            fit_bounded_binary_head(x, y, sample_weight=weights)


def test_checkpoint_identity_materialization_is_also_bounded(tmp_path):
    x, y, weights = fixture()
    x, y, weights = np.tile(x, (4, 1)), np.tile(y, 4), np.tile(weights, 4)
    requested = []
    class BoundedView:
        shape, ndim = x.shape, x.ndim
        def __getitem__(self, index):
            assert isinstance(index, slice) and 0 < index.stop - index.start <= 256
            requested.append(index.stop - index.start)
            return x[index]
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        fit_bounded_binary_head(BoundedView(), y, sample_weight=weights)
    assert requested and max(requested) == 256


def test_an_interruption_after_periodic_save_retains_only_an_accepted_iterate(tmp_path):
    rng = np.random.default_rng(42)
    x = (rng.normal(size=(240, 32)) * np.geomspace(1., 20., 32)).astype(np.float32)
    y = (rng.normal(size=240) + x[:, 0] > .2).astype(np.int8)
    def interrupted(row):
        if row["stage"] == "binary_fit_iteration":
            raise RuntimeError("interrupted after checkpoint")
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        with pytest.raises(RuntimeError, match="after checkpoint"):
            fit_bounded_binary_head(x, y, tolerance=1e-7, progress=interrupted)
        assert binary_fit_checkpoint_receipt()["records"] == []
    path, = tmp_path.glob("*.npz")
    body, _arrays = read_fit_archive(path, ("weight", "center"))
    assert body["status"] == "iterating" and body["iterations"] == 25
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        actual, bias = fit_bounded_binary_head(x, y, tolerance=1e-7)
        assert binary_fit_checkpoint_receipt()["records"][0]["resumed_iterations"] == 25
    expected, expected_bias = fit_bounded_binary_head(x, y, tolerance=1e-7)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert bias == pytest.approx(expected_bias, abs=3e-5)


def test_scope_restoration_and_receipt_snapshot_do_not_mutate_prior_evidence(tmp_path):
    x, y, weights = fixture()
    with binary_fit_checkpoint_scope(tmp_path / "outer", "a" * 64):
        fit_bounded_binary_head(x, y, sample_weight=weights)
        first = binary_fit_checkpoint_receipt()
        with binary_fit_checkpoint_scope(tmp_path / "inner", "b" * 64):
            fit_bounded_binary_head(x, y, sample_weight=weights)
            assert binary_fit_checkpoint_receipt()["context_identity"] == "b" * 64
        fit_bounded_binary_head(x, y, sample_weight=weights)
        second = binary_fit_checkpoint_receipt()
    assert len(first["records"]) == 1 and len(second["records"]) == 2
    assert first["receipt_sha256"] == fit_identity({key: value for key, value in first.items()
                                                    if key != "receipt_sha256"})
    assert binary_fit_checkpoint_receipt() is None
    with pytest.raises(ValueError, match="immutable"):
        with binary_fit_checkpoint_scope(tmp_path, "unbound"):
            pass
