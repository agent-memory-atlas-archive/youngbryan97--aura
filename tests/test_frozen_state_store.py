"""Frozen state spilling preserves values and refuses corrupted artifacts."""

import hashlib

import mlx.core as mx
import pytest

from core.learning.frozen_state_store import FrozenStateStore


def store(tmp_path, bound=64):
    return FrozenStateStore(tmp_path, plan_sha256="a" * 64, max_resident_bytes=bound)


def write(cache, source, value):
    key = (source, 0, 1)
    return cache.write_source(source, {key: value}, sequence_digests={key: "b" * 64})


def test_lossless_roundtrip_and_eviction(tmp_path):
    with mx.stream(mx.cpu):
        cache = store(tmp_path, 32)
        a = mx.arange(8, dtype=mx.float32).reshape(1, 2, 4)
        b = mx.arange(8, dtype=mx.float32) + .12345
        write(cache, "a", a)
        write(cache, "b", b)
        assert len(cache) == 2
        assert cache.resident_bytes == 0
        assert mx.array_equal(cache[("a", 0, 1)], a).item()
        assert mx.array_equal(cache[("b", 0, 1)], b).item()
        assert mx.array_equal(cache[("a", 0, 1)], a).item()
        assert cache.loads == 3
        assert cache.peak_resident_bytes == 32
        assert cache.receipt()["lossy_compression"] is False


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32, mx.uint32])
def test_dtype_shape_and_duplicate_reads(tmp_path, dtype):
    with mx.stream(mx.cpu):
        cache = store(tmp_path)
        value = mx.arange(8).astype(dtype).reshape(1, 2, 4)
        write(cache, "a", value)
        actual = cache[("a", 0, 1)]
        assert actual.dtype == value.dtype
        assert actual.shape == value.shape
        assert mx.array_equal(actual, value).item()
        assert cache[("a", 0, 1)] is actual
        assert cache.loads == 1


def test_corruption_refused_before_tensor_load(tmp_path):
    with mx.stream(mx.cpu):
        cache = store(tmp_path)
        write(cache, "a", mx.arange(8))
        path = tmp_path / (hashlib.sha256(b"a").hexdigest() + ".safetensors")
        path.chmod(0o600)
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="digest differs"):
            cache[("a", 0, 1)]
        assert cache.loads == 0


def test_bound_and_identity_refusals(tmp_path):
    with mx.stream(mx.cpu):
        cache = store(tmp_path, 16)
        with pytest.raises(ValueError, match="exceeds"):
            write(cache, "a", mx.arange(8))
        assert len(cache) == 0
        write(cache, "a", mx.arange(4))
        with pytest.raises(ValueError, match="repeat"):
            write(cache, "a", mx.arange(4))
        with pytest.raises(KeyError):
            cache[("missing", 0, 1)]


def test_all_alternatives_retained_and_bound_together(tmp_path):
    with mx.stream(mx.cpu):
        cache = store(tmp_path, 64)
        states = {("a", 0, i): mx.arange(4) + i for i in range(4)}
        receipt = cache.write_source("a", states, sequence_digests={key: "c" * 64 for key in states})
        assert receipt["array_bytes"] == 64
        assert len(cache) == 4
        for key, expected in states.items():
            assert mx.array_equal(cache[key], expected).item()
        assert cache.loads == 1


def test_cpu_receipt_verifies_all_sequence_bytes_and_detects_omissions(tmp_path):
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.verify_semantic_native_fit import verify_state_storage

    with mx.stream(mx.cpu):
        cache = store(tmp_path / "prefix-states")
        key = ("a", 0, 1)
        cache.write_source("a", {key: mx.arange(4)}, sequence_digests={key: digest([1, 2])})
        plan = {"plan_sha256": "a" * 64, "prefix_storage_contract": {
            "schema": "aura.frozen_state_storage_contract.v1", "mode": "source_shards",
            "max_resident_bytes": 64, "lossy_compression": False, "all_alternatives_retained": True}}
        report = {"prefix_storage_receipt": cache.receipt()}
        supervision = {"rows": [{"source": "a", "decision_index": 0,
                                 "choice_index": 1, "tokens": [1, 2]}]}
        result = verify_state_storage(tmp_path, plan, report, supervision)
        assert result["sequences"] == 1
        assert result["hidden_states_independently_recomputed"] is False
        report["prefix_storage_receipt"]["shards"] = []
        with pytest.raises(ValueError, match="coverage differs"):
            verify_state_storage(tmp_path, plan, report, supervision)


def test_spilling_does_not_change_loss_or_gradients(tmp_path):
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    with mx.stream(mx.cpu):
        cache = store(tmp_path)
        key = ("a", 0, 1)
        hidden = mx.arange(8, dtype=mx.float32).reshape(2, 4) / 3
        write(cache, "a", hidden)
        model = nn.Linear(4, 2)
        def loss(tail, value):
            return mx.sum(tail(value) ** 2)
        direct_loss, direct_gradient = nn.value_and_grad(model, lambda tail: loss(tail, hidden))(model)
        disk_loss, disk_gradient = nn.value_and_grad(model, lambda tail: loss(tail, cache[key]))(model)
        assert mx.array_equal(direct_loss, disk_loss).item()
        for (left_name, left), (right_name, right) in zip(
                tree_flatten(direct_gradient), tree_flatten(disk_gradient), strict=True):
            assert left_name == right_name
            assert mx.array_equal(left, right).item()
