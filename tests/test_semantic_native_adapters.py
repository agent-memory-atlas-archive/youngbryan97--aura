"""Capacity changes replay the same adapter arithmetic they train."""

import math
from copy import deepcopy

import pytest

from tools.semantic_native_adapters import (
    adapter_config_from_plan,
    adapter_contract,
    install_native_adapters,
    native_adapter_scale,
    native_optimizer,
)


def plan(**options):
    contract = adapter_contract(rank=options.pop("rank", 8), layers=options.pop("layers", 1), **options)
    return {"rank": contract["rank"], "suffix_layers": contract["suffix_layers"],
            "adapter_keys": contract["keys"], "adapter_contract": contract}


def test_old_plan_replays_the_original_direct_multiplier():
    old = {"rank": 8, "suffix_layers": 1,
           "adapter_keys": ["self_attn.q_proj", "mlp.down_proj"]}
    assert adapter_config_from_plan(old) == {
        "rank": 8, "scale": 16., "dropout": 0., "keys": old["adapter_keys"]}


@pytest.mark.parametrize("rank", [8, 16, 32, 64, 128])
def test_rank_stabilized_scaling_is_applied_once(rank):
    value = plan(rank=rank, scaling="alpha_over_sqrt_rank_v1", alpha=32.)
    assert native_adapter_scale(value) == pytest.approx(32. / math.sqrt(rank))
    assert adapter_config_from_plan(value)["rank"] == rank


def test_lora_alpha_over_rank_is_distinct_from_legacy_direct_scale():
    assert native_adapter_scale(plan(rank=32, scaling="alpha_over_rank_v1", alpha=64.)) == 2.
    assert native_adapter_scale(plan(rank=32, alpha=64.)) == 64.


@pytest.mark.parametrize("field,value", [("rank", 32), ("suffix_layers", 4),
                                       ("adapter_keys", ["self_attn.q_proj"])])
def test_replay_refuses_geometry_drift(field, value):
    frozen = plan()
    frozen[field] = value
    with pytest.raises(ValueError, match="geometry"):
        adapter_config_from_plan(frozen)


def test_replay_refuses_a_multiplier_not_derived_from_the_declared_policy():
    frozen = deepcopy(plan())
    frozen["adapter_contract"]["effective_scale"] = 100.
    with pytest.raises(ValueError, match="geometry"):
        adapter_config_from_plan(frozen)


@pytest.mark.parametrize("options", [{"rank": 0}, {"layers": 0}, {"rank": True},
                                    {"alpha": float("nan")}, {"alpha": 0.}])
def test_invalid_capacity_is_rejected_before_loading(options):
    with pytest.raises(ValueError, match="capacity"):
        plan(**options)


def tiny_model():
    nn = pytest.importorskip("mlx.nn")

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj = nn.Linear(4, 4), nn.Linear(4, 4)
            self.v_proj, self.o_proj = nn.Linear(4, 4), nn.Linear(4, 4)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = Attention(), MLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block() for _ in range(3)]

    model = Model()
    model.freeze()
    return model


def test_real_mlx_installs_all_declared_sites_at_the_declared_depth():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear

    model = tiny_model()
    value = plan(rank=2, layers=2, sites="attention_mlp_v1", scaling="alpha_over_sqrt_rank_v1", alpha=4.)
    x = mx.ones((1, 4))
    before = model.layers[-1].self_attn.q_proj(x)
    mx.eval(before)
    install_native_adapters(model, value)
    assert not isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
    for layer in model.layers[-2:]:
        modules = dict(layer.named_modules())
        for key in value["adapter_keys"]:
            assert isinstance(modules[key], LoRALinear)
            assert modules[key].scale == pytest.approx(4. / math.sqrt(2))
    assert bool(mx.allclose(before, model.layers[-1].self_attn.q_proj(x)).item())
    trainable = tree_flatten(model.trainable_parameters())
    assert len(trainable) == 2 * 7 * 2
    assert all("lora_" in name for name, _value in trainable)


def test_requested_depth_and_missing_sites_are_not_silently_ignored():
    model = tiny_model()
    with pytest.raises(ValueError, match="depth"):
        install_native_adapters(model, plan(layers=4))
    del model.layers[-1].self_attn.k_proj
    with pytest.raises(ValueError, match="absent"):
        install_native_adapters(model, plan(sites="attention_mlp_v1"))


@pytest.mark.parametrize("kind", ["silu", "product", "routed", "dora", "square", "dense"])
def test_new_function_classes_preserve_baseline_train_and_lesion_exactly(kind):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    model = tiny_model()
    mx.random.seed(47)
    x = mx.random.normal((8, 4))
    original = model.layers[-1].self_attn.q_proj(x)
    mx.eval(original)
    value = plan(rank=2, kind=kind, experts=3 if kind == "routed" else 1)
    install_native_adapters(model, value)
    module = model.layers[-1].self_attn.q_proj
    assert bool(mx.allclose(original, module(x), atol=1e-6).item())
    optimizer = native_optimizer({"learning_rate": .01, "weight_decay": 0.,
                                  "adapter_b_learning_rate_ratio": 1. if kind in {"square", "dense"} else 3.})
    for _ in range(3):
        loss, gradient = nn.value_and_grad(module, lambda layer: mx.mean((layer(x) - 2.) ** 2))(module)
        optimizer.update(module, gradient)
        mx.eval(module.parameters(), loss)
    assert all("lora_" in name for name, _ in tree_flatten(module.trainable_parameters()))
    assert not bool(mx.allclose(original, module(x)).item())
    module.scale = 0.
    assert bool(mx.array_equal(original, module(x)).item())
    with pytest.raises(ValueError, match="unmerged"):
        module.fuse()


def test_rank_schedule_replays_per_site_scaling_without_using_the_current_residual_gate():
    model = tiny_model()
    value = plan(rank=8, layers=3, layer_ranks=[1, 2, 4],
                 scaling="alpha_over_sqrt_rank_v1", alpha=4., kind="product")
    install_native_adapters(model, value)
    for layer, rank in zip(model.layers, [1, 2, 4], strict=True):
        module = layer.self_attn.q_proj
        assert module.lora_a.shape[-1] == rank
        assert module.scale == pytest.approx(4. / math.sqrt(rank))
        module.scale = 0.
        assert native_adapter_scale(value, module) == pytest.approx(4. / math.sqrt(rank))


def test_lora_plus_applies_the_rate_ratio_only_to_the_output_factor():
    import mlx.core as mx
    import mlx.nn as nn

    model = nn.Module()
    model.lora_a, model.lora_b, model.lora_right = mx.ones((2,)), mx.ones((2,)), mx.ones((2,))
    optimizer = native_optimizer({"learning_rate": .01, "weight_decay": 0.,
                                  "adapter_b_learning_rate_ratio": 4.})
    # Module paths can be root leaves as well as dotted decoder paths.
    optimizer.update(model, {name: mx.ones((2,)) for name in ("lora_a", "lora_b", "lora_right")})
    delta_a = 1. - float(model.lora_a[0].item())
    delta_b = 1. - float(model.lora_b[0].item())
    assert delta_b == pytest.approx(4. * delta_a, abs=1e-6)
    assert float(model.lora_right[0].item()) == pytest.approx(float(model.lora_a[0].item()))


@pytest.mark.parametrize("options", [{"layer_ranks": [1, 2]}, {"kind": "missing"},
                                    {"experts": 2}, {"kind": "routed", "experts": 0}])
def test_invalid_extended_capacity_is_not_silently_accepted(options):
    with pytest.raises(ValueError):
        plan(**options)


def test_hybrid_topology_uses_native_mixing_sites_not_nonexistent_dense_projections():
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    from tools.semantic_native_adapters import native_adapter_keys_by_layer

    model = tiny_model()
    del model.layers[-2].self_attn
    linear = nn.Module()
    for key in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        setattr(linear, key, nn.Linear(4, 4))
    model.layers[-2].linear_attn = linear
    model.freeze()
    value = plan(rank=2, layers=2, sites="native_topology_v1")
    resolved = native_adapter_keys_by_layer(model, value)
    assert resolved[0][0] == "linear_attn.in_proj_qkv" and resolved[1][0] == "self_attn.q_proj"
    install_native_adapters(model, value)
    for layer, keys in zip(model.layers[-2:], resolved, strict=True):
        assert all(isinstance(dict(layer.named_modules())[key], LoRALinear) for key in keys)


def test_router_balancing_uses_real_calls_has_gradients_and_releases_graphs():
    import mlx.core as mx
    import mlx.nn as nn

    from tools.semantic_native_adapters import native_router_regularized_objective

    model = tiny_model()
    install_native_adapters(model, plan(rank=2, kind="routed", experts=3))
    module = model.layers[-1].self_attn.q_proj
    module.lora_gate = mx.array([[5., 0., 0.]] * 4)
    x = mx.ones((3, 4))
    loss, gradient = nn.value_and_grad(module, lambda value: native_router_regularized_objective(value,
        lambda: mx.sum(value(x)) * 0., weight=.3))(module)
    assert loss.item() > 0. and mx.sum(mx.abs(gradient["lora_gate"])).item() > 0.
    assert module._router_balance_terms == [] and not module._record_router_balance
    with pytest.raises(RuntimeError):
        native_router_regularized_objective(module, lambda: (_ for _ in ()).throw(RuntimeError()), weight=.3)
    assert module._router_balance_terms == [] and not module._record_router_balance


@pytest.mark.parametrize("kind", ["lora", "silu", "product", "routed", "dora", "square", "dense"])
def test_parameter_projection_matches_actual_installed_qwen_sites(kind):
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from tools.semantic_native_adapters import native_adapter_parameter_estimate

    config = {"model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
              "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
              "vocab_size": 32, "rms_norm_eps": 1e-6}
    model = Model(ModelArgs.from_dict(config))
    model.freeze()
    value = plan(rank=2, layers=2, sites="native_topology_v1", kind=kind, experts=2 if kind == "routed" else 1)
    estimate = native_adapter_parameter_estimate(config, value["adapter_contract"])
    install_native_adapters(model, value)
    count = sum(value.size for _key, value in tree_flatten(model.trainable_parameters()))
    assert count == estimate["trainable_parameters"]
    assert estimate["activation_bytes"] is None and estimate["fit_is_guaranteed"] is False


def test_unknown_memory_geometry_is_unmeasured_not_zero():
    from tools.semantic_native_adapters import native_adapter_parameter_estimate

    assert native_adapter_parameter_estimate({"model_type": "unknown"}, plan()["adapter_contract"]) is None


def test_mixed_layer_function_classes_are_installed_and_replayed_as_one_suffix():
    from mlx_lm.tuner.lora import LoRALinear

    from tools.semantic_native_adapter_layers import SemanticAdapterLinear

    model = tiny_model()
    value = plan(rank=2, layers=3, layer_kinds=["product", "routed", "dora"], experts=3)
    install_native_adapters(model, value)
    for layer, kind in zip(model.layers, ["product", "routed", "dora"], strict=True):
        module = layer.self_attn.q_proj
        assert isinstance(module, SemanticAdapterLinear) and isinstance(module, LoRALinear)
        assert module.kind == kind
        assert module.experts == (3 if kind == "routed" else 1)
    assert adapter_config_from_plan(value)["rank"] == 2


@pytest.mark.parametrize("kind", ["silu", "product", "routed", "dora", "square", "dense"])
def test_quantized_base_can_train_restore_and_lesion_custom_adapters(kind, tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from tools.semantic_native_adapter_layers import SemanticAdapterLinear

    owner = nn.Module()
    owner.projection = nn.Linear(32, 32)
    nn.quantize(owner, group_size=32, bits=4)
    base = owner.projection
    assert isinstance(base, nn.QuantizedLinear)
    base.freeze()
    layer = SemanticAdapterLinear.from_base(base, r=2, scale=2., kind=kind, experts=2 if kind == "routed" else 1)
    x = mx.ones((2, 32))
    baseline = base(x)
    assert mx.allclose(layer(x), baseline, atol=1e-5).item()
    optimizer = optim.Adam(learning_rate=.01)
    for _ in range(2):
        loss, grads = nn.value_and_grad(layer, lambda value: mx.mean((value(x) - 1.) ** 2))(layer)
        optimizer.update(layer, grads)
        mx.eval(layer.parameters(), loss)
    path = tmp_path / "weights.safetensors"
    layer.save_weights(str(path))
    restored = SemanticAdapterLinear.from_base(base, r=2, scale=2., kind=kind, experts=2 if kind == "routed" else 1)
    restored.load_weights(str(path), strict=True)
    assert mx.array_equal(layer(x), restored(x)).item()
    restored.scale = 0.
    assert mx.array_equal(restored(x), baseline).item()
