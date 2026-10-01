"""MLX residual adapters with a common, exact baseline lesion interface."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear


class SemanticAdapterLinear(LoRALinear):
    """Nonlinear/routed updates cannot be fused into a fixed weight matrix.

    All fitted state is named lora_* so the native frozen-parameter custody
    checks and checkpoint writer apply to these adapters without exemptions.
    DoRA uses scale as an outer residual gate, including learned magnitude.
    """

    @classmethod
    def from_base(cls, linear, *, r, scale, dropout=0., kind, experts=1):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits
        result = cls(input_dims, output_dims, r, dropout, scale)
        result.linear = linear
        result.kind = kind
        result.experts = experts
        result._direction_scale = float(scale)
        result._record_router_balance = False
        result._router_balance_terms = []
        if kind in {"square", "dense"}:
            del result.lora_a
            del result.lora_b
            result.lora_budget_rank = r
            result.input_dims, result.output_dims = input_dims, output_dims
            if kind == "dense":
                result.lora_full = mx.zeros((input_dims, output_dims))
            else:
                # A compression-square-expansion update, derived from the
                # high-rank budget idea; not a replica of a MoRA release.
                width = min(input_dims, output_dims, math.isqrt(r * (input_dims + output_dims)))
                result.square_width = max(1, width)
                result.lora_square = mx.zeros((result.square_width, result.square_width))
            return result
        if kind == "product":
            bound = 1. / math.sqrt(input_dims)
            result.lora_right = mx.random.uniform(-bound, bound, (input_dims, r))
        elif kind == "routed":
            bound = 1. / math.sqrt(input_dims)
            result.lora_a = mx.random.uniform(-bound, bound, (experts, input_dims, r))
            result.lora_b = mx.zeros((experts, r, output_dims))
            result.lora_gate = mx.zeros((input_dims, experts))
        elif kind == "dora":
            result.lora_m = mx.linalg.norm(result._weight().astype(mx.float32), axis=1)
        elif kind != "silu":
            raise ValueError("unsupported nonlinear native adapter")
        return result

    def _weight(self):
        linear = self.linear
        if isinstance(linear, nn.QuantizedLinear):
            return mx.dequantize(linear.weight, linear.scales, linear.biases,
                                 group_size=linear.group_size, bits=linear.bits, mode=linear.mode)
        return linear.weight

    def __call__(self, x):
        baseline = self.linear(x)
        if self.scale == 0.:
            return baseline
        dropped = self.dropout(x)
        if self.kind == "dense":
            delta = dropped @ self.lora_full
        elif self.kind == "square":
            width = self.square_width
            groups = math.ceil(self.input_dims / width)
            padded = mx.pad(dropped, [(0, 0)] * (dropped.ndim - 1) + [(0, groups * width - self.input_dims)])
            compressed = mx.sum(padded.reshape(*dropped.shape[:-1], groups, width), axis=-2) / math.sqrt(groups)
            transformed = compressed @ self.lora_square
            delta = mx.take(transformed, mx.arange(self.output_dims) % width, axis=-1)
            delta = delta / math.sqrt(math.ceil(self.output_dims / width))
        elif self.kind == "routed":
            gates = mx.softmax(x @ self.lora_gate, axis=-1)
            if self._record_router_balance:
                average = mx.mean(gates.reshape(-1, self.experts), axis=0)
                self._router_balance_terms.append(self.experts * mx.sum(average ** 2) - 1.)
            delta = sum(gates[..., index:index + 1] *
                        (nn.silu(dropped @ self.lora_a[index]) @ self.lora_b[index])
                        for index in range(self.experts))
        elif self.kind == "product":
            delta = ((dropped @ self.lora_a) * (dropped @ self.lora_right)) @ self.lora_b
        elif self.kind == "silu":
            delta = nn.silu(dropped @ self.lora_a) @ self.lora_b
        else:
            weight = self._weight()
            adapted = weight + self._direction_scale * (self.lora_b.T @ self.lora_a.T)
            denominator = mx.stop_gradient(mx.linalg.norm(adapted.astype(mx.float32), axis=1))
            ratio = self.lora_m / mx.maximum(denominator, 1e-30)
            value = ratio.astype(x.dtype) * (x @ weight.T +
                (self._direction_scale * ((dropped @ self.lora_a) @ self.lora_b)).astype(x.dtype))
            if "bias" in self.linear:
                value = value + self.linear.bias
            return baseline + (self.scale / self._direction_scale) * (value - baseline)
        return baseline + (self.scale * delta).astype(x.dtype)

    def fuse(self, dequantize=False):
        raise ValueError("semantic nonlinear/residual-gated adapters require unmerged serving")

    def router_balance_loss(self, x):
        if self.kind != "routed":
            raise ValueError("router balance applies only to routed adapters")
        gates = mx.softmax(x @ self.lora_gate, axis=-1)
        average = mx.mean(gates.reshape(-1, self.experts), axis=0)
        return self.experts * mx.sum(average ** 2) - 1.
