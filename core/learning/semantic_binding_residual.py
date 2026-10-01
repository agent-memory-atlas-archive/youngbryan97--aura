"""A source-fit nonlinear residual on the existing role/mention/filler features."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NonlinearBindingResidual:
    center: np.ndarray
    scale: np.ndarray
    kernel: np.ndarray
    hidden_bias: np.ndarray
    output: np.ndarray
    bias: float

    def __post_init__(self):
        for name in ("center", "scale", "kernel", "hidden_bias", "output"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        width = self.center.size
        if (self.center.shape != (width,) or width < 1 or self.scale.shape != (width,)
                or self.kernel.ndim != 2 or self.kernel.shape[0] != width or self.kernel.shape[1] < 1
                or self.hidden_bias.shape != (self.kernel.shape[1],)
                or self.output.shape != self.hidden_bias.shape or not np.isfinite(self.bias)
                or any(not np.isfinite(getattr(self, name)).all()
                       for name in ("center", "scale", "kernel", "hidden_bias", "output"))
                or np.any(self.scale <= 0)):
            raise ValueError("invalid nonlinear binding residual")

    def score(self, feature):
        feature = np.asarray(feature, dtype=np.float64)
        if feature.shape != self.center.shape or not np.isfinite(feature).all():
            raise ValueError("nonlinear binding feature differs")
        value = np.tanh(((feature - self.center) / self.scale) @ self.kernel + self.hidden_bias)
        return float(value @ self.output + self.bias)

    def scaled(self, factor):
        if not np.isfinite(factor) or factor < 0:
            raise ValueError("invalid nonlinear binding score scale")
        return NonlinearBindingResidual(self.center, self.scale, self.kernel, self.hidden_bias,
                                        self.output * factor, self.bias * factor)

    def feature_lesion(self, indices):
        kernel = self.kernel.copy()
        kernel[list(indices)] = 0.
        return NonlinearBindingResidual(self.center, self.scale, kernel, self.hidden_bias,
                                        self.output, self.bias)

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in
                ("center", "scale", "kernel", "hidden_bias", "output")} | {"bias": float(self.bias)}

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"center", "scale", "kernel", "hidden_bias", "output", "bias"}:
            raise ValueError("nonlinear binding residual fields differ")
        return cls(**value)


def fit_binding_residual(features, labels, weights, baseline_scores, *, width, steps=200,
                         seed=0, learning_rate=.01, ridge=1e-4):
    """Fit only supplied source contrasts, never target-family routing features.

    Candidate rows use the same scorer. Normalization and initialization are
    local to this fit; global RNG state and input feature arrays are untouched.
    """
    features, labels, weights, baseline = (np.asarray(value, dtype=np.float64)
        for value in (features, labels, weights, baseline_scores))
    if (features.ndim != 2 or min(features.shape) < 1 or type(width) is not int or width < 1
            or type(steps) is not int or steps < 1 or type(seed) is not int
            or labels.shape != (len(features),) or weights.shape != labels.shape
            or baseline.shape != labels.shape or not set(labels) <= {0., 1.}
            or len(set(labels)) != 2 or np.any(weights <= 0)
            or any(not np.isfinite(value).all() for value in (features, labels, weights, baseline))
            or not np.isfinite(learning_rate) or learning_rate <= 0
            or not np.isfinite(ridge) or ridge < 0):
        raise ValueError("invalid source nonlinear binding fit")
    weights = weights / weights.sum()
    center = weights @ features
    scale = np.maximum(np.sqrt(weights @ ((features - center) ** 2)), 1e-6)
    normalized = (features - center) / scale
    rng = np.random.default_rng(seed)
    parameters = [rng.normal(size=(features.shape[1], width)) / np.sqrt(features.shape[1]),
                  np.zeros(width), np.zeros(width), np.zeros(1)]
    moments = [np.zeros_like(value) for value in parameters]
    variances = [np.zeros_like(value) for value in parameters]
    for step in range(1, steps + 1):
        kernel, hidden_bias, output, bias = parameters
        hidden = np.tanh(normalized @ kernel + hidden_bias)
        scores = baseline + hidden @ output + bias[0]
        # Stable logistic derivatives without clipping the residual gradient.
        probability = np.exp(-np.logaddexp(0., -scores))
        derivative = weights * (probability - labels)
        hidden_gradient = derivative[:, None] * output * (1. - hidden ** 2)
        gradients = [normalized.T @ hidden_gradient + ridge * kernel,
                     hidden_gradient.sum(axis=0), hidden.T @ derivative + ridge * output,
                     np.asarray([derivative.sum()])]
        for parameter, gradient, moment, variance in zip(parameters, gradients, moments, variances, strict=True):
            moment *= .9
            moment += .1 * gradient
            variance *= .999
            variance += .001 * gradient ** 2
            parameter -= learning_rate * (moment / (1. - .9 ** step)) / (
                np.sqrt(variance / (1. - .999 ** step)) + 1e-8)
    return NonlinearBindingResidual(center, scale, *parameters[:3], float(parameters[3][0]))
