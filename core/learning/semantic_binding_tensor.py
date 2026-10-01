"""Lossless role-addressed tensor storage, not a natural-language interpreter."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class BindingTensor:
    roles: tuple[str, ...]
    role_vectors: np.ndarray
    tensor: np.ndarray
    source_keys: object = None

    def __post_init__(self):
        roles = tuple(self.roles)
        vectors = np.array(self.role_vectors, dtype=np.float64, copy=True)
        tensor = np.array(self.tensor, dtype=np.float64, copy=True)
        if (not roles or len(set(roles)) != len(roles) or any(not isinstance(role, str) or not role for role in roles)
                or vectors.ndim != 2 or vectors.shape[0] != len(roles)
                or tensor.ndim != 2 or tensor.shape[0] != vectors.shape[1] or tensor.shape[1] < 1
                or not np.isfinite(vectors).all() or not np.isfinite(tensor).all()
                or np.linalg.matrix_rank(vectors) != len(roles)
                or np.linalg.cond(vectors) > 1e8):
            raise ValueError("binding tensor needs independent well-conditioned roles")
        vectors.setflags(write=False)
        tensor.setflags(write=False)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "role_vectors", vectors)
        object.__setattr__(self, "tensor", tensor)
        keys = {} if self.source_keys is None else dict(self.source_keys)
        if keys and (set(keys) != set(roles) or any(len(key) != 2 or
                any(not isinstance(part, str) or not part for part in key) for key in keys.values())):
            raise ValueError("binding tensor source identities differ from its role addresses")
        object.__setattr__(self, "source_keys", MappingProxyType({role: tuple(key) for role, key in keys.items()}))

    @classmethod
    def pack(cls, fillers, *, role_vectors=None, source_keys=None):
        """Default exact basis has no random cross-role interference.

        Non-orthogonal independent supplied roles use their dual basis for
        unbinding. Fillers may coincide; this does not merge role identities.
        """
        roles = tuple(sorted(fillers))
        if not roles:
            raise ValueError("binding tensor needs at least one grounded role")
        values = np.asarray([fillers[role] for role in roles], dtype=np.float64)
        vectors = np.eye(len(roles)) if role_vectors is None else np.asarray(role_vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 1 or vectors.ndim != 2 or vectors.shape[0] != len(roles):
            raise ValueError("binding tensor role/filler geometry differs")
        return cls(roles, vectors, vectors.T @ values, source_keys)

    def unbind(self, role):
        if role not in self.roles:
            raise KeyError(role)
        dual = np.linalg.pinv(self.role_vectors.T)
        return dual[self.roles.index(role)] @ self.tensor

    def to_dict(self):
        return {"schema": "aura.binding_tensor.v1", "roles": list(self.roles),
                "role_vectors": self.role_vectors.tolist(), "tensor": self.tensor.tolist(),
                "source_keys": {role: list(key) for role, key in self.source_keys.items()}}

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != "aura.binding_tensor.v1":
            raise ValueError("unknown binding tensor storage contract")
        tensor = cls(tuple(value["roles"]), value["role_vectors"], value["tensor"], value["source_keys"])
        if tensor.to_dict() != value:
            raise ValueError("binding tensor storage contract differs")
        return tensor


def tensor_from_grounded_binding(binding, context):
    """Require accepted, actually observed vectors; never synthesize fillers."""
    if not binding.executable:
        raise ValueError("unresolved binding has no tensor storage authority")
    records = {record.key: record for record in context.referents}
    fillers = {}
    for role, key in binding.bindings:
        if key not in records or not records[key].embedding:
            raise ValueError("grounded source lacks an observed vector")
        fillers[role] = records[key].embedding
    return BindingTensor.pack(fillers, source_keys=dict(binding.bindings))
