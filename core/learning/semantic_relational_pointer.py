"""Learned, multi-depth role/filler interaction with equivariant graph refinement.

No family ID, candidate index, absolute position or denotation enters scoring.
Upstream parsers still own source discovery and token-to-object alignment.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from core.learning.procedure_induction import PRIMITIVES_BY_NAME
from core.learning.semantic_program_floor import semantic_primitive_type_signature

SEMANTIC_ROLE_NAMES = ("agent", "patient", "theme", "instrument", "recipient", "source",
                      "destination", "cause", "goal", "location", "value", "minuend",
                      "subtrahend", "dividend", "divisor", "giver", "receiver", "variable", "body")
SEMANTIC_OPERATIONS = tuple(sorted(name for name in PRIMITIVES_BY_NAME
                                 if semantic_primitive_type_signature(name) is not None))
SEMANTIC_ROLE_WIDTH = len(SEMANTIC_ROLE_NAMES) + 6 + 3 * len(SEMANTIC_OPERATIONS)


def semantic_role_features(roles):
    """Encode logical roles, never source occurrence indices or constructions.

    Operand addresses belong to the public typed operation, not word order.
    A public operation/slot has its own learned direction in addition to
    shared slot and type directions. Unknown roles require an explicit
    caller-supplied representation; they are never silently collapsed.
    """
    rows = []
    for role in roles:
        row = [0.] * SEMANTIC_ROLE_WIDTH
        name = role.role.lower()
        if name in SEMANTIC_ROLE_NAMES:
            row[SEMANTIC_ROLE_NAMES.index(name)] = 1.
        elif ":operand:" in name:
            operation, slot = name.split(":operand:", 1)
            signature = semantic_primitive_type_signature(operation)
            if signature is None or not slot.isdecimal() or not 0 <= int(slot) < len(signature[0]):
                raise ValueError("semantic role query has no declared operation operand")
            if len(signature[0]) > 3:
                raise ValueError("semantic role query exceeds its declared operand basis")
            if role.type_name != signature[0][int(slot)]:
                raise ValueError("semantic role query type differs from the public operation")
            row[len(SEMANTIC_ROLE_NAMES) + int(slot)] = 1.
            row[len(SEMANTIC_ROLE_NAMES) + 6 + 3 * SEMANTIC_OPERATIONS.index(operation) + int(slot)] = 1.
        else:
            raise ValueError("semantic role query needs a declared role or explicit representation")
        if role.type_name in {"integer", "Number"}:
            row[len(SEMANTIC_ROLE_NAMES) + 3] = 1.
        elif role.type_name == "integer_sequence":
            row[len(SEMANTIC_ROLE_NAMES) + 4] = 1.
        row[len(SEMANTIC_ROLE_NAMES) + 5] = float(role.unbound_cost is not None)
        rows.append(row)
    return mx.array(rows, dtype=mx.float32)


class RelationalBindingPointer(nn.Module):
    def __init__(self, hidden_width, *, depths=1, relation_width=32, rounds=2, role_queries=True):
        super().__init__()
        if (any(type(value) is not int or value < 1 for value in
                (hidden_width, depths, relation_width)) or type(rounds) is not int or not 0 <= rounds <= 8
                or type(role_queries) is not bool):
            raise ValueError("invalid relational binding geometry")
        self.hidden_width, self.depths = hidden_width, depths
        self.relation_width, self.rounds = relation_width, rounds
        self.role_queries = role_queries
        bound = 1. / math.sqrt(hidden_width)
        self.lora_depth_query = mx.zeros((hidden_width, depths))
        self.lora_depth_filler = mx.zeros((hidden_width, depths))
        self.lora_operation = mx.random.uniform(-bound, bound, (hidden_width, relation_width))
        self.lora_mention = mx.random.uniform(-bound, bound, (hidden_width, relation_width))
        self.lora_candidate = mx.random.uniform(-bound, bound, (hidden_width, relation_width))
        if role_queries:
            self.lora_role_query = mx.random.uniform(-bound, bound, (SEMANTIC_ROLE_WIDTH, relation_width))
        self.lora_message = mx.random.uniform(-bound, bound, (relation_width, relation_width))
        self.lora_update = mx.random.uniform(-bound, bound, (2 * relation_width, relation_width))
        self.feature_blocks = 16 if role_queries else 8
        self.lora_b = mx.zeros((self.feature_blocks * relation_width, 1))

    def _pool(self, values, gate):
        if values.ndim != 3 or values.shape[1:] != (self.depths, self.hidden_width):
            raise ValueError("relational pointer needs occurrence/depth/hidden tensors")
        query = mx.mean(values, axis=1)
        weights = mx.softmax(query @ gate, axis=-1)
        return mx.sum(values * weights[..., None], axis=1)

    def relation_features(self, operations, mentions, candidates, *, adjacency=None, role_features=None):
        operation = self._pool(operations, self.lora_depth_query) @ self.lora_operation
        mention = self._pool(mentions, self.lora_depth_query) @ self.lora_mention
        candidate = self._pool(candidates, self.lora_depth_filler) @ self.lora_candidate
        if operation.shape != mention.shape or operation.shape[0] < 1 or candidate.shape[0] < 1:
            raise ValueError("relational pointer role occurrences differ")
        role_query = None
        if self.role_queries:
            if (role_features is None or role_features.shape != (operation.shape[0], SEMANTIC_ROLE_WIDTH)
                    or not mx.all(mx.isfinite(role_features)).item()):
                raise ValueError("relational pointer requires explicit semantic role queries")
            role_query = role_features @ self.lora_role_query
        role_count, candidate_count = operation.shape[0], candidate.shape[0]
        nodes = mx.concatenate([operation + mention + (0. if role_query is None else role_query), candidate], axis=0)
        count = len(nodes)
        if adjacency is not None:
            if adjacency.shape != (count, count):
                raise ValueError("relational pointer adjacency differs from object identities")
            adjacency = adjacency.astype(mx.float32)
            if not mx.all(mx.isfinite(adjacency) & (adjacency >= 0)).item():
                raise ValueError("relational pointer needs finite nonnegative graph evidence")
            degree = mx.sum(adjacency, axis=-1, keepdims=True)
            weights = adjacency / mx.maximum(degree, 1.)
            for _ in range(self.rounds):
                message = weights @ (nodes @ self.lora_message)
                update = mx.tanh(mx.concatenate([nodes, message], axis=-1) @ self.lora_update)
                # No neighbors means no invented graph message.
                nodes = nodes + mx.where(degree > 0, update, 0.)
        role_context = nodes[:role_count]
        candidate = nodes[role_count:]
        o = mx.broadcast_to(operation[:, None, :], (role_count, candidate_count, self.relation_width))
        m = mx.broadcast_to(mention[:, None, :], o.shape)
        c = mx.broadcast_to(candidate[None, :, :], o.shape)
        contextual = mx.broadcast_to(role_context[:, None, :], o.shape)
        if role_query is None:
            feature = mx.concatenate([o, m, c, o * m, o * c, m * c, o * m * c, contextual - c], axis=-1)
        else:
            r = mx.broadcast_to(role_query[:, None, :], o.shape)
            feature = mx.concatenate([o, r, m, c, o * r, o * m, o * c, r * m, r * c, m * c,
                o * r * m, o * r * c, o * m * c, r * m * c, o * r * m * c, contextual - c], axis=-1)
        return nn.silu(feature)

    def __call__(self, operations, mentions, candidates, *, adjacency=None, allowed=None, role_features=None):
        feature = self.relation_features(operations, mentions, candidates, adjacency=adjacency, role_features=role_features)
        scores = (feature @ self.lora_b).squeeze(-1)
        if allowed is not None:
            if allowed.shape != scores.shape or allowed.dtype != mx.bool_:
                raise ValueError("relational pointer admission mask differs")
            scores = mx.where(allowed, scores, -mx.inf)
        return scores

    def to_contract(self):
        return {"schema": "aura.relational_binding_pointer.v3" if self.role_queries else "aura.relational_binding_pointer.v1",
                "hidden_width": self.hidden_width,
                "depths": self.depths, "relation_width": self.relation_width, "rounds": self.rounds,
                "candidate_order_feature": False, "construction_family_feature": False,
                **({"semantic_role_query": "public_operation_slot_type_v2", "role_feature_width": SEMANTIC_ROLE_WIDTH,
                    "role_names": list(SEMANTIC_ROLE_NAMES), "operations": list(SEMANTIC_OPERATIONS),
                    "feature_blocks": self.feature_blocks}
                   if self.role_queries else {})}


def pointer_role_margin_loss(scores, positive_sets, *, margin):
    """Separate source-witnessed positives from every admitted wrong identity.

    Equivalent positives are never pushed apart. A score margin provides
    training supervision, not a probability or proof of semantic truth.
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("binding role margin must be finite and nonnegative")
    pointer_choice_loss(scores, positive_sets)
    penalties = []
    for row, positive in zip(scores, positive_sets, strict=True):
        negative = tuple(index for index in range(len(row)) if index not in positive and mx.isfinite(row[index]).item())
        if negative:
            best_positive = mx.max(row[mx.array(positive)])
            penalties.append(mx.mean(mx.maximum(0., margin + row[mx.array(negative)] - best_positive)))
    return mx.mean(mx.stack(penalties)) if penalties else mx.sum(mx.where(mx.isfinite(scores), scores, 0.)) * 0.


def pointer_choice_loss(scores, positive_sets):
    """Supervise identities, including explicitly equivalent acceptable sources."""
    from core.learning.semantic_native_program import native_choice_loss

    if scores.ndim != 2 or min(scores.shape) < 1 or len(positive_sets) != scores.shape[0]:
        raise ValueError("pointer supervision roles differ")
    losses = []
    for index, positives in enumerate(positive_sets):
        positives = tuple(positives)
        if (not positives or len(set(positives)) != len(positives)
                or any(type(column) is not int or not 0 <= column < scores.shape[1]
                       for column in positives)):
            raise ValueError("pointer needs distinct source-positive candidate indices")
        row = scores[index]
        if (mx.any(mx.isnan(row) | (row == mx.inf)).item()
                or not mx.all(mx.isfinite(row[mx.array(positives)])).item()):
            raise ValueError("pointer positives must have admitted finite evidence")
        losses.append(native_choice_loss(row, positives) if scores.shape[1] > 1
                      else row[0] * 0.)
    return mx.mean(mx.stack(losses))


def pointer_context_costs(pointer, context, roles, operations, mentions, candidate_states, *, adjacency=None):
    """Use model evidence and source identities in the existing contextual solve.

    Missing rows are not generated or filled with random vectors. This API
    requires evidence for every candidate admitted by the supplied context.
    """
    records = tuple(context.referents)
    if not records or not roles:
        raise ValueError("pointer context needs roles and observed candidates")
    if set(candidate_states) != {record.key for record in records}:
        raise ValueError("pointer context candidate evidence is incomplete")
    if set(operations) != {role.identity for role in roles} or set(mentions) != set(operations):
        raise ValueError("pointer context role evidence is incomplete")
    allowed = mx.array([[context.eligible(role, record) for record in records] for role in roles])
    scores = pointer(mx.stack([operations[role.identity] for role in roles]),
                     mx.stack([mentions[role.identity] for role in roles]),
                     mx.stack([candidate_states[record.key] for record in records]),
                     adjacency=adjacency, allowed=allowed, role_features=semantic_role_features(roles))
    mx.eval(scores)
    costs = {}
    for index, role in enumerate(roles):
        for column, record in enumerate(records):
            if context.eligible(role, record):
                score = float(scores[index, column].item())
                if not math.isfinite(score):
                    raise ValueError("pointer context supplied nonfinite evidence")
                costs[role.identity, record.key] = -score
    return costs
