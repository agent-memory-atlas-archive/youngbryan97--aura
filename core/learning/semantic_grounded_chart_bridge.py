"""Observed relation evidence enters ordinary compositional chart decoding.

The decoder supplies its actual public operation/mention/definition pools.
No target graph, construction label or expected execution value is read.
"""

import math
import time
from dataclasses import replace

import mlx.core as mx

from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.learning.semantic_context_binding import (
    BindingContext,
    BindingRole,
    ContextReferent,
    candidate_cost,
)
from core.learning.semantic_grounded_binding_engine import (
    GroundedBindingEvidence,
    project_grounded_evidence,
)
from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_relational_pointer import semantic_role_features


class GroundedNativeReferenceScorer:
    """Carry actual grammar operation/slot queries to a measured binding bank.

    Non-reference choices keep the caller's native scorer. The observation
    reader receives public operation/slot metadata and admitted registers,
    never target graphs, answers, labels or alternative output text.
    """
    def __init__(self, engine, native_scorer, observation_reader):
        self.engine, self.native_scorer = engine, native_scorer
        self.observation_reader = observation_reader
        self.receipts = []

    def __call__(self, decisions):
        baseline = self.native_scorer(decisions)
        if (not isinstance(baseline, tuple) or len(baseline) != len(decisions)
                or not decisions or any(not math.isfinite(value) for value in baseline)):
            raise ValueError("grounded native scorer needs complete finite baseline decisions")
        first = decisions[0]
        if first.role_index < 0:
            return baseline
        query = first.operation_name, first.step_index, first.role_index
        if (any((decision.operation_name, decision.step_index, decision.role_index) != query for decision in decisions)
                or len({decision.value for decision in decisions}) != len(decisions)):
            raise ValueError("grounded native decisions mix role slots or register identities")
        evidence, keys = self.observation_reader(operation_name=query[0], step_index=query[1],
            role_index=query[2], admitted_registers=tuple(decision.value for decision in decisions))
        if (len(evidence.roles) != 1 or evidence.roles[0].role != f"{query[0]}:operand:{query[2]}"
                or set(keys) != {decision.value for decision in decisions} or len(set(keys.values())) != len(keys)):
            raise ValueError("grounded native observation changed the grammar role or source identities")
        role = evidence.roles[0]
        costs = self.engine.costs(evidence)
        if any((role.identity, key) not in costs for key in keys.values()):
            raise ValueError("grounded native observation lacks an admitted register")
        records = {record.key: record for record in evidence.context.referents}
        contextual = {key: candidate_cost(evidence.context, role, records[key],
                      observed_cost=costs[role.identity, key]) for key in keys.values()}
        if any(value is None or not math.isfinite(value) for value in contextual.values()):
            raise ValueError("grounded native observation contains inadmissible contextual evidence")
        combined = tuple(score - contextual[keys[decision.value]]
                         for score, decision in zip(baseline, decisions, strict=True))
        self.receipts.append({"source_id": evidence.source_id, "operation": query[0], "step": query[1],
            "role_id": role.role, "candidate_ids": tuple(keys[decision.value] for decision in decisions),
            "baseline_scores": baseline, "combined_scores": combined,
            "contextual_costs": tuple(contextual[keys[decision.value]] for decision in decisions),
            "source_observations": evidence.receipt(), "labels_available_to_scorer": False})
        return combined


class GroundedBindingChartSolver:
    """One source-bound, measured depth bank and the fitted pointer engine.

    Depth states have shape token x depth x hidden, not copies of one depth.
    Callers own acquisition, exact model identity and model-lane custody.
    """
    def __init__(self, engine, source_id, depth_states, *, minimum_margin=0., max_seconds=5.):
        if (not isinstance(source_id, str) or not source_id or depth_states.ndim != 3
                or min(depth_states.shape) < 1 or not mx.all(mx.isfinite(depth_states)).item()
                or depth_states.shape[1:] != (engine.pointer.depths, engine.pointer.hidden_width)
                or not math.isfinite(minimum_margin) or minimum_margin < 0
                or not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("grounded chart bridge needs aligned actual source depths and bounded inference")
        self.engine, self.source_id = engine, source_id
        self.depth_states = mx.array(depth_states)
        self.minimum_margin, self.max_seconds = minimum_margin, max_seconds
        self.last_resolution = None
        self.resolutions = []

    def __call__(self, chart, *, operation_nodes, input_spans, inputs, source_id, time_limit_s=None):
        if source_id != self.source_id or len(operation_nodes) != len(chart.options) or len(inputs) != chart.n_inputs:
            raise ValueError("grounded chart bridge belongs to a different public source")
        allowance = self.max_seconds if time_limit_s is None else min(self.max_seconds, time_limit_s)
        deadline = time.monotonic() + allowance
        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise ArgumentOptimizationIncompleteError("grounded_chart_bridge_budget_exhausted")
            return value

        def observe(span):
            remaining()
            span.validate_bound(len(self.depth_states))
            return mx.mean(self.depth_states[span.start:span.end], axis=0)

        signatures = [semantic_primitive_type_signature(node.operation) for node in operation_nodes]
        if any(signature is None for signature in signatures):
            raise ValueError("grounded chart bridge needs typed public operations")
        anchors = (*input_spans, *(node.span for node in operation_nodes))
        types = tuple("integer_sequence" if isinstance(value, tuple) else "integer" for value in inputs)
        types += tuple(signature[1] for signature in signatures)
        records = tuple(ContextReferent(source_id, f"register:{index}", types[index],
            f"source-token-span:{span.start}:{span.end}", scope=("program",)) for index, span in enumerate(anchors))
        context = BindingContext(records, {name: None for name in types})
        register_keys = {index: record.key for index, record in enumerate(records)}
        roles, operations, mentions, addresses = [], {}, {}, []
        for step, (node, slots, signature) in enumerate(zip(operation_nodes, chart.options, signatures, strict=True)):
            if len(slots) != len(signature[0]) or any(not pool for pool in slots):
                self.last_resolution = {"status": "infeasible", "margin": None}
                return None
            for slot, (pool, required) in enumerate(zip(slots, signature[0], strict=True)):
                role = BindingRole(f"argument:{step}:{slot}", f"{node.operation}:operand:{slot}", required,
                    scope=("program",), referents=tuple(key for register, key in register_keys.items()
                                                      if register != chart.n_inputs + step))
                roles.append(role)
                operations[role.identity] = observe(node.span)
                # Keep all mention hypotheses; baseline scores weight the
                # starting workspace, never truncate it to the old winner.
                weights = mx.softmax(mx.array([score for score, _register, _span in pool]))
                mentions[role.identity] = mx.sum(mx.stack([observe(span) for _score, _register, span in pool])
                                                * weights[:, None, None], axis=0)
                addresses.append((step, slot))
        candidates = {record.key: observe(span) for record, span in zip(records, anchors, strict=True)}
        allowed = mx.array([[context.eligible(role, record) for record in records] for role in roles])
        adjacency = mx.zeros((len(roles) + len(records),) * 2)
        adjacency[:len(roles), len(roles):] = allowed.astype(mx.float32)
        adjacency[len(roles):, :len(roles)] = allowed.T.astype(mx.float32)
        evidence = GroundedBindingEvidence(source_id, context, tuple(roles), operations, mentions, candidates, adjacency)
        role_features = semantic_role_features(roles)
        updated = [[[] for _slot in node] for node in chart.options]
        edge_receipts = []
        for row, ((step, slot), role) in enumerate(zip(addresses, roles, strict=True)):
            for option_index, (score, register, span) in enumerate(chart.options[step][slot]):
                remaining()
                if type(register) is not int or register not in register_keys or not context.eligible(role, records[register]):
                    raise ValueError("grounded public chart contains an inadmissible register")
                conditioned_mentions = {**mentions, role.identity: observe(span)}
                conditioned_candidates = candidates
                if chart.definition_options is not None:
                    definition = chart.definition_options[step][slot][option_index]
                    if definition is not None:
                        conditioned_candidates = {**candidates, register_keys[register]: observe(definition)}
                conditional = project_grounded_evidence(replace(evidence, mentions=conditioned_mentions,
                    candidates=conditioned_candidates), self.engine.nuisance_projection)
                arrays, mask = conditional.arrays()
                value = self.engine.pointer(*arrays, adjacency=adjacency, allowed=mask,
                                            role_features=role_features)[row, register].item()
                if not math.isfinite(value):
                    raise ValueError("grounded public chart has nonfinite learned relation evidence")
                updated[step][slot].append((score + self.engine.evidence_weight * value, register, span))
                edge_receipts.append({"operation_id": step, "operation": operation_nodes[step].operation,
                    "role_instance": role.identity, "role_id": role.role, "required_type": role.type_name,
                    "candidate_id": register_keys[register], "candidate_source": records[register].source,
                    "mention_span": (span.start, span.end), "baseline_score": score,
                    "learned_relation_score": value, "evidence_weight": self.engine.evidence_weight,
                    "combined_score": score + self.engine.evidence_weight * value})
        augmented = replace(chart, options=tuple(tuple(tuple(pool) for pool in node) for node in updated),
                            option_factors=None, option_relation_evidence=None)
        nested_roles = []
        offset = 0
        for node in chart.options:
            nested_roles.append(tuple(roles[offset:offset + len(node)]))
            offset += len(node)
        resolution = augmented.solve_grounded(context, tuple(nested_roles), register_keys,
            minimum_margin=self.minimum_margin, time_limit_s=remaining())
        self.last_resolution = {"status": resolution.status, "margin": resolution.margin,
                                "source_id": source_id, "all_options_retained": True,
                                "role_bindings": resolution.bindings, "edge_evidence": edge_receipts,
                                "margin_is_probability": False}
        if resolution.assignment is not None:
            arguments = resolution.assignment[1]
            self.last_resolution["graph_signature"] = tuple(sorted(
                (node.operation, node.span.start, node.span.end,
                 tuple((anchors[register].start, anchors[register].end) for register in values))
                for node, values in zip(operation_nodes, arguments, strict=True)))
            self.last_resolution["argument_graph_score"] = resolution.assignment[0]
        self.resolutions.append(self.last_resolution)
        return resolution.assignment if resolution.status == "bound" else None
