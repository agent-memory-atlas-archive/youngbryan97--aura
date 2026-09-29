"""Induce operand-role permutations from source-local relational evidence.

The rule vocabulary is not a list of arithmetic phrases. Each observation
supplies an operation span, its argument spans, and the semantic role carried
by each span. The resulting rule is a permutation of arbitrary arity over
surface arguments. Numeric values and absolute character positions are never
features. Conflicting evidence leaves a relation unresolved.

This is an evidence producer for the existing semantic decoder, not a second
parser or an answer key. Runtime discovery only returns bindings for clauses
whose operation and complete argument frame were observed in fitting data.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from core.learning.semantic_public_inputs import semantic_public_character_inputs
from core.learning.semantic_register_identity import RegisterIdentity

if TYPE_CHECKING:
    from core.learning.semantic_native_grammar import NativeGrammarDecision
    from core.learning.semantic_program_corpus import SemanticProgramExample

_WORD = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)
_ALIAS_WORD_BUDGET = 12


def _input_mentions(source: str) -> tuple[tuple[int, int, int], ...]:
    """Find source-local repeated names for public literals, in either order."""
    literals = semantic_public_character_inputs(source).literals
    tokens = tuple(_WORD.finditer(source))
    mentions = {(item.character_start, item.character_end, index)
                for index, item in enumerate(literals)}
    for index, literal in enumerate(literals):
        before = [token for token in tokens if token.end() <= literal.character_start]
        after = [token for token in tokens if token.start() >= literal.character_end]
        for side, boundary in ((tuple(reversed(before)), literal.character_start),
                               (tuple(after), literal.character_end)):
            if not side:
                continue
            if source[min(boundary, side[0].end()):max(boundary, side[0].start())].strip():
                continue
            included = []
            for token in side[:_ALIAS_WORD_BUDGET]:
                if (not token.group(0).isalpha() and token.group(0) != "-"
                        or any(token.start() < other.character_end
                               and other.character_start < token.end() for other in literals)):
                    break
                included.append(token)
                if token.group(0) == "-" or not included[-1].group(0).isalpha():
                    continue
                start = min(part.start() for part in included)
                end = max(part.end() for part in included)
                phrase = source[start:end]
                pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)
                for match in pattern.finditer(source):
                    if match.span() != (start, end):
                        mentions.add((*match.span(), index))
    return tuple(sorted(mentions))


@dataclass(frozen=True)
class RoleObservation:
    source: str
    operation: tuple[int, int]
    arguments: tuple[tuple[int, int], ...]
    construction: str
    operation_name: str

    def __post_init__(self) -> None:
        spans = (self.operation, *self.arguments)
        if (not self.source or not self.construction or not self.operation_name
                or len(self.arguments) < 2
                or any(len(span) != 2 or not 0 <= span[0] < span[1] <= len(self.source)
                       for span in spans)
                or any(left[0] < right[1] and right[0] < left[1]
                       for index, left in enumerate(spans) for right in spans[index + 1:])):
            raise ValueError("role observation needs disjoint source-bound spans")

    def frame(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Return a local relation and semantic-role-to-surface-slot mapping."""
        ordered = sorted(((self.operation, "operation", -1),
                          *((span, "argument", role) for role, span in enumerate(self.arguments))),
                         key=lambda item: item[0][0])
        tokens: list[str] = []
        cursor = ordered[0][0][0]
        surface_roles: list[int] = []
        for (start, end), kind, role in ordered:
            tokens.extend(word.casefold() for word in _WORD.findall(self.source[cursor:start]))
            tokens.append("<op>" if kind == "operation" else "<arg>")
            if kind == "argument":
                surface_roles.append(role)
            cursor = end
        return tuple(tokens), tuple(surface_roles.index(role) for role in range(len(self.arguments)))


@dataclass(frozen=True)
class RoleRule:
    frame: tuple[str, ...]
    role_to_surface_slot: tuple[int, ...]
    fit_constructions: tuple[str, ...]
    observations: int

    def bind(self, surface_references: tuple[int, ...]) -> tuple[int, ...] | None:
        if len(surface_references) != len(self.role_to_surface_slot):
            return None
        return tuple(surface_references[index] for index in self.role_to_surface_slot)


@dataclass(frozen=True)
class RoleRuleBank:
    """Only consistent, cross-construction rules may propose a binding."""

    rules: tuple[RoleRule, ...]
    operation_forms: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        return {"schema": "aura.semantic_role_rule_bank.v1",
                "rules": [{"frame": list(rule.frame),
                           "role_to_surface_slot": list(rule.role_to_surface_slot),
                           "fit_constructions": list(rule.fit_constructions),
                           "observations": rule.observations} for rule in self.rules],
                "operation_forms": [list(pair) for pair in self.operation_forms]}

    @classmethod
    def from_dict(cls, value: dict) -> RoleRuleBank:
        if not isinstance(value, dict) or value.get("schema") != "aura.semantic_role_rule_bank.v1":
            raise ValueError("role rule bank schema differs")
        try:
            rules = tuple(RoleRule(tuple(row["frame"]), tuple(row["role_to_surface_slot"]),
                tuple(row["fit_constructions"]), row["observations"])
                for row in value["rules"])
            forms = tuple(tuple(pair) for pair in value["operation_forms"])
            bank = cls(rules, forms)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("role rule bank is malformed") from exc
        if (bank.to_dict() != value
                or any(not rule.frame or not all(isinstance(token, str) for token in rule.frame)
                       or not all(type(index) is int for index in rule.role_to_surface_slot)
                       or not all(isinstance(name, str) and name for name in rule.fit_constructions)
                       or type(rule.observations) is not int for rule in rules)
                or any(len(pair) != 2 or not all(isinstance(item, str) and item
                                                  for item in pair) for pair in forms)
                or len({rule.frame for rule in rules}) != len(rules)
                or any(len(rule.role_to_surface_slot) < 2
                       or set(rule.role_to_surface_slot) != set(range(len(rule.role_to_surface_slot)))
                       or not rule.fit_constructions or rule.observations < len(rule.fit_constructions)
                       for rule in rules)
                or len({form for form, _ in forms}) != len(forms)):
            raise ValueError("role rule bank is not canonical")
        return bank

    def for_frame(self, frame: tuple[str, ...]) -> RoleRule | None:
        return next((rule for rule in self.rules if rule.frame == frame), None)

    def input_binding_hypotheses(self, source: str, operation_name: str
                                 ) -> tuple[tuple[int, ...], ...]:
        """Return every locally supported binding without choosing a clause."""
        return _cached_input_binding_hypotheses(self, source, operation_name)

    def _input_binding_hypotheses_uncached(self, source: str, operation_name: str
                                           ) -> tuple[tuple[int, ...], ...]:
        if not source:
            return ()
        mentions = _input_mentions(source)
        aliases = {form for form, name in self.operation_forms if name == operation_name}
        if not aliases or len(semantic_public_character_inputs(source).literals) < 2:
            return ()
        hits = []
        for form in aliases:
            pattern = re.compile(r"(?<!\w)" + re.escape(form) + r"(?!\w)", re.IGNORECASE)
            hits.extend(match.span() for match in pattern.finditer(source))
        if not hits:
            return ()
        proposed = []
        for op_start, op_end in sorted(set(hits)):
            clause_end = min((position for position in (
                source.find(mark, op_end) for mark in (".", ";", "\n"))
                if position >= 0), default=len(source))
            local = [item for item in mentions if op_end <= item[0] < clause_end
                     and item[1] <= clause_end]
            for left_index, left in enumerate(local):
                for right in local[left_index + 1:]:
                    if left[1] > right[0] or left[2] == right[2]:
                        continue
                    observation = RoleObservation(source, (op_start, op_end),
                        ((left[0], left[1]), (right[0], right[1])),
                        "runtime", operation_name)
                    rule = self.for_frame(observation.frame()[0])
                    if rule is None:
                        continue
                    binding = rule.bind((left[2], right[2]))
                    if binding is not None and binding not in proposed:
                        proposed.append(binding)
        return tuple(proposed)

    def first_input_binding(self, source: str, operation_name: str) -> tuple[int, ...] | None:
        """Resolve only one supported binding; leave discourse order open."""
        hypotheses = self.input_binding_hypotheses(source, operation_name)
        return hypotheses[0] if len(hypotheses) == 1 else None


@lru_cache(maxsize=1024)
def _cached_input_binding_hypotheses(bank: RoleRuleBank, source: str, operation_name: str
                                     ) -> tuple[tuple[int, ...], ...]:
    return bank._input_binding_hypotheses_uncached(source, operation_name)


def induce_role_rules(observations: tuple[RoleObservation, ...],
                      *, minimum_constructions: int = 2) -> RoleRuleBank:
    """Fit only relations that survive independent surface constructions."""
    if (type(minimum_constructions) is not int or minimum_constructions < 1
            or not isinstance(observations, tuple)):
        raise ValueError("role induction needs a declared evidence threshold")
    grouped: dict[tuple[str, ...], list[tuple[tuple[int, ...], str]]] = defaultdict(list)
    forms: dict[str, set[str]] = defaultdict(set)
    for item in observations:
        if not isinstance(item, RoleObservation):
            raise ValueError("role induction received an untyped observation")
        frame, permutation = item.frame()
        grouped[frame].append((permutation, item.construction))
        form = item.source[slice(*item.operation)].casefold()
        forms[form].add(item.operation_name)
    rules = []
    for frame, rows in grouped.items():
        mappings = {permutation for permutation, _construction in rows}
        constructions = tuple(sorted({construction for _permutation, construction in rows}))
        if len(mappings) == 1 and len(constructions) >= minimum_constructions:
            rules.append(RoleRule(frame, next(iter(mappings)), constructions, len(rows)))
    return RoleRuleBank(tuple(sorted(rules, key=lambda rule: rule.frame)),
                        tuple(sorted((form, next(iter(names))) for form, names in forms.items()
                                     if len(names) == 1)))


def observations_from_examples(examples: tuple[SemanticProgramExample, ...]
                               ) -> tuple[RoleObservation, ...]:
    """Use only fitting annotations; targets never enter runtime inference."""
    observations = []
    for example in examples:
        if example.split != "train":
            raise ValueError("role rules cannot fit on held construction labels")
        for annotated in example.instructions:
            if len(annotated.argument_spans) != 2:
                continue
            observations.append(RoleObservation(
                example.source_text,
                (annotated.operation_span.start, annotated.operation_span.end),
                tuple((span.start, span.end) for span in annotated.argument_spans),
                example.construction_id, annotated.instruction.op))
    return tuple(observations)


def native_role_rule_adjustments(
    bank: RoleRuleBank,
    source: str,
    choices: tuple[NativeGrammarDecision, ...],
    *,
    input_count: int,
    strength: float,
) -> tuple[float, ...]:
    """Supply an optional nonpositive prior for an unambiguous first binding.

    The relation is fitted on training annotations. No target program or
    answer is available here. A repeated or unrecognized source relation
    leaves the model's scores untouched.
    """
    if (not isinstance(bank, RoleRuleBank) or not isinstance(source, str)
            or type(input_count) is not int or input_count < 1
            or type(strength) not in (int, float) or not math.isfinite(strength)
            or strength < 0 or not isinstance(choices, tuple)):
        raise ValueError("role evidence needs a bank, source, input count, and finite strength")
    neutral = (0.,) * len(choices)
    if not choices or not source or strength == 0:
        return neutral
    first = choices[0]
    if (first.step_index != 0 or first.role_index < 0 or not first.operation_name
            or any((item.step_index, item.role_index, item.operation_name)
                   != (first.step_index, first.role_index, first.operation_name)
                   for item in choices)):
        return neutral
    if len(semantic_public_character_inputs(source).literals) != input_count:
        return neutral
    binding = bank.first_input_binding(source, first.operation_name)
    if binding is None or first.role_index >= len(binding):
        return neutral
    expected = binding[first.role_index]
    registers = []
    for choice in choices:
        if type(choice.value) is int:
            identity = RegisterIdentity.from_absolute(choice.value, input_count=input_count)
        elif isinstance(choice.value, str):
            identity = RegisterIdentity.parse(choice.value)
        else:
            raise ValueError("native role choice is not a register")
        registers.append(identity)
    if RegisterIdentity("input", expected) not in registers:
        return neutral
    return tuple(0. if item == RegisterIdentity("input", expected) else -float(strength)
                 for item in registers)
