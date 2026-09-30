"""Source-bound meaning hypotheses before exact program execution.

This opt-in layer preserves distinct source occurrences even when values match.
Its hypotheses come from target-blind proposals; they are not observations of
what the source meant and carry no serving authority.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from core.evidence.candidate_portfolio import CandidatePortfolioDecision, select_candidate_portfolio
from core.evidence.necessary_condition_selector import (
    NecessaryEvidenceCondition,
    build_necessary_condition_selector,
)
from core.evidence.packet import fuse, observe
from core.learning.procedure_induction import PRIMITIVES_BY_NAME, Instruction, Program
from core.learning.semantic_candidate_bank import SemanticCandidateBank
from core.learning.semantic_graph_counterexamples import (
    ProgramObservationCache,
    compare_program_meanings,
    controlled_counterfactual_inputs,
    counterfactual_inputs,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_floor import (
    compile_source_independent_program_to_floor,
    execute_semantic_floor_program,
)
from core.learning.semantic_program_ir import (
    TokenSpan,
    _is_sha256,
    normalize_semantic_value,
    semantic_value_to_json,
)
from core.learning.semantic_program_portfolio import (
    SemanticProgramPortfolio,
    select_semantic_program_portfolio,
)
from core.learning.semantic_register_identity import RegisterIdentity

if TYPE_CHECKING:
    from core.runtime.gateways import StateGateway, StateMutationReceipt


@dataclass(frozen=True, slots=True)
class SourceOccurrence:
    source_text_sha256: str
    span: TokenSpan

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_text_sha256) or not isinstance(self.span, TokenSpan):
            raise ValueError("meaning occurrence needs a source and token span")

    @property
    def identity(self) -> str:
        return _sha({"source": self.source_text_sha256, "span": self.span.to_dict()})


@dataclass(frozen=True, slots=True)
class MeaningBinding:
    role_index: int
    register: int
    mention: SourceOccurrence
    definition: SourceOccurrence | None

    def __post_init__(self) -> None:
        if (type(self.role_index) is not int or self.role_index < 0
                or type(self.register) is not int or self.register < 0
                or not isinstance(self.mention, SourceOccurrence)
                or (self.definition is not None
                    and (not isinstance(self.definition, SourceOccurrence)
                         or self.definition.source_text_sha256
                         != self.mention.source_text_sha256))):
            raise ValueError("meaning binding has invalid source or register")


@dataclass(frozen=True, slots=True)
class MeaningOperation:
    name: str
    occurrence: SourceOccurrence
    bindings: tuple[MeaningBinding, ...]

    def __post_init__(self) -> None:
        primitive = PRIMITIVES_BY_NAME.get(self.name)
        if (primitive is None
                or not isinstance(self.occurrence, SourceOccurrence)
                or not isinstance(self.bindings, tuple)
                or len(self.bindings) != primitive.arity
                or tuple(binding.role_index for binding in self.bindings)
                != tuple(range(len(self.bindings)))
                or any(binding.mention.source_text_sha256
                       != self.occurrence.source_text_sha256 for binding in self.bindings)):
            raise ValueError("meaning operation has invalid role structure")


@dataclass(frozen=True, slots=True)
class MeaningHypothesis:
    """One coherent, source-grounded proposal with uncalibrated source score."""

    source_text_sha256: str
    bank_receipt_sha256: str
    input_occurrences: tuple[SourceOccurrence, ...]
    operations: tuple[MeaningOperation, ...]
    source_score: float | None
    chart_index: int | None
    graph_index: int | None
    definition_provenance: str

    def __post_init__(self) -> None:
        if (not _is_sha256(self.source_text_sha256)
                or not _is_sha256(self.bank_receipt_sha256)
                or not self.input_occurrences or not self.operations
                or any(item.source_text_sha256 != self.source_text_sha256
                       for item in self.input_occurrences)
                or len({item.identity for item in self.input_occurrences})
                != len(self.input_occurrences)
                or any(op.occurrence.source_text_sha256 != self.source_text_sha256
                       for op in self.operations)
                or (self.source_score is not None and
                    (type(self.source_score) not in (int, float)
                     or not math.isfinite(self.source_score)))
                or self.definition_provenance not in {
                    "unavailable", "register_anchor", "optimizer_selected"}):
            raise ValueError("meaning hypothesis source or evidence differs")
        self.to_program()

    def to_program(self) -> Program:
        instructions = tuple(Instruction(op.name, tuple(binding.register
            for binding in op.bindings)) for op in self.operations)
        program = Program(len(self.input_occurrences), instructions)
        for index, instruction in enumerate(instructions):
            if any(register >= len(self.input_occurrences) + index
                   for register in instruction.args):
                raise ValueError("meaning hypothesis violates forward dataflow")
        return program

    @property
    def identity(self) -> str:
        return _sha({
            "source": self.source_text_sha256,
            "bank": self.bank_receipt_sha256,
            "inputs": [item.identity for item in self.input_occurrences],
            "operations": [{"name": op.name, "occurrence": op.occurrence.identity,
                            "bindings": [{"role": binding.role_index,
                                          "register": binding.register,
                                          "mention": binding.mention.identity,
                                          "definition": (binding.definition.identity
                                                         if binding.definition else None)}
                                         for binding in op.bindings]}
                           for op in self.operations],
        })


@dataclass(frozen=True, slots=True)
class GroundedProgramProposal:
    """A program compiled from one immutable meaning hypothesis."""

    hypothesis: MeaningHypothesis
    program: Program

    def __post_init__(self) -> None:
        if self.program != self.hypothesis.to_program():
            raise ValueError("program differs from its meaning hypothesis")

    @property
    def receipt_sha256(self) -> str:
        return _sha({"meaning": self.hypothesis.identity,
                     "program": self.program.sha(),
                     "source": self.hypothesis.source_text_sha256,
                     "bank": self.hypothesis.bank_receipt_sha256})


@dataclass(frozen=True, slots=True)
class MeaningTransition:
    """One predicted state change, with stable referents and source ancestry."""

    ordinal: int
    operation: MeaningOperation
    argument_identities: tuple[RegisterIdentity, ...]
    argument_values: tuple[Any, ...]
    result_identity: RegisterIdentity
    result: Any
    execution_receipt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MeaningTrajectory:
    hypothesis: MeaningHypothesis
    public_inputs: tuple[Any, ...]
    transitions: tuple[MeaningTransition, ...]

    @property
    def result(self) -> Any:
        return self.transitions[-1].result


@dataclass(frozen=True, slots=True)
class MeaningStageContrast:
    """A source-aligned disagreement to investigate, never an answer key."""

    operation: SourceOccurrence
    left_ordinal: int
    right_ordinal: int
    left_result: Any
    right_result: Any
    left_arguments: tuple[RegisterIdentity, ...]
    right_arguments: tuple[RegisterIdentity, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class MeaningStageInquiry:
    """A source-bound disagreement requiring an outside observation of one step."""

    source_text_sha256: str
    bank_receipt_sha256: str
    operation: SourceOccurrence
    occurrence_index: int
    inputs: tuple[Any, ...]
    predictions: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if (not _is_sha256(self.source_text_sha256)
                or not _is_sha256(self.bank_receipt_sha256)
                or not isinstance(self.operation, SourceOccurrence)
                or self.operation.source_text_sha256 != self.source_text_sha256
                or type(self.occurrence_index) is not int or self.occurrence_index < 0
                or not isinstance(self.inputs, tuple) or not isinstance(self.predictions, tuple)
                or not self.inputs or len(self.inputs) > 32
                or len(self.predictions) < 2
                or any(not _is_sha256(receipt) for receipt, _ in self.predictions)
                or len({receipt for receipt, _ in self.predictions}) != len(self.predictions)
                or len({_sha(value) for _, value in self.predictions}) < 2):
            raise ValueError("meaning stage inquiry needs distinct source-bound predictions")
        try:
            if (any(value != normalize_semantic_value(value) for value in self.inputs)
                    or any(value != normalize_semantic_value(value)
                           for _, value in self.predictions)):
                raise ValueError("stage inquiry values must use the exact semantic algebra")
        except (TypeError, ValueError) as exc:
            raise ValueError("stage inquiry values must use the exact semantic algebra") from exc

    @property
    def identity(self) -> str:
        return _sha({
            "source": self.source_text_sha256,
            "bank": self.bank_receipt_sha256,
            "operation": self.operation.identity,
            "occurrence_index": self.occurrence_index,
            "inputs": self.inputs,
            "predictions": self.predictions,
        })

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "aura.semantic_meaning_stage_inquiry.v1",
            "identity": self.identity,
            "source_text_sha256": self.source_text_sha256,
            "bank_receipt_sha256": self.bank_receipt_sha256,
            "operation": {"source_text_sha256": self.operation.source_text_sha256,
                          "span": self.operation.span.to_dict()},
            "occurrence_index": self.occurrence_index,
            "inputs": [semantic_value_to_json(value) for value in self.inputs],
            "predictions": [[receipt, semantic_value_to_json(value)]
                            for receipt, value in self.predictions],
            "observation_required": True,
            "serving_authority": False,
        }
        return {**body, "content_sha256": _sha(body)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MeaningStageInquiry:
        if not isinstance(payload, Mapping) or payload.get("schema") != (
                "aura.semantic_meaning_stage_inquiry.v1"):
            raise ValueError("invalid stage inquiry record")
        operation = payload["operation"]
        inquiry = cls(
            payload["source_text_sha256"], payload["bank_receipt_sha256"],
            SourceOccurrence(operation["source_text_sha256"], TokenSpan(**operation["span"])),
            payload["occurrence_index"],
            tuple(normalize_semantic_value(value) for value in payload["inputs"]),
            tuple((receipt, normalize_semantic_value(value))
                  for receipt, value in payload["predictions"]),
        )
        if _sha(inquiry.to_dict()) != _sha(dict(payload)):
            raise ValueError("stage inquiry content or source binding differs")
        return inquiry

    async def retain(self, gateway: StateGateway) -> StateMutationReceipt:
        from core.runtime.gateways import StateMutationRequest

        return await gateway.mutate(StateMutationRequest(
            key=self.identity, new_value=self.to_dict(), domain="semantic_stage_inquiries",
            cause="retain source-bound intermediate interpretation distinction",
        ))

    @classmethod
    async def restore(cls, gateway: StateGateway, identity: str) -> MeaningStageInquiry | None:
        payload = await gateway.read(identity, domain="semantic_stage_inquiries", fresh=True)
        if payload is None:
            return None
        inquiry = cls.from_dict(payload)
        if inquiry.identity != identity:
            raise ValueError("stored stage inquiry belongs to another request")
        return inquiry


@dataclass(frozen=True, slots=True)
class ObservedMeaningStageInquiry:
    """Caller-supplied stage result whose measurement authority is upstream."""

    inquiry: MeaningStageInquiry
    observed_result: int | tuple[int, ...]
    origin: str
    ref: str

    def __post_init__(self) -> None:
        if (not isinstance(self.inquiry, MeaningStageInquiry)
                or not isinstance(self.origin, str) or not self.origin.strip()
                or not isinstance(self.ref, str) or not self.ref.strip()):
            raise ValueError("stage observation needs a bound source and reference")
        normalize_semantic_value(self.observed_result)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "aura.observed_meaning_stage_inquiry.v1",
            "inquiry": self.inquiry.to_dict(),
            "observed_result": semantic_value_to_json(self.observed_result),
            "origin": self.origin, "ref": self.ref,
            "serving_authority": False,
        }
        return {**body, "content_sha256": _sha(body)}

    @property
    def identity(self) -> str:
        return self.to_dict()["content_sha256"]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObservedMeaningStageInquiry:
        if not isinstance(payload, Mapping) or payload.get("schema") != (
                "aura.observed_meaning_stage_inquiry.v1"):
            raise ValueError("invalid observed stage inquiry record")
        observation = cls(
            MeaningStageInquiry.from_dict(payload["inquiry"]),
            normalize_semantic_value(payload["observed_result"]),
            payload["origin"], payload["ref"],
        )
        if _sha(observation.to_dict()) != _sha(dict(payload)):
            raise ValueError("observed stage inquiry content or bindings differ")
        return observation

    async def retain(self, gateway: StateGateway) -> StateMutationReceipt:
        from core.runtime.gateways import StateMutationRequest

        return await gateway.mutate(StateMutationRequest(
            key=self.identity, new_value=self.to_dict(), domain="semantic_stage_observations",
            cause="record caller-identified intermediate interpretation feedback",
        ))

    @classmethod
    async def restore_all(cls, gateway: StateGateway) -> tuple[ObservedMeaningStageInquiry, ...]:
        rows = await gateway.snapshot(domain="semantic_stage_observations")
        observations = []
        for key, payload in sorted(rows.items()):
            observation = cls.from_dict(payload)
            if observation.identity != key:
                raise ValueError("stored stage observation identity differs")
            observations.append(observation)
        return tuple(observations)


def plan_meaning_stage_inquiries(
    proposals: Sequence[GroundedProgramProposal], probes: Sequence[tuple[Any, ...]], *,
    max_inquiries: int=8, fuel: int=100_000,
) -> tuple[MeaningStageInquiry, ...]:
    """Find discriminating step tests without treating predictions as truth."""
    if (not proposals or type(max_inquiries) is not int or max_inquiries < 1
            or type(fuel) is not int or fuel < 1):
        raise ValueError("meaning stage inquiry settings are invalid")
    source = proposals[0].hypothesis.source_text_sha256
    bank = proposals[0].hypothesis.bank_receipt_sha256
    receipts = tuple(proposal.receipt_sha256 for proposal in proposals)
    if (len(set(receipts)) != len(receipts)
            or any(proposal.hypothesis.source_text_sha256 != source
                   or proposal.hypothesis.bank_receipt_sha256 != bank
                   for proposal in proposals)):
        raise ValueError("meaning stage inquiry needs one unique source bank")
    rows = []
    baseline = probes[0] if probes else None
    for raw_inputs in dict.fromkeys(probes):
        if not isinstance(raw_inputs, tuple) or any(
                proposal.program.n_inputs != len(raw_inputs) for proposal in proposals):
            raise ValueError("meaning stage inquiry input geometry differs")
        try:
            trajectories = tuple(predict_meaning_trajectory(
                proposal.hypothesis, raw_inputs, fuel=fuel) for proposal in proposals)
        except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
            continue
        by_proposal = []
        for trajectory in trajectories:
            seen: Counter[str] = Counter()
            stages = {}
            for transition in trajectory.transitions:
                key = (transition.operation.occurrence.identity,
                       seen[transition.operation.occurrence.identity])
                seen[transition.operation.occurrence.identity] += 1
                stages[key] = transition
            by_proposal.append(stages)
        shared = set.intersection(*(set(stages) for stages in by_proposal))
        for key in shared:
            predictions = tuple(sorted((receipt, stages[key].result)
                                       for receipt, stages in zip(receipts, by_proposal, strict=True)))
            if len({_sha(value) for _, value in predictions}) < 2:
                continue
            operation = by_proposal[0][key].operation.occurrence
            inquiry = MeaningStageInquiry(source, bank, operation, key[1],
                                          trajectories[0].public_inputs, predictions)
            partition = Counter(_sha(value) for _, value in predictions)
            changed_inputs = sum(left != right for left, right in zip(
                baseline, raw_inputs, strict=True)) if baseline is not None else 0
            rows.append(((-len(partition), sum(count * count for count in partition.values()),
                          changed_inputs, inquiry.operation.span.start, inquiry.identity), inquiry))
    rows.sort(key=lambda item: item[0])
    return tuple(inquiry for _, inquiry in rows[:max_inquiries])


def reconcile_meaning_stage_inquiries(
    portfolio: SemanticProgramPortfolio,
    feedback: Sequence[tuple[MeaningStageInquiry, Any, str, str]], *,
    fuel: int=100_000,
) -> CandidatePortfolioDecision | None:
    """Use only caller-observed step results to eliminate incompatible proposals."""
    proposals = portfolio.source_bound_proposals
    if not proposals or not feedback:
        raise ValueError("stage feedback needs a source-bound portfolio and observations")
    observed = {}
    predictions = {}
    for inquiry, raw_result, origin, ref in feedback:
        if (not isinstance(inquiry, MeaningStageInquiry)
                or not isinstance(origin, str) or not origin.strip()
                or not isinstance(ref, str) or not ref.strip()):
            raise ValueError("stage feedback needs identified independent observations")
        replay = plan_meaning_stage_inquiries(proposals, (inquiry.inputs,),
                                              max_inquiries=128, fuel=fuel)
        if inquiry.identity not in {item.identity for item in replay}:
            raise ValueError("stage feedback differs from source-bound execution")
        result = normalize_semantic_value(raw_result)
        identity = (origin, ref)
        if identity in observed and observed[identity] != (inquiry.identity, result):
            raise ValueError("one stage observation identity cannot contradict itself")
        observed[identity] = (inquiry.identity, result)
        predictions[inquiry.identity] = dict(inquiry.predictions)
    executions = dict(portfolio.executions)
    measurements = {
        proposal.receipt_sha256: {
            "executable_program": float(executions[proposal.receipt_sha256]["completed"]),
            "observed_stage_agreement": float(all(
                predictions[inquiry_id][proposal.receipt_sha256] == result
                for inquiry_id, result in observed.values())),
        } for proposal in proposals
    }
    if not any(all(values.values()) for values in measurements.values()):
        return None
    subject = "meaning_stage:" + _sha(sorted((origin, ref, inquiry_id, result)
        for (origin, ref), (inquiry_id, result) in observed.items()))
    packet = fuse(tuple(observe(1., origin=origin, ref=ref, subject=subject)
                        for origin, ref in sorted(observed)))
    provenance = {name: packet for name in measurements}
    selector = build_necessary_condition_selector((
        NecessaryEvidenceCondition("executable_program", 1.,
                                   "stage_answer_requires_completed_floor_execution"),
        NecessaryEvidenceCondition("observed_stage_agreement", 1.,
                                   "program_must_agree_with_independently_observed_stage"),
    ))
    incumbent = portfolio.decision.selected
    if incumbent not in measurements:
        raise ValueError("stage portfolio incumbent is absent")
    return select_candidate_portfolio(selector, incumbent=incumbent,
                                      measurements=measurements, provenance=provenance)


def predict_meaning_trajectory(
    hypothesis: MeaningHypothesis, public_inputs: tuple[Any, ...], *, fuel: int = 100_000,
) -> MeaningTrajectory:
    """Run every program prefix on the canonical floor and retain causal roles.

    A result is a hypothesis prediction. Its source spans say where the
    proposal came from; they do not establish that the source intended it.
    """
    if not isinstance(hypothesis, MeaningHypothesis) or type(fuel) is not int or fuel < 1:
        raise ValueError("meaning trajectory needs a hypothesis and positive floor fuel")
    program = hypothesis.to_program()
    if not isinstance(public_inputs, tuple) or len(public_inputs) != program.n_inputs:
        raise ValueError("meaning trajectory public inputs differ")
    inputs = tuple(normalize_semantic_value(value) for value in public_inputs)
    values = list(inputs)
    transitions = []
    for ordinal, operation in enumerate(hypothesis.operations):
        prefix = Program(program.n_inputs, program.instructions[:ordinal + 1])
        compiled = compile_source_independent_program_to_floor(
            prefix, inputs, provenance_receipt_sha256=hypothesis.identity)
        execution = execute_semantic_floor_program(compiled, fuel=fuel)
        identities = tuple(RegisterIdentity.from_absolute(
            binding.register, input_count=program.n_inputs) for binding in operation.bindings)
        transitions.append(MeaningTransition(
            ordinal, operation, identities,
            tuple(values[binding.register] for binding in operation.bindings),
            RegisterIdentity("result", ordinal), execution.result, execution.receipt))
        values.append(execution.result)
    return MeaningTrajectory(hypothesis, inputs, tuple(transitions))


def compare_meaning_stages(
    left: MeaningTrajectory, right: MeaningTrajectory,
) -> tuple[MeaningStageContrast, ...]:
    """Find disagreements at shared source operations, without aligning by slot."""
    if (left.hypothesis.source_text_sha256 != right.hypothesis.source_text_sha256
            or left.hypothesis.bank_receipt_sha256 != right.hypothesis.bank_receipt_sha256
            or left.public_inputs != right.public_inputs):
        raise ValueError("meaning stage comparison needs one source bank and input state")
    right_by_occurrence: dict[str, list[MeaningTransition]] = {}
    for step in right.transitions:
        right_by_occurrence.setdefault(step.operation.occurrence.identity, []).append(step)
    seen: dict[str, int] = {}
    contrasts = []
    for own in left.transitions:
        identity = own.operation.occurrence.identity
        position = seen.get(identity, 0)
        seen[identity] = position + 1
        peers = right_by_occurrence.get(identity, ())
        if position >= len(peers):
            continue
        peer = peers[position]
        binding_differs = own.argument_identities != peer.argument_identities
        result_differs = own.result != peer.result
        if not binding_differs and not result_differs:
            continue
        reason = ("binding_and_result" if binding_differs and result_differs else
                  "binding" if binding_differs else "result")
        contrasts.append(MeaningStageContrast(
            own.operation.occurrence, own.ordinal, peer.ordinal, own.result,
            peer.result, own.argument_identities, peer.argument_identities, reason))
    return tuple(contrasts)


def compare_source_bound_meanings(
    left: GroundedProgramProposal,
    right: GroundedProgramProposal,
    public_inputs: tuple[Any, ...],
    *,
    counterfactual_count: int=32,
    seed: int=0,
    fuel: int=100000,
    observation_cache: ProgramObservationCache | None=None,
) -> dict[str, Any]:
    """Test proposed consequences without treating execution as source truth."""
    if (left.hypothesis.source_text_sha256 != right.hypothesis.source_text_sha256
            or left.hypothesis.bank_receipt_sha256 != right.hypothesis.bank_receipt_sha256):
        raise ValueError("meaning comparison requires one immutable source bank")
    if len(public_inputs) != left.program.n_inputs or left.program.n_inputs != right.program.n_inputs:
        raise ValueError("meaning comparison public input geometry differs")
    controlled = controlled_counterfactual_inputs(
        public_inputs, count=counterfactual_count // 2, seed=seed)
    varied = counterfactual_inputs(
        public_inputs, count=counterfactual_count - counterfactual_count // 2, seed=seed)
    probes = tuple(dict.fromkeys((*controlled, *varied)))
    comparison = compare_program_meanings(left.program, right.program, probes,
        fuel=fuel, observation_cache=observation_cache)
    return {
        "schema": "aura.source_bound_meaning_comparison.v1",
        "source_text_sha256": left.hypothesis.source_text_sha256,
        "bank_receipt_sha256": left.hypothesis.bank_receipt_sha256,
        "left_receipt_sha256": left.receipt_sha256,
        "right_receipt_sha256": right.receipt_sha256,
        "proposed_consequence_comparison": comparison,
        "source_interpretation_status": "unresolved",
        "claim": "program_consequences_only_not_source_interpretation_or_phenomenology",
    }


def select_source_bound_portfolio(
    proposals: Sequence[GroundedProgramProposal],
    public_inputs: tuple[Any, ...],
    *,
    incumbent_receipt_sha256: str,
    fuel: int=2_000_000,
) -> SemanticProgramPortfolio:
    """Keep source ancestry through the canonical execution and inquiry loop."""
    if not proposals:
        raise ValueError("meaning portfolio needs source-bound proposals")
    source = proposals[0].hypothesis.source_text_sha256
    bank = proposals[0].hypothesis.bank_receipt_sha256
    if any(proposal.hypothesis.source_text_sha256 != source
           or proposal.hypothesis.bank_receipt_sha256 != bank for proposal in proposals):
        raise ValueError("meaning inquiry requires one immutable source bank")
    if any(proposal.program.n_inputs != len(public_inputs) for proposal in proposals):
        raise ValueError("meaning inquiry public input geometry differs")
    names = [proposal.receipt_sha256 for proposal in proposals]
    if len(names) != len(set(names)):
        raise ValueError("meaning portfolio has duplicate proposal receipts")
    portfolio = select_semantic_program_portfolio(
        proposals={proposal.receipt_sha256: proposal.program for proposal in proposals},
        provenance={proposal.receipt_sha256: proposal.receipt_sha256 for proposal in proposals},
        public_inputs=public_inputs, observation_sha256=source,
        incumbent=incumbent_receipt_sha256, fuel=fuel,
    )
    return replace(portfolio, source_bound_proposals=tuple(proposals))


def meaning_hypotheses_from_bank(bank: SemanticCandidateBank) -> tuple[MeaningHypothesis, ...]:
    """Expose all retained proposal meanings without assigning a posterior."""
    bank.validate()
    source = bank.receipt["source_text_sha256"]
    if not _is_sha256(source):
        raise ValueError("candidate bank lacks source identity")
    inputs = tuple(SourceOccurrence(source, span) for span in bank.input_spans)
    hypotheses = []
    for candidate in bank.candidates:
        if candidate.argument_spans is None:
            continue
        operations = []
        for index, instruction in enumerate(candidate.program.instructions):
            definitions = (candidate.definition_spans[index]
                           if candidate.definition_spans is not None else None)
            bindings = tuple(MeaningBinding(
                role, register, SourceOccurrence(source, mention),
                SourceOccurrence(source, definitions[role]) if definitions else None,
            ) for role, (register, mention) in enumerate(zip(
                instruction.args, candidate.argument_spans[index], strict=True)))
            operations.append(MeaningOperation(
                instruction.op, SourceOccurrence(source, candidate.operation_spans[index]),
                bindings))
        hypothesis = MeaningHypothesis(source, bank.receipt["receipt_sha256"],
            inputs, tuple(operations), candidate.joint_score, candidate.chart_index,
            candidate.graph_index, candidate.definition_provenance)
        if hypothesis.to_program() != candidate.program:
            raise ValueError("candidate graph and meaning hypothesis differ")
        hypotheses.append(hypothesis)
    return tuple(hypotheses)
