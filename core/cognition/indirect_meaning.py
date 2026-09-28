"""Source-bound, audience-aware interpretation of indirect communication.

The literal utterance, the situation it might describe, and the speaker's
possible communicative purposes remain separate. This adapter uses Aura's
existing relation cases, pragmatic observations, and interpretation audits;
it does not turn a structural analogy or a user's reading into speaker intent.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Literal

from core.cognition.relational_generalization import (
    Interpretation,
    RelationalCase,
    RelationalGeneralizer,
)
from core.cognition.structure_mapping import (
    Graph,
    Relation,
    map_structures_alternatives,
    shuffled_null,
)
from core.language.contextual_usage import (
    UsageEvent,
    UsageRelation,
    lexical_terms,
    source_text_digest,
)
from core.language.pragmatic_evidence import compare_pragmatic_context


@dataclass(frozen=True, slots=True)
class Audience:
    identity: str
    access: Literal["addressed", "overheard", "unknown"]
    source_id: str
    known_source_ids: tuple[str, ...] = ()
    access_excerpt: str = ""


@dataclass(frozen=True, slots=True)
class SourcePassage:
    """Transient source text for checking a proposed graph's cited excerpts."""

    source_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.text.strip() or len(self.text) > 8_000:
            raise ValueError("source passage needs bounded text and identity")


@dataclass(frozen=True, slots=True)
class RelationAnchor:
    """The passage behind one proposed relation, not proof of that relation."""

    case_id: str
    relation: tuple[str, ...]
    source_id: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class AudienceFunction:
    audience: str
    function: str


@dataclass(frozen=True, slots=True)
class PragmaticEvidence:
    """A sourced proposition; a report is not an independent observation."""

    proposition: str
    value: bool
    source_id: str
    kind: Literal["report", "observation"]
    observed_at: float
    excerpt: str
    relation: UsageRelation | None = None


@dataclass(frozen=True, slots=True)
class IndirectReading:
    identity: str
    interpretation: Interpretation
    entity_map: tuple[tuple[str, str], ...]
    predicate_map: tuple[tuple[str, str], ...]
    functions: tuple[AudienceFunction, ...]
    basis_source_ids: tuple[str, ...]
    kind: Literal["literal", "indirect"] = "indirect"


@dataclass(frozen=True, slots=True)
class PragmaticEpisode:
    """The described story and its surrounding situation are distinct cases."""

    utterance: UsageEvent
    context: tuple[UsageEvent, ...]
    literal_case: RelationalCase | Graph
    situation_case: RelationalCase | Graph
    audiences: tuple[Audience, ...]
    passages: tuple[SourcePassage, ...] = ()
    relation_anchors: tuple[RelationAnchor, ...] = ()
    entity_kinds: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        events = (self.utterance, *self.context)
        sources = {event.source_id for event in events}
        passages = {row.source_id: row.text for row in self.passages}
        audience_ids = [audience.identity for audience in self.audiences]
        if (_case_id(self.literal_case) == _case_id(self.situation_case)
                or len(sources) != len(self.context) + 1
                or any(event.context_id != self.utterance.context_id
                       for event in self.context)
                or len(passages) != len(self.passages)
                or set(passages) != sources
                or any(not passage.text or lexical_terms(passage.text) != event.terms
                       or not event.source_text_sha256
                       or source_text_digest(passage.text) != event.source_text_sha256
                       for event in events for passage in self.passages
                       if passage.source_id == event.source_id)
                or not audience_ids or len(set(audience_ids)) != len(audience_ids)
                or any(not audience.identity or audience.source_id not in sources
                       or audience.access not in {"addressed", "overheard", "unknown"}
                       or not audience.access_excerpt
                       or audience.access_excerpt not in passages[audience.source_id]
                       or len(set(audience.known_source_ids)) != len(audience.known_source_ids)
                       or not set(audience.known_source_ids) <= sources
                       or (audience.access != "unknown" and
                           self.utterance.source_id not in audience.known_source_ids)
                       for audience in self.audiences)):
            raise ValueError("pragmatic episode needs distinct sourced cases and audiences")
        cases = {_case_id(self.literal_case): self.literal_case,
                 _case_id(self.situation_case): self.situation_case}
        if (len({(row.case_id, row.relation, row.source_id)
                 for row in self.relation_anchors}) != len(self.relation_anchors)
                or any(row.case_id not in cases
                       or row.relation not in _case_relations(cases[row.case_id])
                       or row.source_id not in passages or not row.excerpt
                       or row.excerpt not in passages[row.source_id]
                       for row in self.relation_anchors)):
            raise ValueError("relation anchors need an exact cited source excerpt")
        if (len({(case_id, entity) for case_id, entity, _kind in self.entity_kinds})
                != len(self.entity_kinds)
                or any(case_id not in cases or entity not in _case_graph(cases[case_id]).objects
                       or not kind for case_id, entity, kind in self.entity_kinds)):
            raise ValueError("graph entity kinds need distinct declared entities")


def _case_id(case: RelationalCase | Graph) -> str:
    return case.case_id if isinstance(case, RelationalCase) else case.name


def _case_graph(case: RelationalCase | Graph) -> Graph:
    if isinstance(case, Graph):
        return case
    return Graph(case.case_id, tuple(Relation(row[0], row[1:])
                                      for row in case.relations))


def _case_relations(case: RelationalCase | Graph) -> tuple[tuple[str, ...], ...]:
    return tuple((row.predicate, *row.args) for row in _case_graph(case).relations)


def _project_relation(
    row: tuple[str, ...], entities: dict[str, str],
    predicates: dict[str, str],
) -> tuple[str, ...] | None:
    if row[0] not in predicates or any(arg not in entities for arg in row[1:]):
        return None
    return (predicates[row[0]], *(entities[arg] for arg in row[1:]))


def _mapped_relations(
    literal: RelationalCase | Graph,
    situation: RelationalCase | Graph,
    entities: dict[str, str],
    predicates: dict[str, str],
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    target = set(_case_relations(situation))
    matched, missing = [], []
    for row in _case_relations(literal):
        projected = _project_relation(row, entities, predicates)
        (matched if projected in target else missing).append(row)
    return tuple(matched), tuple(missing)


def propose_role_bridges(
    episode: PragmaticEpisode, *, as_of: float | None = None,
    max_results: int = 16,
) -> dict:
    """Find source-bound structural possibilities without choosing a message.

    Relation names may differ across domains. This reuses the existing graph
    matcher and its shuffled control. The proposed graphs still need a source
    audit, and a correspondence alone never establishes an intended analogy.
    """
    if as_of is None:
        as_of = max(event.observed_at for event in (
            episode.utterance, *episode.context))
    if not math.isfinite(as_of) or as_of < episode.utterance.observed_at:
        raise ValueError("role-bridge cutoff must follow the utterance")
    literal = _case_graph(episode.literal_case)
    situation = _case_graph(episode.situation_case)
    alternatives = map_structures_alternatives(
        literal, situation, max_results=max_results)
    control = shuffled_null(literal, situation) if alternatives.readings else {
        "measurable": False}
    events = {event.source_id: event for event in (
        episode.utterance, *episode.context)}
    anchors = {(row.case_id, row.relation) for row in episode.relation_anchors
               if events[row.source_id].observed_at <= as_of}
    candidates = []
    for alignment in alternatives.readings:
        anchored = sum((_case_id(episode.literal_case),
                        (source.predicate, *source.args)) in anchors
                       and (_case_id(episode.situation_case),
                            (target.predicate, *target.args)) in anchors
                       for source, target in alignment.matched)
        candidates.append({
            "entity_map": tuple(sorted(alignment.mapping.items())),
            "predicate_map": tuple(sorted(alignment.predicate_mapping.items())),
            "matched_relations": len(alignment.matched),
            "source_anchored_relations": anchored,
            "score": alignment.score,
            "status": ("no_relation_match" if not alignment.matched
                       else "source_anchored_candidate"
                       if anchored == len(alignment.matched)
                       else "unanchored_graph_candidate"),
        })
    return {"schema": "aura.indirect_role_bridges.v1",
            "candidates": tuple(candidates),
            "tied_candidates_truncated": alternatives.truncated,
            "shuffled_control": {key: control[key] for key in (
                "measurable", "score", "null_mean", "separation", "structural")
                if key in control},
            "speaker_intent": "unmeasured", "serving_authority": False}


def _analogy(episode: PragmaticEpisode, reading: IndirectReading, *, as_of: float) -> dict:
    literal_entities = set(_case_graph(episode.literal_case).objects)
    situation_entities = set(_case_graph(episode.situation_case).objects)
    source_predicates = {row[0] for row in _case_relations(episode.literal_case)}
    entities = dict(reading.entity_map)
    predicates = dict(reading.predicate_map)
    if (not entities or not predicates or len(entities) != len(reading.entity_map)
            or len(predicates) != len(reading.predicate_map)
            or len(set(entities.values())) != len(entities)
            or len(set(predicates.values())) != len(predicates)
            or not set(entities) <= literal_entities
            or not set(entities.values()) <= situation_entities
            or not set(predicates) <= source_predicates
            or any(not name or len(name) > 96 for name in predicates.values())):
        raise ValueError("indirect reading has an invalid one-to-one role bridge")
    matched, unmatched = _mapped_relations(
        episode.literal_case, episode.situation_case, entities, predicates)
    events = {event.source_id: event for event in (
        episode.utterance, *episode.context)}
    anchors = {(row.case_id, row.relation): row.source_id
               for row in episode.relation_anchors
               if events[row.source_id].observed_at <= as_of}
    anchored = tuple(row for row in matched
                     if (_case_id(episode.literal_case), row) in anchors
                     and (_case_id(episode.situation_case),
                          _project_relation(row, entities, predicates)) in anchors)
    target_relations = set(_case_relations(episode.situation_case))
    projected_unobserved = tuple(
        projected for row in _case_relations(episode.literal_case)
        if (projected := _project_relation(row, entities, predicates)) is not None
        and (projected not in target_relations
             or (_case_id(episode.situation_case), projected) not in anchors))
    alternatives = []
    targets = tuple(entities.values())
    if len(targets) <= 7:
        for permutation in itertools.permutations(targets):
            if permutation == targets:
                continue
            changed = dict(zip(entities, permutation, strict=True))
            alternative, _ = _mapped_relations(
                episode.literal_case, episode.situation_case, changed, predicates)
            alternatives.append(len(alternative))
    return {
        "matched_relations": matched,
        "source_anchored_relations": anchored,
        "unanchored_matched_relations": tuple(row for row in matched
                                              if row not in anchored),
        "unmatched_literal_relations": unmatched,
        "projected_unobserved_relations": projected_unobserved,
        "unmatched_situation_relations": tuple(
            row for row in _case_relations(episode.situation_case)
            if not any(_project_relation(source, entities, predicates) == row
                       for source in matched)),
        "counterfactual_role_swaps": len(alternatives),
        "best_swapped_match": max(alternatives) if alternatives else None,
        "role_selective": (len(matched) > max(alternatives) if alternatives else None),
        "control_status": ("unmeasured_budget" if len(targets) > 7 else
                           "unmeasured_one_role" if not alternatives else "measured"),
    }


def assess_indirect_meaning(
    episode: PragmaticEpisode,
    readings: tuple[IndirectReading, ...],
    evidence: tuple[PragmaticEvidence, ...],
    *,
    as_of: float | None = None,
) -> dict:
    """Compare rival readings without collapsing a joke, warning, or claim.

    The caller supplies proposed graphs and readings. Structural agreement is
    only a candidate explanation. Reports can make it coherent, observations
    can test its predictions, and neither establishes an unobserved intention.
    """
    if not readings or len({row.identity for row in readings}) != len(readings):
        raise ValueError("pragmatic comparison needs distinct readings")
    if as_of is None:
        as_of = max(event.observed_at for event in (
            episode.utterance, *episode.context))
    if not math.isfinite(as_of) or as_of < episode.utterance.observed_at:
        raise ValueError("pragmatic cutoff must follow the utterance")
    events = {event.source_id: event for event in (
        episode.utterance, *episode.context)}
    passages = {row.source_id: row.text for row in episode.passages}
    sources = set(events)
    available_sources = {source for source, event in events.items()
                         if event.observed_at <= as_of}
    audience_ids = {item.identity for item in episode.audiences}
    if (any(not row.proposition or not row.source_id or type(row.value) is not bool
            or row.source_id not in sources
            or row.kind not in {"report", "observation"}
            or not math.isfinite(row.observed_at)
            or row.observed_at < events[row.source_id].observed_at
            or not row.excerpt or row.excerpt not in passages[row.source_id]
            or (row.kind == "observation" and (
                row.relation is None or row.relation not in events[row.source_id].relations
                or row.relation.kind != "observation"
                or row.relation.source_id != row.source_id
                or row.relation.predicate != row.proposition
                or row.relation.polarity != row.value
                or row.relation.observed_at > row.observed_at))
            or (row.kind == "report" and row.relation is not None and (
                row.relation not in events[row.source_id].relations
                or row.relation.kind != "claim"
                or row.relation.source_id != row.source_id
                or row.relation.predicate != row.proposition
                or row.relation.polarity != row.value
                or row.relation.observed_at > row.observed_at))
            for row in evidence)
            or len({(row.proposition, row.source_id, row.kind)
                    for row in evidence}) != len(evidence)):
        raise ValueError("pragmatic evidence needs distinct sourced propositions")
    observed = {}
    reported = {}
    for item in evidence:
        if item.observed_at > as_of or item.source_id not in available_sources:
            continue
        destination = observed if item.kind == "observation" else reported
        destination.setdefault(item.proposition, []).append(item)
    generalizer = RelationalGeneralizer()
    direct_fit = compare_pragmatic_context(
        episode.utterance, episode.context, as_of=as_of)
    results = []
    if len({row.interpretation.hypothesis for row in readings}) != len(readings):
        raise ValueError("rival readings need distinct hypotheses")
    for reading in readings:
        if (not reading.identity or not reading.interpretation.hypothesis
                or reading.kind not in {"literal", "indirect"}
                or not reading.functions or not reading.basis_source_ids
                or len(set(reading.basis_source_ids)) != len(reading.basis_source_ids)
                or any(source not in sources for source in reading.basis_source_ids)
                or any(not item.function or item.audience not in audience_ids
                       for item in reading.functions)):
            raise ValueError("indirect reading lacks attributed audience functions")
        if reading.kind == "literal":
            if reading.entity_map or reading.predicate_map:
                raise ValueError("literal reading cannot assert a cross-domain bridge")
            bridge = None
        else:
            bridge = _analogy(episode, reading, as_of=as_of)
        claims = dict(reading.interpretation.believed_facts)
        if len(claims) != len(reading.interpretation.believed_facts):
            raise ValueError("one reading predicts conflicting values for a fact")
        # Conflicting independent observations do not become a convenient vote.
        stable_observed = {name: (rows[0].value, rows[0].source_id)
                           for name, rows in observed.items()
                           if len({item.value for item in rows}) == 1}
        audit = generalizer.scrutinize(
            (reading.interpretation,), observations=stable_observed)[0]
        reported_fit = tuple(
            (name, item.value == expected, item.source_id)
            for name, expected in claims.items()
            for item in reported.get(name, ()))
        disputed = tuple(name for name in claims if name in observed
                         and name not in stable_observed)
        missing_basis = tuple(source for source in reading.basis_source_ids
                              if source not in available_sources)
        audience_views = tuple({
            "audience": audience.identity,
            "access": audience.access,
            "function_hypotheses": tuple(item.function for item in reading.functions
                                          if item.audience == audience.identity),
            "available_basis_sources": tuple(source for source in reading.basis_source_ids
                                               if source in audience.known_source_ids
                                               and source in available_sources),
            "missing_basis_sources": tuple(source for source in reading.basis_source_ids
                                             if source not in audience.known_source_ids
                                             or source not in available_sources),
        } for audience in episode.audiences)
        results.append({
            "reading_id": reading.identity,
            "kind": reading.kind,
            "hypothesis": reading.interpretation.hypothesis,
            "audience_functions": tuple((item.audience, item.function)
                                        for item in reading.functions),
            "audience_access": tuple((item.identity, item.access)
                                     for item in episode.audiences),
            "audience_claims_status": "proposed_not_verified",
            "audience_views": audience_views,
            "basis_source_ids": reading.basis_source_ids,
            "missing_basis_at_cutoff": missing_basis,
            "analogy": bridge,
            "observed_support": audit.supported,
            "observed_contradictions": audit.contradicted,
            "unmeasured_predictions": audit.unmeasured,
            "disputed_observations": disputed,
            "reported_fit": reported_fit,
            "status": ("basis_not_yet_available" if missing_basis
                       else "observationally_contradicted" if audit.contradicted
                       else "unresolved_conflicting_observations" if disputed
                       else "literal_candidate" if reading.kind == "literal"
                       else "structurally_unsupported" if not bridge["matched_relations"]
                       else "source_unanchored_analogy" if not bridge[
                           "source_anchored_relations"]
                       else "partially_source_anchored_analogy" if len(bridge[
                           "source_anchored_relations"]) < len(bridge["matched_relations"])
                       else "ambiguous_role_mapping" if bridge["role_selective"] is False
                       else "candidate_not_intent_proof"),
        })
    inquiries = generalizer.plan_discrimination(
        tuple(row.interpretation for row in readings), observations={
            name: value for name, value in stable_observed.items()
            if name not in {item for row in results for item in row["disputed_observations"]}
        }) if len(readings) > 1 else ()
    return {
        "schema": "aura.indirect_meaning_assessment.v1",
        "utterance_source_id": episode.utterance.source_id,
        "literal_case_id": _case_id(episode.literal_case),
        "situation_case_id": _case_id(episode.situation_case),
        "direct_context_fit": direct_fit,
        "entity_kinds": episode.entity_kinds,
        "delivery_cues": tuple((event.source_id, cue.name, cue.value,
                                 cue.channel, cue.source_id)
                                for event in (episode.utterance, *episode.context)
                                if event.observed_at <= as_of
                                for cue in event.cues if cue.observed_at <= as_of),
        "readings": tuple(results),
        "discriminating_inquiries": tuple((item.proposition, item.decisive)
                                         for item in inquiries),
        "speaker_intent": "unmeasured",
        "literal_content_retained": True,
        "serving_authority": False,
    }


__all__ = ["Audience", "AudienceFunction", "IndirectReading", "RelationAnchor",
           "PragmaticEpisode", "PragmaticEvidence", "SourcePassage",
           "assess_indirect_meaning", "propose_role_bridges"]
