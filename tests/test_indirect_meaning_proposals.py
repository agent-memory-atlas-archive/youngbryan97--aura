"""Model-generated scene graphs remain source-cited hypotheses."""

import pytest
from pydantic import ValidationError

from core.cognition import semantic_runtime
from core.cognition.concept_handle import ConceptRegistry
from core.cognition.indirect_meaning import SourcePassage, assess_indirect_meaning
from core.cognition.indirect_meaning_proposals import (
    IndirectMeaningProposal,
    admit_indirect_proposal,
    consult_indirect_meaning,
)
from core.cognition.semantic_development import SemanticDevelopment
from core.language.contextual_usage import UsageEvent


def _inputs():
    texts = {
        "utterance": "A mentor told an engineer a fable: a ruler removed a maker's "
                     "permit after the maker challenged a decree.",
        "context": "The board revoked the engineer's credential after the engineer "
                   "challenged policy. The board hears their conversations.",
    }
    utterance = UsageEvent.from_text("utterance", "setting", texts["utterance"],
                                     observed_at=10)
    context = (UsageEvent.from_text("context", "setting", texts["context"],
                                    observed_at=9),)
    passages = tuple(SourcePassage(key, value) for key, value in texts.items())
    proposal = {
        "literal_graph": {
            "entities": [{"name": name, "kind": kind} for name, kind in (
                ("ruler", "authority"), ("maker", "person"),
                ("permit", "credential"), ("decree", "policy"))],
            "relations": [
                {"predicate": "removes", "args": ["ruler", "permit", "maker"],
                 "source_id": "utterance", "excerpt": "ruler removed a maker's permit"},
                {"predicate": "challenges", "args": ["maker", "decree"],
                 "source_id": "utterance", "excerpt": "maker challenged a decree"},
            ],
        },
        "situation_graph": {
            "entities": [{"name": name, "kind": kind} for name, kind in (
                ("board", "institution"), ("engineer", "person"),
                ("credential", "credential"), ("policy", "policy"))],
            "relations": [
                {"predicate": "revokes", "args": ["board", "credential", "engineer"],
                 "source_id": "context", "excerpt": "board revoked the engineer's credential"},
                {"predicate": "challenges", "args": ["engineer", "policy"],
                 "source_id": "context", "excerpt": "engineer challenged policy"},
            ],
        },
        "audiences": [
            {"identity": "engineer", "access": "addressed", "source_id": "utterance",
             "access_excerpt": "mentor told an engineer",
             "known_source_ids": ["utterance", "context"]},
            {"identity": "board", "access": "overheard", "source_id": "context",
             "access_excerpt": "board hears their conversations",
             "known_source_ids": ["utterance", "context"]},
        ],
        "readings": [{
            "identity": "caution", "hypothesis": "the fable cautions the engineer",
            "kind": "indirect",
            "entity_map": [["ruler", "board"], ["maker", "engineer"],
                           ["permit", "credential"], ["decree", "policy"]],
            "predicate_map": [["removes", "revokes"],
                              ["challenges", "challenges"]],
            "functions": [{"audience": "engineer", "function": "caution"},
                          {"audience": "board", "function": "fable"}],
            "basis_source_ids": ["utterance", "context"],
            "predictions": [{"proposition": "mentor_intended_caution",
                             "value": True}],
        }],
    }
    return utterance, context, passages, proposal


class _Advisor:
    def __init__(self, proposal):
        self.proposal = proposal
        self.prompts = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return self.proposal


@pytest.mark.asyncio
async def test_typed_model_proposal_supports_three_place_relation_without_truth_upgrade():
    utterance, context, passages, payload = _inputs()
    advisor = _Advisor(payload)
    result = await consult_indirect_meaning(
        utterance, context, passages, advisor=advisor)
    assert result["status"] == "source_cited_hypotheses"
    assert result["readings"][0].kind == "literal"
    assert result["readings"][1].kind == "indirect"
    assert result["episode"].entity_kinds[0] == ("literal", "ruler", "authority")
    reading = result["assessment"]["readings"][1]
    assert reading["analogy"]["matched_relations"] == (
        ("removes", "ruler", "permit", "maker"),
        ("challenges", "maker", "decree"))
    assert len(reading["analogy"]["source_anchored_relations"]) == 2
    assert reading["observed_support"] == ()
    assert reading["unmeasured_predictions"] == ("mentor_intended_caution",)
    assert result["assessment"]["speaker_intent"] == "unmeasured"
    assert result["serving_authority"] is False
    assert len(advisor.prompts) == 1


@pytest.mark.asyncio
async def test_forged_graph_excerpt_is_not_admitted():
    utterance, context, passages, payload = _inputs()
    payload["literal_graph"]["relations"][0]["excerpt"] = "a nonexisting line"
    result = await consult_indirect_meaning(
        utterance, context, passages, advisor=_Advisor(payload))
    assert result["status"] == "unadmitted_model_proposal"
    assert result["serving_authority"] is False


@pytest.mark.asyncio
async def test_consultation_rejects_source_text_with_unchanged_words():
    utterance, context, passages, payload = _inputs()
    tampered = tuple(SourcePassage(row.source_id, row.text.replace(
        "fable:", "fable;")) if row.source_id == "utterance" else row
        for row in passages)
    advisor = _Advisor(payload)
    with pytest.raises(ValueError, match="exact observed source texts"):
        await consult_indirect_meaning(
            utterance, context, tampered, advisor=advisor)
    assert advisor.prompts == []


def test_model_cannot_supply_an_observation_or_source_of_its_own():
    utterance, context, passages, payload = _inputs()
    payload["observations"] = [{"proposition": "mentor_intended_caution",
                                "value": True}]
    with pytest.raises(ValidationError, match="Extra inputs"):
        IndirectMeaningProposal.model_validate(payload)
    payload.pop("observations")
    payload["situation_graph"]["relations"][0]["source_id"] = "utterance"
    payload["situation_graph"]["relations"][0]["excerpt"] = (
        "ruler removed a maker's permit")
    with pytest.raises(ValueError, match="independent context"):
        admit_indirect_proposal(utterance, context, passages,
                                IndirectMeaningProposal.model_validate(payload))


def test_admitted_graphs_still_require_external_observations():
    utterance, context, passages, payload = _inputs()
    episode, readings = admit_indirect_proposal(
        utterance, context, passages, IndirectMeaningProposal.model_validate(payload))
    assessment = assess_indirect_meaning(episode, readings, ())
    assert assessment["readings"][1]["status"] == "candidate_not_intent_proof"
    assert assessment["readings"][1]["reported_fit"] == ()


@pytest.mark.asyncio
async def test_invalid_typed_output_returns_unadmitted_instead_of_escaping():
    utterance, context, passages, payload = _inputs()
    payload["observations"] = [{"proposition": "mentor_intended_caution",
                                "value": True}]
    result = await consult_indirect_meaning(
        utterance, context, passages, advisor=_Advisor(payload))
    assert result["status"] == "unadmitted_model_proposal"
    assert result["serving_authority"] is False


@pytest.mark.asyncio
async def test_live_service_requires_retained_sources_before_and_after_model_call(
        tmp_path, monkeypatch):
    utterance, context, passages, payload = _inputs()
    service = SemanticDevelopment(registry=ConceptRegistry(),
                                  state_path=tmp_path / "semantic.json")
    monkeypatch.setattr(semantic_runtime, "get_semantic_development", lambda: service)
    advisor = _Advisor(payload)
    with pytest.raises(ValueError, match="not retained unchanged"):
        await semantic_runtime.consult_indirect_episode(
            utterance, context, passages, advisor=advisor)
    assert advisor.prompts == []
    for event in (utterance, *context):
        service.observe_usage(event)
    result = await semantic_runtime.consult_indirect_episode(
        utterance, context, passages, advisor=advisor)
    assert result["status"] == "source_cited_hypotheses"
    assert result["assessment"]["serving_authority"] is False

    class _EvictingAdvisor(_Advisor):
        async def generate(self, prompt, **kwargs):
            service.usage_events.pop("context")
            return await super().generate(prompt, **kwargs)

    with pytest.raises(ValueError, match="not retained unchanged"):
        await semantic_runtime.consult_indirect_episode(
            utterance, context, passages, advisor=_EvictingAdvisor(payload))
