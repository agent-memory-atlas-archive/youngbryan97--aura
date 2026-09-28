"""Indirect communication is tested as sourced relations, not a keyword rule."""

from dataclasses import replace

import pytest

from core.cognition import semantic_runtime
from core.cognition.concept_handle import ConceptRegistry
from core.cognition.indirect_meaning import (
    Audience,
    AudienceFunction,
    IndirectReading,
    PragmaticEpisode,
    PragmaticEvidence,
    RelationAnchor,
    SourcePassage,
    assess_indirect_meaning,
    propose_role_bridges,
)
from core.cognition.relational_generalization import Interpretation, RelationalCase
from core.cognition.semantic_development import SemanticDevelopment
from core.language.contextual_usage import (
    MeaningFeedback,
    UsageCue,
    UsageEvent,
    UsageRelation,
)


def _scene():
    texts = {
        "fable": "A cartographer told a pilot a fable: a monarch punished a musician "
                 "and broke an instrument after the musician joined the royal tune.",
        "public": "The council monitors hallway conversations and punishes open warnings.",
        "private": "The pilot planned to challenge the council with a new signal. "
                   "The cartographer sought the pilot out afterward.",
        "record": "A council record shows it retaliated against the pilot and disabled "
                  "the pilot's signal after a previous challenge.",
    }
    at = {"fable": 10.0, "public": 9.0, "private": 8.0, "record": 11.0}
    retaliation = UsageRelation("council", "retaliation_likely", "pilot",
                                "scene", "observation", "record", 11)
    events = {key: UsageEvent.from_text(
        key, "scene", text, observed_at=at[key],
        cues=(UsageCue("strained_laugh", True, "audio", "mic:1", at[key]),)
        if key == "fable" else (),
        relations=(retaliation,) if key == "record" else ())
        for key, text in texts.items()}
    literal = RelationalCase("story", (("musician", "person"),
                                       ("monarch", "ruler"),
                                       ("instrument", "object")), (
        ("joins", "musician", "monarch"),
        ("punishes", "monarch", "musician"),
        ("breaks", "monarch", "instrument"),
    ))
    situation = RelationalCase("situation", (("pilot", "person"),
                                              ("council", "institution"),
                                              ("signal", "object")), (
        ("challenges", "pilot", "council"),
        ("retaliates", "council", "pilot"),
        ("disables", "council", "signal"),
    ))
    episode = PragmaticEpisode(
        events["fable"], (events["public"], events["private"], events["record"]),
        literal, situation,
        (Audience("pilot", "addressed", "private",
                  ("fable", "public", "private", "record"),
                  "pilot planned to challenge the council"),
         Audience("council", "overheard", "public", ("fable", "public"),
                  "council monitors hallway conversations")),
        tuple(SourcePassage(key, text) for key, text in texts.items()),
        (RelationAnchor("story", ("joins", "musician", "monarch"), "fable",
                        "musician joined the royal tune"),
         RelationAnchor("story", ("punishes", "monarch", "musician"), "fable",
                        "monarch punished a musician"),
         RelationAnchor("story", ("breaks", "monarch", "instrument"), "fable",
                        "broke an instrument"),
         RelationAnchor("situation", ("challenges", "pilot", "council"),
                        "private", "pilot planned to challenge the council"),
         RelationAnchor("situation", ("retaliates", "council", "pilot"),
                        "record", "retaliated against the pilot"),
         RelationAnchor("situation", ("disables", "council", "signal"),
                        "record", "disabled the pilot's signal")),
    )
    bridge = (("musician", "pilot"), ("monarch", "council"),
              ("instrument", "signal"))
    predicates = (("joins", "challenges"), ("punishes", "retaliates"),
                  ("breaks", "disables"))
    literal_reading = IndirectReading(
        "literal", Interpretation("story also works as fiction"), (), (),
        (AudienceFunction("pilot", "amusement"),
         AudienceFunction("council", "amusement")), ("fable",), kind="literal")
    warning = IndirectReading(
        "warning", Interpretation("coded caution", believed_facts=(
            ("retaliation_likely", True), ("monitoring_present", True),
            ("speaker_intended_warning", True))), bridge, predicates,
        (AudienceFunction("pilot", "caution"),
         AudienceFunction("council", "ordinary story")),
        ("fable", "private", "record"))
    observations = (
        PragmaticEvidence("retaliation_likely", True, "record", "observation", 11,
                          "retaliated against the pilot", retaliation),
        PragmaticEvidence("monitoring_present", True, "public", "report", 9,
                          "monitors hallway conversations"),
    )
    return episode, (literal_reading, warning), observations


def test_story_and_covert_warning_coexist_without_intent_certification():
    episode, readings, evidence = _scene()
    result = assess_indirect_meaning(episode, readings, evidence, as_of=11)
    literal, warning = result["readings"]
    assert literal["kind"] == "literal"
    assert literal["status"] == "literal_candidate"
    assert warning["status"] == "candidate_not_intent_proof"
    assert len(warning["analogy"]["matched_relations"]) == 3
    assert len(warning["analogy"]["source_anchored_relations"]) == 3
    assert warning["analogy"]["role_selective"] is True
    assert warning["observed_support"] == (("retaliation_likely", "record"),)
    assert warning["reported_fit"] == (("monitoring_present", True, "public"),)
    assert warning["unmeasured_predictions"] == (
        "monitoring_present", "speaker_intended_warning")
    assert warning["audience_views"][0]["missing_basis_sources"] == ()
    assert warning["audience_views"][1]["missing_basis_sources"] == (
        "private", "record")
    assert result["speaker_intent"] == "unmeasured"
    assert result["delivery_cues"] == (("fable", "strained_laugh", True,
                                        "audio", "mic:1"),)
    assert result["literal_content_retained"] is True
    assert result["serving_authority"] is False


def test_cutoff_excludes_later_anchor_and_observation():
    episode, readings, evidence = _scene()
    result = assess_indirect_meaning(episode, readings, evidence, as_of=10)
    literal, warning = result["readings"]
    assert literal["status"] == "literal_candidate"
    assert warning["status"] == "basis_not_yet_available"
    assert warning["missing_basis_at_cutoff"] == ("record",)
    assert len(warning["analogy"]["source_anchored_relations"]) == 1
    assert ("retaliates", "council", "pilot") in warning["analogy"][
        "projected_unobserved_relations"]
    assert warning["observed_support"] == ()


def test_unseen_consequence_is_a_graph_prediction_not_an_observation():
    episode, readings, evidence = _scene()
    situation = replace(episode.situation_case, relations=(
        ("challenges", "pilot", "council"),
        ("disables", "council", "signal")))
    anchors = tuple(row for row in episode.relation_anchors
                    if row.relation != ("retaliates", "council", "pilot"))
    candidate = assess_indirect_meaning(
        replace(episode, situation_case=situation, relation_anchors=anchors),
        readings, (), as_of=10)["readings"][1]
    assert ("retaliates", "council", "pilot") in candidate["analogy"][
        "projected_unobserved_relations"]
    assert candidate["observed_support"] == ()
    assert "speaker_intended_warning" in candidate["unmeasured_predictions"]


def test_report_and_observation_are_not_interchangeable():
    episode, readings, evidence = _scene()
    reports = tuple(replace(row, kind="report", relation=None) for row in evidence)
    result = assess_indirect_meaning(episode, readings, reports, as_of=11)
    assert result["readings"][0]["status"] == "literal_candidate"
    assert result["readings"][1]["observed_support"] == ()
    assert result["readings"][1]["reported_fit"] == (
        ("retaliation_likely", True, "record"),
        ("monitoring_present", True, "public"))


def test_same_analogy_does_not_rescue_a_false_consequence():
    episode, readings, evidence = _scene()
    reassurance = replace(
        readings[1], identity="reassurance",
        interpretation=Interpretation("safe to proceed", believed_facts=(
            ("retaliation_likely", False),)),
        functions=(AudienceFunction("pilot", "reassurance"),
                   AudienceFunction("council", "ordinary story")))
    result = assess_indirect_meaning(episode, (readings[0], reassurance),
                                     evidence, as_of=11)
    assert result["readings"][0]["status"] == "literal_candidate"
    assert result["readings"][1]["analogy"]["source_anchored_relations"]
    assert result["readings"][1]["status"] == "observationally_contradicted"


def test_graph_match_without_cited_scene_does_not_count_as_source_grounded():
    episode, readings, evidence = _scene()
    result = assess_indirect_meaning(replace(episode, relation_anchors=()),
                                     readings, evidence, as_of=11)
    warning = result["readings"][1]
    assert warning["analogy"]["matched_relations"]
    assert warning["analogy"]["source_anchored_relations"] == ()
    assert warning["status"] == "source_unanchored_analogy"


def test_one_cited_relation_does_not_ground_the_whole_bridge():
    episode, readings, evidence = _scene()
    episode = replace(episode, relation_anchors=episode.relation_anchors[:1]
                      + episode.relation_anchors[3:4])
    warning = assess_indirect_meaning(
        episode, readings, evidence, as_of=11)["readings"][1]
    assert len(warning["analogy"]["source_anchored_relations"]) == 1
    assert warning["status"] == "partially_source_anchored_analogy"


def test_bad_role_bridge_is_rejected_by_counterfactual_mapping():
    episode, readings, evidence = _scene()
    wrong = replace(readings[1], entity_map=(
        ("musician", "council"), ("monarch", "pilot"),
        ("instrument", "signal")))
    result = assess_indirect_meaning(episode, (readings[0], wrong), evidence,
                                     as_of=11)
    assert result["readings"][1]["status"] == "structurally_unsupported"
    assert result["readings"][1]["analogy"]["matched_relations"] == ()


def test_graph_proposer_finds_a_bridge_without_baking_in_role_names():
    episode, _readings, _evidence = _scene()
    proposed = propose_role_bridges(episode, as_of=11)
    matching = [row for row in proposed["candidates"] if dict(row[
        "entity_map"]) == {"musician": "pilot", "monarch": "council",
                            "instrument": "signal"}]
    assert matching
    assert matching[0]["source_anchored_relations"] == 3
    assert proposed["shuffled_control"]["measurable"] is True
    assert proposed["speaker_intent"] == "unmeasured"
    early = propose_role_bridges(episode, as_of=10)
    assert any(row["status"] == "unanchored_graph_candidate"
               for row in early["candidates"])


def test_same_relation_can_have_different_public_and_private_functions():
    episode, readings, evidence = _scene()
    warning = replace(readings[1], functions=(
        AudienceFunction("pilot", "request to reconsider action"),
        AudienceFunction("council", "fictional entertainment")))
    result = assess_indirect_meaning(episode, (readings[0], warning), evidence,
                                     as_of=11)
    assert result["readings"][1]["audience_views"][0]["function_hypotheses"] == (
        "request to reconsider action",)
    assert result["readings"][1]["audience_views"][1]["function_hypotheses"] == (
        "fictional entertainment",)
    assert result["speaker_intent"] == "unmeasured"


def test_individual_and_collective_targets_remain_distinct_readings():
    episode, readings, evidence = _scene()
    texts = {row.source_id: row.text for row in episode.passages}
    texts["private"] += " The flight crew also planned to challenge the council."
    texts["record"] += " The council retaliated against the flight crew."
    events = {row.source_id: UsageEvent.from_text(
        row.source_id, "scene", texts[row.source_id], observed_at=row.observed_at,
        cues=row.cues, relations=row.relations)
        for row in (episode.utterance, *episode.context)}
    situation = replace(episode.situation_case,
                        entities=episode.situation_case.entities + (("crew", "group"),),
                        relations=episode.situation_case.relations + (
                            ("challenges", "crew", "council"),
                            ("retaliates", "council", "crew")))
    variant = replace(
        episode, utterance=events["fable"],
        context=(events["public"], events["private"], events["record"]),
        situation_case=situation,
        passages=tuple(SourcePassage(key, text) for key, text in texts.items()),
        relation_anchors=episode.relation_anchors + (
            RelationAnchor("situation", ("challenges", "crew", "council"),
                           "private", "flight crew also planned to challenge"),
            RelationAnchor("situation", ("retaliates", "council", "crew"),
                           "record", "retaliated against the flight crew")))
    individual = replace(readings[1], interpretation=Interpretation(
        "individual warning", believed_facts=(("collective_primary_target", False),)))
    collective = replace(readings[1], identity="collective",
                         interpretation=Interpretation(
                             "collective warning", believed_facts=(
                                 ("collective_primary_target", True),)),
                         entity_map=(("musician", "crew"),
                                     ("monarch", "council"),
                                     ("instrument", "signal")))
    result = assess_indirect_meaning(
        variant, (readings[0], individual, collective), evidence, as_of=11)
    assert len(result["readings"][1]["analogy"]["source_anchored_relations"]) == 3
    assert len(result["readings"][2]["analogy"]["source_anchored_relations"]) == 3
    assert ("collective_primary_target", True) in result[
        "discriminating_inquiries"]
    proposed = propose_role_bridges(variant, as_of=11, max_results=16)
    targets = {dict(row["entity_map"]).get("musician")
               for row in proposed["candidates"]}
    assert {"pilot", "crew"} <= targets


def test_followup_metacommentary_changes_reports_without_proving_intent():
    episode, readings, evidence = _scene()
    text = "The cartographer later said that some fables have one intended listener."
    remark = UsageEvent.from_text("remark", "scene", text, observed_at=12)
    episode = replace(episode, context=(*episode.context, remark),
                      passages=(*episode.passages, SourcePassage("remark", text)))
    warning = replace(readings[1], interpretation=Interpretation(
        "coded caution", believed_facts=(
            ("retaliation_likely", True), ("private_addressee", True))),
        basis_source_ids=(*readings[1].basis_source_ids, "remark"))
    report = PragmaticEvidence("private_addressee", True, "remark", "report", 12,
                               "some fables have one intended listener")
    before = assess_indirect_meaning(episode, (readings[0], warning),
                                     evidence + (report,), as_of=11)
    after = assess_indirect_meaning(episode, (readings[0], warning),
                                    evidence + (report,), as_of=12)
    assert before["readings"][1]["status"] == "basis_not_yet_available"
    assert after["readings"][1]["reported_fit"] == (
        ("private_addressee", True, "remark"),)
    assert after["readings"][1]["unmeasured_predictions"] == (
        "private_addressee",)
    assert after["speaker_intent"] == "unmeasured"


def test_source_excerpts_and_audience_knowledge_cannot_be_fabricated():
    episode, readings, evidence = _scene()
    with pytest.raises(ValueError, match="exact cited source excerpt"):
        replace(episode, relation_anchors=(RelationAnchor(
            "story", ("joins", "musician", "monarch"), "fable",
            "this sentence is not in the fable"),))
    with pytest.raises(ValueError, match="distinct sourced cases"):
        replace(episode, audiences=(Audience("pilot", "addressed", "private",
                                           ("private",),
                                           "pilot planned to challenge"),))
    with pytest.raises(ValueError, match="sourced propositions"):
        assess_indirect_meaning(episode, readings, evidence + (
            PragmaticEvidence("invented", True, "record", "observation", 11,
                              "no such excerpt"),), as_of=11)


def test_punctuation_change_cannot_rebind_a_retained_story_source():
    episode, _, _ = _scene()
    passage = next(row for row in episode.passages if row.source_id == "fable")
    tampered = replace(passage, text=passage.text.replace("fable:", "fable;"))
    assert UsageEvent.from_text("fable", "scene", tampered.text,
                                observed_at=10).terms == episode.utterance.terms
    with pytest.raises(ValueError, match="sourced cases"):
        replace(episode, passages=tuple(tampered if row.source_id == "fable" else row
                                        for row in episode.passages))


def test_legacy_usage_record_is_readable_but_not_exact_source_evidence():
    episode, _, _ = _scene()
    legacy = episode.utterance.to_dict()
    legacy.pop("source_text_sha256")
    restored = UsageEvent.from_dict(legacy)
    assert restored.terms == episode.utterance.terms
    with pytest.raises(ValueError, match="sourced cases"):
        replace(episode, utterance=restored)


def test_contradictory_observations_remain_disputed():
    episode, readings, evidence = _scene()
    other = UsageRelation("council", "retaliation_likely", "pilot", "scene",
                          "observation", "private", 11, polarity=False)
    private_text = next(row.text for row in episode.passages
                        if row.source_id == "private") + " A camera saw no retaliation."
    private = UsageEvent.from_text("private", "scene", private_text,
                                   observed_at=8, relations=(other,))
    episode = replace(episode,
                      context=(episode.context[0], private, episode.context[2]),
                      passages=tuple(SourcePassage(
                          row.source_id, private_text if row.source_id == "private"
                          else row.text) for row in episode.passages))
    contrary = PragmaticEvidence("retaliation_likely", False, "private",
                                 "observation", 11, "A camera saw no retaliation", other)
    result = assess_indirect_meaning(episode, readings, evidence + (contrary,),
                                     as_of=11)
    assert result["readings"][1]["status"] == "unresolved_conflicting_observations"
    assert result["readings"][1]["disputed_observations"] == (
        "retaliation_likely",)


def test_retained_runtime_path_and_later_correction(tmp_path, monkeypatch):
    episode, readings, evidence = _scene()
    service = SemanticDevelopment(registry=ConceptRegistry(),
                                  state_path=tmp_path / "semantic.json")
    monkeypatch.setattr(semantic_runtime, "get_semantic_development", lambda: service)
    with pytest.raises(ValueError, match="not retained unchanged"):
        semantic_runtime.assess_indirect_episode(episode, readings, evidence, as_of=11)
    for event in (episode.utterance, *episode.context):
        assert semantic_runtime.record_contextual_usage(event)
    result = semantic_runtime.assess_indirect_episode(
        episode, readings, evidence, as_of=11)
    assert result["readings"][1]["status"] == "candidate_not_intent_proof"
    assert semantic_runtime.propose_indirect_bridges(
        episode, as_of=11)["candidates"]
    feedback = MeaningFeedback("later-correction", "fable", "fable",
                               "coded caution", "user_correction", observed_at=12,
                               mode="indirect")
    assert semantic_runtime.record_attributed_meaning(feedback)
    assert service.meaning_feedback["later-correction"] == feedback
    with pytest.raises(ValueError, match="not retained unchanged"):
        semantic_runtime.assess_indirect_episode(
            replace(episode, utterance=replace(episode.utterance, speaker="other")),
            readings, evidence, as_of=11)
