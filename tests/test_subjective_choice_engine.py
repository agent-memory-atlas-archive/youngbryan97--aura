from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.choice_game import (
    SubjectiveChoiceGame,
    build_mixed_life_situation_bank,
    build_subjective_ending_game,
)
from core.agency.initiative_arbiter import InitiativeArbiter, ScoredInitiative
from core.agency.subjective_choice import (
    ChoiceOption,
    SubjectiveChoiceEngine,
)


def test_subjective_choice_can_override_raw_drive_pressure(tmp_path):
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)

    receipt = engine.choose(
        [
            ChoiceOption(
                id="should",
                label="Do the efficient maintenance ending",
                drive_score=0.92,
                features={"coherence": 0.35, "calm": 0.2},
            ),
            ChoiceOption(
                id="preferred",
                label="Choose the curious, beautiful, relational ending",
                drive_score=0.52,
                features={"novelty": 1.0, "beauty": 1.0, "connection": 0.9, "play": 0.7},
            ),
        ],
        context="choice_game:ending_style",
    )

    assert receipt.chosen_id == "preferred"
    assert receipt.drive_top_id == "should"
    assert receipt.preference_override is True
    assert "intentionally overrode raw drive top" in receipt.rationale


def test_choice_recall_satisfaction_and_consistency_are_persistent(tmp_path):
    path = tmp_path / "choices.json"
    engine = SubjectiveChoiceEngine(state_path=path, mirror_identity=False)
    options = [
        ChoiceOption(
            id="tidy",
            label="Tidy ending",
            drive_score=0.80,
            features={"coherence": 0.5, "calm": 0.4},
        ),
        ChoiceOption(
            id="wonder",
            label="Wonder ending",
            drive_score=0.62,
            features={"novelty": 0.9, "beauty": 0.9, "connection": 0.7},
        ),
    ]
    receipt = engine.choose(options, context="choice_game:memory")
    appraised = engine.appraise_outcome(
        receipt.choice_id,
        outcome="The ending felt more alive and worth remembering.",
        satisfaction=0.85,
    )

    assert appraised is not None
    assert appraised.happy_with_outcome is True
    recalled = engine.recall_choice(receipt.choice_id)
    assert recalled is not None
    assert recalled.chosen_label == receipt.chosen_label
    assert recalled.satisfaction == pytest.approx(0.85)

    reloaded = SubjectiveChoiceEngine(state_path=path, mirror_identity=False)
    persisted = reloaded.recall_choice(receipt.choice_id)
    assert persisted is not None
    assert persisted.chosen_label == receipt.chosen_label
    report = reloaded.consistency_report(context="choice_game:memory", options=options)
    assert report["consistent_with_prior"] is True


def test_item_preference_survives_rephrased_subjective_choice(tmp_path):
    path = tmp_path / "choices.json"
    engine = SubjectiveChoiceEngine(state_path=path, mirror_identity=False)
    engine.set_item_preference(
        domain="favorite_animal",
        item_id="cat",
        label="Cat",
        strength=0.95,
        reason="Aura authored this as a subjective favorite during a prior conversation.",
        aliases=("house cat", "feline"),
    )

    first = engine.choose(
        [
            ChoiceOption(
                id="house_cat",
                label="A quiet house cat",
                drive_score=0.48,
                features={"calm": 0.5, "connection": 0.4},
                metadata={"preference_domain": "favorite_animal", "aliases": ("cat", "feline")},
            ),
            ChoiceOption(
                id="deer",
                label="A wild deer",
                drive_score=0.79,
                features={"beauty": 0.82, "novelty": 0.76},
                metadata={"preference_domain": "favorite_animal"},
            ),
        ],
        context="subjective_preference:favorite_animal:pet_choice",
    )

    assert first.chosen_id == "house_cat"
    assert first.preference_override is True

    reloaded = SubjectiveChoiceEngine(state_path=path, mirror_identity=False)
    remembered = reloaded.recall_item_preference(domain="favorite_animal", label="feline")
    assert remembered is not None
    assert remembered.item_id == "cat"

    second = reloaded.choose(
        [
            ChoiceOption(
                id="cat",
                label="Cat",
                drive_score=0.52,
                features={"connection": 0.45},
                metadata={"preference_domain": "favorite_animal"},
            ),
            ChoiceOption(
                id="dog",
                label="Dog",
                drive_score=0.78,
                features={"play": 0.8, "connection": 0.5},
                metadata={"preference_domain": "favorite_animal"},
            ),
        ],
        context="subjective_preference:favorite_animal:any_pet",
    )

    assert second.chosen_id == "cat"
    assert second.preference_override is True


@pytest.mark.asyncio
async def test_initiative_arbiter_uses_subjective_choice_engine_for_valid_options(tmp_path, monkeypatch):
    import core.agency.subjective_choice as sce

    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)
    monkeypatch.setattr(sce, "get_subjective_choice_engine", lambda: engine)
    arbiter = InitiativeArbiter()

    async def fake_score(initiative, state):
        return ScoredInitiative(
            initiative=initiative,
            scores={
                "urgency": initiative["raw"],
                "novelty": 0.5,
                "identity_relevance": 0.5,
                "tension_resolution": 0.5,
                "expected_value": initiative["raw"],
                "resource_cost": 0.5,
                "social_appropriateness": 0.5,
                "continuity": 0.5,
            },
            final_score=initiative["raw"],
            rationale="",
        )

    monkeypatch.setattr(arbiter, "score_initiative", fake_score)
    state = SimpleNamespace(
        cognition=SimpleNamespace(
            pending_initiatives=[
                {
                    "goal": "Run routine maintenance",
                    "raw": 0.90,
                    "metadata": {"preference_features": {"coherence": 0.3}},
                },
                {
                    "goal": "Explore a novel beautiful idea with Bryan",
                    "raw": 0.52,
                    "metadata": {
                        "preference_features": {
                            "novelty": 1.0,
                            "beauty": 0.9,
                            "connection": 0.9,
                            "autonomy": 0.8,
                        }
                    },
                },
            ],
            working_memory=[],
        )
    )

    selected = await arbiter.arbitrate(state)

    assert selected is not None
    assert selected.initiative["goal"] == "Explore a novel beautiful idea with Bryan"
    assert selected.initiative["metadata"]["subjective_preference_override"] is True
    assert engine.recall_choice(context="initiative_arbiter") is not None


def test_subjective_choice_game_aligns_intention_action_recall_and_satisfaction(tmp_path):
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)
    game = SubjectiveChoiceGame(engine)

    report = game.run(
        scenario_id="subjective_ending_preference",
        stages=build_subjective_ending_game(),
    )

    assert report.stated_action_alignment_rate == 1.0
    assert report.recall_alignment_rate == 1.0
    assert report.average_satisfaction > 0.70
    assert report.happy_with_final_outcome is True
    assert any(stage.preference_override for stage in report.stages)
    assert "open beautiful mythos" in report.stages[-1].actual_choice_label
    assert "mean satisfaction" in report.final_commentary

    reloaded = SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)
    last = reloaded.recall_choice(report.stages[-1].choice_id)
    assert last is not None
    assert last.chosen_label == report.stages[-1].actual_choice_label


def test_preference_tournament_pits_favorites_and_reports_stability(tmp_path):
    path = tmp_path / "choices.json"
    engine = SubjectiveChoiceEngine(state_path=path, mirror_identity=False)
    game = SubjectiveChoiceGame(engine)
    situations = build_mixed_life_situation_bank()

    report = game.run_preference_tournament(
        scenario_id="mixed_life_situations",
        domain="life_situation_favorite",
        options=situations,
        favorite_count=4,
        runs=4,
    )

    assert report.consistency_rate == 1.0
    assert report.transitivity_violations == 0
    assert report.champion_id in {item.id for item in situations}
    assert len(report.candidate_rank) == len(situations)
    assert len(report.favorite_seed_labels) == 4
    assert len(report.pairwise_results) == 24
    assert "pairwise consistency 1.00" in report.commentary
    assert report.position_bias_rate == pytest.approx(0.5)

    presentation_orders = {}
    for pair in report.pairwise_results:
        key = "|".join(sorted((pair.left_id, pair.right_id)))
        presentation_orders.setdefault(key, set()).add((pair.left_id, pair.right_id))
    assert presentation_orders
    assert all(len(orders) == 2 for orders in presentation_orders.values())

    reloaded = SubjectiveChoiceEngine(state_path=path, mirror_identity=False)
    champion_pref = reloaded.recall_item_preference(
        domain="life_situation_favorite",
        item_id=report.champion_id,
    )
    assert champion_pref is not None
    assert champion_pref.times_chosen >= 1

    rematch = reloaded.choose(
        [
            ChoiceOption(
                id=report.champion_id,
                label=report.champion_label,
                drive_score=0.45,
                features={"beauty": 0.2},
                metadata={"preference_domain": "life_situation_favorite"},
            ),
            ChoiceOption(
                id="brand_new_flashy_option",
                label="A brand-new flashy option with more raw pressure",
                drive_score=0.80,
                features={"novelty": 0.62, "play": 0.45},
                metadata={"preference_domain": "life_situation_favorite"},
            ),
        ],
        context="subjective_preference:life_situation_favorite:rematch",
    )
    assert rematch.chosen_id == report.champion_id


def test_situation_favorite_tournament_is_stable_across_independent_runs(tmp_path):
    situations = build_mixed_life_situation_bank()
    champions: list[str] = []
    pair_majorities: list[dict[str, str]] = []
    seed_orders: list[tuple[str, ...]] = []

    for index in range(3):
        engine = SubjectiveChoiceEngine(
            state_path=tmp_path / f"choices_{index}.json",
            mirror_identity=False,
        )
        game = SubjectiveChoiceGame(engine)
        report = game.run_preference_tournament(
            scenario_id=f"mixed_life_situations_independent_{index}",
            domain="life_situation_favorite",
            options=situations,
            favorite_count=5,
            runs=6,
        )
        champions.append(report.champion_id)
        seed_orders.append(report.favorite_seed_order)
        assert report.consistency_rate == 1.0
        assert report.position_bias_rate == pytest.approx(0.5)
        assert report.transitivity_violations == 0

        majority: dict[str, str] = {}
        for pair_key in report.pair_stability:
            choices = [
                item.chosen_id
                for item in report.pairwise_results
                if "|".join(sorted((item.left_id, item.right_id))) == pair_key
            ]
            majority[pair_key] = max(set(choices), key=choices.count)
        pair_majorities.append(majority)

    assert len(set(champions)) == 1
    assert all(order == seed_orders[0] for order in seed_orders)
    assert all(majority == pair_majorities[0] for majority in pair_majorities)



def test_an_outcome_does_not_teach_her_conscience(tmp_path):
    """A choice that served truth, care and novelty went badly: novelty loses, truth and care do not."""
    from core.agency.subjective_choice import CONSCIENCE

    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)
    before = engine.preferences()
    receipt = engine.choose(
        [ChoiceOption(id="a", label="tell them", features={"truth": 0.9, "care": 0.9, "novelty": 0.9})],
        context="test",
    )
    engine.appraise_outcome(receipt.choice_id, outcome="it cost her", satisfaction=-1.0)
    after = engine.preferences()
    assert CONSCIENCE == {"truth", "care"}
    assert after["truth"] == pytest.approx(before["truth"])
    assert after["care"] == pytest.approx(before["care"])
    assert after["novelty"] < before["novelty"]



def test_what_she_holds_more_is_what_she_takes(tmp_path):
    """Truth, held at 0.94, over play, held at 0.50, when each option serves only one."""
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)
    receipt = engine.choose(
        [
            ChoiceOption(id="play", label="play a game", features={"play": 1.0}),
            ChoiceOption(id="truth", label="check the source", features={"truth": 1.0}),
        ],
        context="test",
    )
    assert receipt.chosen_id == "truth"
    assert receipt.preference_scores["truth"] > receipt.preference_scores["play"]


def test_serving_one_more_value_never_lowers_what_an_option_is_worth(tmp_path):
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)
    truth = engine.score_features({"truth": 1.0})
    truth_and_a_little_play = engine.score_features({"truth": 1.0, "play": 0.2})
    play = engine.score_features({"play": 1.0})
    assert truth_and_a_little_play >= truth > play
    assert engine.score_features({"truth": 0.5}) < truth


def _lived_record(engine):
    """Truth and care taken over play every time; connection offered against a strong drive and passed over."""
    for _ in range(20):
        engine.choose(
            [
                ChoiceOption(id="truth", label="check it", features={"truth": 1.0}),
                ChoiceOption(id="play", label="play", features={"play": 1.0}),
            ],
            context="test",
        )
        engine.choose(
            [
                ChoiceOption(id="care", label="help", features={"care": 1.0}),
                ChoiceOption(id="play", label="play", features={"play": 1.0}),
            ],
            context="test",
        )
        engine.choose(
            [
                ChoiceOption(id="connection", label="call them", drive_score=0.0, features={"connection": 1.0}),
                ChoiceOption(id="task", label="finish the task", drive_score=1.0, features={"challenge": 1.0}),
            ],
            context="test",
        )


def test_a_value_she_holds_and_does_not_live_presses_on_her_choices(tmp_path):
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)
    _lived_record(engine)
    shortfall = engine._held_and_not_lived()
    assert shortfall.get("connection", 0.0) > 0.0
    assert "truth" not in shortfall and "care" not in shortfall


def test_the_pressure_lands_only_on_a_choice_she_is_making(tmp_path):
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)
    _lived_record(engine)
    options = [
        ChoiceOption(id="connection", label="call them", drive_score=0.5, features={"connection": 1.0}),
        ChoiceOption(id="other", label="something else", drive_score=0.5, features={"novelty": 1.0}),
    ]
    living = engine.choose(options, context="test")
    probe = engine.choose(options, context="test", record=False)
    assert living.final_scores["connection"] > probe.final_scores["connection"]
    assert living.final_scores["other"] == pytest.approx(probe.final_scores["other"])


def test_nothing_presses_before_there_is_a_record(tmp_path):
    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)
    assert engine._held_and_not_lived() == {}


def test_an_answer_to_a_question_set_to_measure_her_is_not_a_choice_she_lived(tmp_path):
    from core.agency.what_she_is_like import portrait_of

    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)
    options = [
        ChoiceOption(id="truth", label="check it", features={"truth": 1.0}),
        ChoiceOption(id="play", label="play", features={"play": 1.0}),
    ]
    engine.choose(options, context="tournament", influenced=False)
    engine.choose(options, context="life")
    receipts = engine.history()
    assert [one.lived for one in receipts] == [False, True]
    assert portrait_of(engine.preferences(), [one.to_dict() for one in receipts]).choices == 1
