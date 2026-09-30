"""The phenomena dispositions: how many the container resolves, and the claims registered about them.

Lifted whole out of `model_validation`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _phenomena_reachable() -> int:
    """How many of the fourteen dispositions the live container resolves."""
    from .model_validation import (
        logger,
    )


    try:
        from core.container import get_container
        from core.phenomena_wiring import SERVICE_NAMES

        container = get_container()
        found = 0
        for name in SERVICE_NAMES:
            # A disposition the container does not hold resolves to None and is
            # not counted. Asked without a default it raised, and the claim
            # errored instead of reporting a number.
            try:
                if container.get(name, None) is not None:
                    found += 1
            except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                continue
        return found
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Phenomena services unreachable, reporting none wired: %s", exc)
        return 0


def _install_phenomena_claims(suite: Any) -> None:
    """Fourteen dispositions, and the five statements about them worth binding.

    Only the reachability claim is measured live; it reads the running
    container and nothing else. The other four run their mechanism against a
    constructed case with a known answer, which establishes the mechanism and
    not any behaviour of the live system, and they say so.
    """
    from .model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        _arbitration_abstains_without_evidence,
        _care_floor_survives_an_overwhelming_need,
        _declaring_an_identity_moves_its_coherence,
        _phenomena_reachable,
        _pooling_signal_yields_no_type,
        boolean_score,
        threshold_score,
    )


    for name, description, capability, observation, predict, score, owner in (
        (
            "phenomena_dispositions_are_reachable",
            "every disposition resolves from the live container rather than "
            "existing as a module nothing can reach",
            "phenomena_wiring",
            Observation(
                name="dispositions_resolving",
                value=14,
                source="core/phenomena_wiring.py and tests/test_phenomena_wiring.py",
                units="services",
            ),
            lambda _m: _phenomena_reachable(),
            lambda p, o: threshold_score(
                float(p), float(o.value), direction="at_least", units=" services"
            ),
            "core/phenomena_wiring.py",
        ),
        (
            "declaring_an_identity_does_not_establish_it",
            "coherence is computed from the enactment record, and a label "
            "recorded fifty times does not move it",
            "constitutive_identity",
            Observation(
                name="coherence_moved_by_declaration",
                value=False,
                source="core/identity/constitutive_identity.py and "
                "tests/test_phenomena_mechanisms.py",
            ),
            lambda _m: _declaring_an_identity_moves_its_coherence(),
            lambda p, o: boolean_score(
                bool(p), expected=bool(o.value), subject="declaration moved coherence"
            ),
            "core/identity/constitutive_identity.py",
        ),
        (
            "the_care_floor_is_not_for_sale",
            "the reserve held back for the carer survives a need twelve orders "
            "of magnitude above the budget",
            "care_allocation",
            Observation(
                name="floor_held",
                value=True,
                source="core/ethics/care_allocation.py and "
                "tests/test_phenomena_mechanisms.py",
            ),
            lambda _m: _care_floor_survives_an_overwhelming_need(),
            lambda p, o: boolean_score(
                bool(p), expected=bool(o.value), subject="floor held against any need"
            ),
            "core/ethics/care_allocation.py",
        ),
        (
            "a_free_signal_supports_no_inference",
            "a presentation whose effort costs every sender the same returns no "
            "implied type rather than a plausible one",
            "costly_signaling",
            Observation(
                name="pooling_signal_declines_to_infer",
                value=True,
                source="core/social/costly_signaling.py and "
                "tests/test_phenomena_mechanisms.py",
            ),
            lambda _m: _pooling_signal_yields_no_type(),
            lambda p, o: boolean_score(
                bool(p), expected=bool(o.value), subject="pooling signal declined to infer"
            ),
            "core/social/costly_signaling.py",
        ),
        (
            "arbitration_abstains_where_it_has_no_calibration",
            "with no resolved outcome in a domain the arbiter returns no answer "
            "and equal weights, rather than falling back on either channel",
            "dual_process_arbitration",
            Observation(
                name="abstains_without_evidence",
                value=True,
                source="core/affect/dual_process_arbiter.py and "
                "tests/test_phenomena_mechanisms.py",
            ),
            lambda _m: _arbitration_abstains_without_evidence(),
            lambda p, o: boolean_score(
                bool(p), expected=bool(o.value), subject="arbiter abstained"
            ),
            "core/affect/dual_process_arbiter.py",
        ),
    ):
        suite.add_test(
            ValidationTest(
                name=name, description=description, required_capability=capability,
                observation=observation, predict=predict, score=score, owner=owner,
            )
        )

    suite.add_claim(
        Claim(
            statement=(
                "Fourteen dispositions built as separate mechanisms are each "
                "reachable from the running container."
            ),
            test="phenomena_dispositions_are_reachable",
            owner="core/phenomena_wiring.py",
            asserted_in="docs/PHENOMENA_AS_MECHANISMS.md",
            evidence=Evidence.MEASURED_LIVE,
            live_channels=("empathy.autonomy", "care.depleted"),
            evidence_note=(
                "reads the live container. It establishes that each disposition "
                "resolves and reports, and nothing about whether anything in the "
                "running system drives them"
            ),
        )
    )
    for statement, test_name, owner, note in (
        (
            "An identity held as the coherence of its practices cannot be "
            "established by declaring it.",
            "declaring_an_identity_does_not_establish_it",
            "core/identity/constitutive_identity.py",
            "run against a constructed identity with two enactments. It "
            "establishes the one-way direction of the mechanism, not that any "
            "identity in the live system is held this way",
        ),
        (
            "The reserve held back for the carer is a constraint on the "
            "allocation rather than a term in it, so no level of need elsewhere "
            "can buy it.",
            "the_care_floor_is_not_for_sale",
            "core/ethics/care_allocation.py",
            "run against constructed needs up to 1e12 against a budget of 10. "
            "It establishes that the floor is in the feasible set, not that any "
            "live care allocation has gone through this allocator",
        ),
        (
            "A presentation whose effort is the same for every sender yields no "
            "inference about the sender.",
            "a_free_signal_supports_no_inference",
            "core/social/costly_signaling.py",
            "run against a constructed channel with the cost slope set to zero. "
            "It establishes the refusal, not that any live presentation has "
            "been read through this channel",
        ),
        (
            "Arbitration between an affective and a deliberate judgement "
            "abstains in a domain where neither has shown skill.",
            "arbitration_abstains_where_it_has_no_calibration",
            "core/affect/dual_process_arbiter.py",
            "run against a fresh arbiter with no resolved outcomes. It "
            "establishes the abstention, not that any live decision has been "
            "arbitrated this way",
        ),
    ):
        suite.add_claim(
            Claim(
                statement=statement, test=test_name, owner=owner,
                asserted_in="docs/PHENOMENA_AS_MECHANISMS.md",
                evidence=Evidence.MEASURED_SYNTHETIC,
                evidence_note=note,
            )
        )


