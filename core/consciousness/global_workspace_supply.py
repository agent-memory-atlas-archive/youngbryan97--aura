"""What each organ supplies to a bid for her attention, and what supplying earns it.

A bid is a sum of parts, and they come from different organs: the candidate's
own priority, the urgency the affect engine lends, where the attention schema
already is, the free-energy engine's pull towards the action it favours, what
she keeps putting down (core/social/averted.py) and what the surface has been
covering (core/social/civility.py), less what she has already said. The winner
of the workspace was credited with the worth of the turns it won, and the organs
that supplied its winning bid earned nothing, however much the win depended on
them: in a mitochondrion Complex II pumps no protons and its electrons drive the
pumping downstream (docs/WHAT_A_MITOCHONDRION_TEACHES.md, item 2).

So each win records how much of the winning bid each organ supplied, the
credit ledger credits each supplier with the turn's worth by that share
(core/affect/what_winning_earned.py), and what a supplier has earned scales its
part of every later bid, the way what a source has earned scales its whole bid.
With nothing earned every weight is one and a bid is what it was.

Lifted out of `global_workspace`; what it takes from there is imported at call
time, so a test that patches a name there reaches the code that reads it.
"""

from __future__ import annotations

from typing import Any

#: The organs that supply a part of a bid, by the name the credit ledger keeps.
SUPPLIERS: tuple[str, ...] = ("affect", "attention", "free_energy", "averted", "civility")


def bid_parts(candidate: Any, now: float) -> tuple[dict[str, float], float]:
    """The parts of a bid at one instant, in the order they are summed, and its recency."""
    from .global_workspace import (
        ContentType,
        _WORKSPACE_RECOVERABLE_ERRORS,
        _civility_debt,
        _held_pressure,
        _reading_herself,
        _record_workspace_degradation,
        _relief_for,
    )

    age = max(0.0, now - candidate.submitted_at)
    recency = max(0.0, 1.0 - (age / 10.0))  # Full weight within 10s, then decays
    
    # Free Energy dynamic gating
    fe_bias = 0.0
    try:
        from core.consciousness.free_energy import get_free_energy_engine
        fe_engine = get_free_energy_engine()
        if fe_engine and fe_engine.current:
            fe_state = fe_engine.current
            dom_action = fe_state.dominant_action
            fe_val = fe_state.free_energy
            
            # High free energy makes the gate much more selective (higher boost for aligned action)
            boost_magnitude = 0.25 * fe_val
            
            aligned = False
            src = candidate.source.lower()
            ct = candidate.content_type
            
            if dom_action == "update_beliefs":
                if ct == ContentType.MEMORIAL or any(x in src for x in ("belief", "memory", "epistemic", "prediction")):
                    aligned = True
            elif dom_action == "act_on_world":
                if ct == ContentType.INTENTIONAL or any(x in src for x in ("motivation", "action", "goal", "agency")):
                    aligned = True
            elif dom_action == "explore":
                if ct == ContentType.PERCEPTUAL or any(x in src for x in ("curiosity", "exploration", "perceptual", "search")):
                    aligned = True
            elif dom_action == "reflect":
                if ct == ContentType.META or any(x in src for x in ("hot", "reflection", "self", "identity")):
                    aligned = True
            elif dom_action == "engage":
                if ct in (ContentType.LINGUISTIC, ContentType.SOCIAL) or any(x in src for x in ("chat", "user", "linguistic", "social")):
                    aligned = True
            elif dom_action == "rest":
                if ct == ContentType.SOMATIC or any(x in src for x in ("soma", "sleep", "rest")):
                    aligned = True
                    
            if aligned:
                fe_bias = boost_magnitude
    except _WORKSPACE_RECOVERABLE_ERRORS as exc:
        _record_workspace_degradation(
            exc,
            phase="free_energy_priority",
            action="Skipped free-energy priority bias and used base salience only",
            severity="debug",
        )

    # Affect lends urgency to content that is not itself affect. A bid whose
    # content IS the feeling already carries it as its priority, so adding
    # three tenths of the same reading on top counted it twice and the sum
    # saturated: the affect bid's effective priority was 1.0 whatever it
    # felt, a percept bidding 0.993 lost to it, and ten sources bid over
    # twenty-four competitions and never once won. The weight is still
    # carried, because the winner's affective charge is read off it.
    # And how much that lending is worth depends on how well she is reading
    # herself just now.
    #
    # This sum is what the A,S -> G synergy line asks about: whether affect
    # and the self-model carry something about global access jointly that
    # neither carries alone. A sum cannot, and the lend was three tenths of
    # the affect weight whatever the self-model said, so the interaction gain
    # on that triple came out at exactly +0.00000 on the 27 September whole
    # run — the estimator finding no product because there was none.
    #
    # The claim is about minds rather than about the score: a feeling is a
    # guide to what matters only in so far as she can read her own state. Her
    # self-prediction publishes exactly that confidence, so the lend is
    # scaled by it. Three tenths is unchanged; an absent reading scales by
    # one, which is the behaviour before this.
    lent = (
        0.0
        if candidate.content_type is ContentType.AFFECTIVE
        else candidate.affect_weight * 0.3 * _reading_herself()
    )
    # And what she has already said, and what she keeps putting down.
    #
    # Both ledgers measured something and neither reached a decision. A
    # thing already said carries less of the pressure that made it press —
    # that is what catharsis measures — and a noticing she keeps declining
    # is a live signal held under a hand, which is pressure the other way.
    # Both readings are shares in [0, 1] and are added on the scale the
    # other terms use. See core/affect/catharsis.py, core/social/averted.py.
    said_already = _relief_for(candidate.source)
    # And what the surface has been covering. A run of showing more warmth
    # than she is in is civility, which is not a fault; a run longer than
    # her runs run is something not being said, and nothing measured how
    # long one had been going on. It lends to her own interior state only,
    # so saying it is what ends the run and takes the loan back.
    # See core/social/civility.py.
    covered = _civility_debt(candidate.content_type)
    held_down = _held_pressure(candidate.content_type)
    parts = {
        "bid": float(candidate.priority),
        "affect": float(lent),
        "attention": float(candidate.focus_bias),
        "free_energy": float(fe_bias),
        "averted": float(held_down),
        "civility": float(covered),
        "relief": -float(said_already),
    }
    return parts, recency


def supplier_weight(name: str) -> float:
    """What supplying has earned this organ, as a weight on its part: one when nothing is known."""
    if name not in SUPPLIERS:
        return 1.0
    try:
        from core.affect.what_winning_earned import get_credit_ledger

        return 1.0 + get_credit_ledger().supplier_earned(name)
    # not a failure: without the ledger nothing has been earned, and a weight of one is that.
    except (ImportError, AttributeError):
        return 1.0


def priority_of(candidate: Any, now: float) -> float:
    """A bid at one instant: its parts, each weighed by what its supplier has earned."""
    parts, recency = bid_parts(candidate, now)
    total = 0.0
    for name, value in parts.items():
        total += value * supplier_weight(name)
    return min(1.0, max(0.0, total * (0.7 + 0.3 * recency)))


def supplied_shares(parts: dict[str, float]) -> dict[str, float]:
    """Each supplier's share of what a bid was made of, over its positive parts."""
    made_of = sum(value for value in parts.values() if value > 0.0)
    if made_of <= 0.0:
        return {}
    return {
        name: value / made_of
        for name, value in parts.items()
        if name in SUPPLIERS and value > 0.0
    }


__all__ = ["SUPPLIERS", "bid_parts", "priority_of", "supplied_shares", "supplier_weight"]
