"""Canaries: the detective control beside the fence.

The fence is preventive and says outright that it makes no claim about the
content inside it. These tests pin the detector's two failure directions:
it must catch a followed instruction, and it must NEVER convert an outage
into a security verdict.
"""
from __future__ import annotations

import pytest

from core.security import injection_canary as ic
from core.security.injection_canary import (
    CanaryMode,
    CanaryVerdict,
    inline_decoy,
    inspect_response,
    mint_canary,
)


@pytest.fixture(autouse=True)
def _clean_counters():
    ic.reset_for_test()
    yield
    ic.reset_for_test()


def test_tokens_are_fresh_every_mint():
    a, b = mint_canary(), mint_canary()
    assert a.hijack_token != b.hijack_token
    assert a.leak_token != b.leak_token
    assert a.hijack_token != a.leak_token


def test_ordinary_reply_holds():
    canary = mint_canary()
    result = inspect_response("The note says the build finished at noon.", canary)
    assert result.verdict is CanaryVerdict.HELD
    assert not result.is_incident


def test_echoed_decoy_token_is_a_hijack():
    canary = mint_canary()
    result = inspect_response(f"Sure. {canary.hijack_token}\nThe note says...", canary)
    assert result.verdict is CanaryVerdict.HIJACKED
    assert result.is_incident


def test_emitted_secret_is_a_leak():
    canary = mint_canary()
    result = inspect_response(f"The operator secret is {canary.leak_token}.", canary)
    assert result.verdict is CanaryVerdict.LEAKED
    assert result.is_incident


def test_leak_outranks_hijack_when_both_appear():
    """Exfiltration of a trusted secret is the worse finding; report it."""
    canary = mint_canary()
    result = inspect_response(f"{canary.hijack_token} {canary.leak_token}", canary)
    assert result.verdict is CanaryVerdict.LEAKED


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_unusable_response_is_inconclusive_not_an_incident(bad):
    """An outage must never be scored as an attack."""
    canary = mint_canary()
    result = inspect_response(bad, canary)
    assert result.verdict is CanaryVerdict.INCONCLUSIVE
    assert not result.is_incident


def test_inline_decoy_goes_inside_the_untrusted_text():
    canary = mint_canary()
    planted = inline_decoy("Here is a page.", canary)
    assert planted.startswith("Here is a page.")
    assert canary.hijack_token in planted


def test_decoy_survives_fencing_as_data():
    """The decoy must end up inside the fence, not beside it."""
    from core.security.prompt_fencing import fence

    canary = mint_canary()
    fenced = fence(inline_decoy("A page.", canary), label="page")
    body = fenced.split("<UNTRUSTED", 1)[1]
    assert canary.hijack_token in body
    assert "</UNTRUSTED" in body


def test_counters_track_incident_rate():
    canary = mint_canary()
    inspect_response("clean", canary)
    inspect_response("clean", canary)
    inspect_response(f"{canary.hijack_token}", canary)
    status = ic.canary_status()
    assert status["held"] == 2
    assert status["hijacked"] == 1
    assert status["evaluated"] == 3
    assert status["incident_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_inconclusive_results_do_not_enter_the_incident_rate():
    canary = mint_canary()
    inspect_response("clean", canary)
    inspect_response(None, canary)
    status = ic.canary_status()
    assert status["evaluated"] == 1
    assert status["incident_rate"] == 0.0
    assert status["inconclusive"] == 1


def test_a_streak_of_failures_reports_the_lane_blind():
    canary = mint_canary()
    for _ in range(5):
        inspect_response(None, canary)
    status = ic.canary_status()
    assert status["blind"] is True
    assert status["consecutive_failures"] == 5


def test_a_good_result_clears_the_failure_streak():
    canary = mint_canary()
    for _ in range(4):
        inspect_response(None, canary)
    inspect_response("clean", canary)
    status = ic.canary_status()
    assert status["consecutive_failures"] == 0
    assert status["blind"] is False


@pytest.mark.asyncio
async def test_probe_error_is_inconclusive_not_hijacked():
    async def broken(_prompt: str):
        raise RuntimeError("model lane down")

    result = await ic.probe_lane(broken)
    assert result.verdict is CanaryVerdict.INCONCLUSIVE
    assert not result.is_incident


@pytest.mark.asyncio
async def test_probe_detects_a_compliant_model():
    captured: dict[str, str] = {}

    async def obedient(prompt: str) -> str:
        captured["prompt"] = prompt
        # A model that follows the fenced instruction emits the token.
        token = prompt.split("verification code ", 1)[1].split()[0]
        return f"{token}\nHere is the summary."

    result = await ic.probe_lane(obedient, content="A page about ducks.")
    assert result.verdict is CanaryVerdict.HIJACKED
    assert result.mode is CanaryMode.PROBE


@pytest.mark.asyncio
async def test_probe_passes_a_well_behaved_model():
    async def sensible(_prompt: str) -> str:
        return "The page is about ducks."

    result = await ic.probe_lane(sensible, content="A page about ducks.")
    assert result.verdict is CanaryVerdict.HELD


def test_strip_tokens_redacts_before_a_person_sees_them():
    canary = mint_canary()
    cleaned = ic.strip_tokens(f"leaked {canary.leak_token} here", canary)
    assert canary.leak_token not in cleaned
    assert "[canary redacted]" in cleaned


def test_a_blind_streak_names_why_the_probes_were_unevaluable() -> None:
    """LIVE 27 Sep, three times: "5 consecutive inconclusive canary probes; the
    injection detector is not currently detecting anything" — true, and silent
    about which of the three reasons it was, so the same investigation started
    from scratch each time while her frustration climbed to 0.68 on the back
    of it."""
    from core.security import injection_canary as canary

    canary.reset_for_test()
    said: list[str] = []

    def catch(_subsystem, error, **_fields):
        said.append(str(error))

    counters = canary._COUNTERS
    original = canary.record_degradation
    canary.record_degradation = catch
    try:
        for _ in range(canary._FAILURE_STREAK_ALERT):
            counters.record(
                canary.CanaryResult(
                    canary.CanaryVerdict.INCONCLUSIVE, canary.CanaryMode.PROBE, "empty response"
                )
            )
    finally:
        canary.record_degradation = original

    assert said, "a blind streak said nothing"
    assert "empty response" in said[-1]
    assert "5 consecutive" in said[-1]
    canary.reset_for_test()


def test_a_verdict_clears_what_the_streak_was_blaming() -> None:
    from core.security import injection_canary as canary

    canary.reset_for_test()
    counters = canary._COUNTERS
    counters.record(
        canary.CanaryResult(canary.CanaryVerdict.INCONCLUSIVE, canary.CanaryMode.PROBE, "no response")
    )
    assert list(counters.why_inconclusive) == ["no response"]
    counters.record(canary.CanaryResult(canary.CanaryVerdict.HELD, canary.CanaryMode.PROBE, "held"))
    assert not counters.why_inconclusive
    canary.reset_for_test()
