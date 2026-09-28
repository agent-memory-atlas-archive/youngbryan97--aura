"""Six things can happen when an answer is checked, and only one is correctness.

CTX2-AMP-001 names them: verified, contradicted, unsupported, not_applicable,
verifier_failed and timed_out, and says no empty verifier set, exception,
parse failure or score default may be treated as correctness. The types had
four states and the registry reached three of them:

* an engine that crashed came back ``ok=True, checked=False`` with the flag
  this type defines for exactly that case left unset, so a crash read the
  same as an answer with nothing in it to check;
* an engine that raised ``TimeoutError`` or any ``OSError`` was not caught at
  all, and took the whole gather, and the amplification with it;
* a task no engine covers could not be told from a task whose answer made no
  checkable claim, because the always-on logic engine is added to every set.

And the tier escalation adopted a stronger model's candidate on ``ok`` alone,
which a crashed verifier satisfies.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.verifiers.base import VerificationResult, combine_results
from core.brain.verifiers.registry import VerifierRegistry

SIX = {"verified", "contradicted", "unsupported", "not_applicable", "verifier_failed", "timed_out"}


def _r(**kw) -> VerificationResult:
    base = {"domain": "t", "ok": True, "checked": False, "engine": "e"}
    base.update(kw)
    return VerificationResult(**base)


def test_each_state_has_its_own_name():
    assert _r(checked=True).outcome == "verified"
    assert _r(checked=True, ok=False).outcome == "contradicted"
    assert _r().outcome == "unsupported"
    assert _r(not_applicable=True).outcome == "not_applicable"
    assert _r(infrastructure_failed=True).outcome == "verifier_failed"
    assert _r(timed_out=True).outcome == "timed_out"


def test_only_verified_is_correctness():
    for kw in ({}, {"not_applicable": True}, {"infrastructure_failed": True},
               {"timed_out": True}, {"checked": True, "ok": False}):
        one = _r(**kw)
        assert one.outcome != "verified"
        assert one.conclusively_ok is False, kw


def test_an_empty_verifier_set_is_not_a_pass():
    folded = combine_results("t", [])
    assert folded.outcome == "not_applicable"
    assert folded.conclusively_ok is False


def test_a_contradiction_outranks_an_engine_that_could_not_run():
    folded = combine_results("t", [_r(checked=True, ok=False, engine="a"),
                                   _r(infrastructure_failed=True, engine="b")])
    assert folded.outcome == "contradicted"


def test_a_pass_beside_a_crash_is_not_verified():
    folded = combine_results("t", [_r(checked=True, engine="a"),
                                   _r(infrastructure_failed=True, engine="b")])
    assert folded.outcome == "verifier_failed"
    assert folded.conclusively_ok is False


class _Engine:
    domains = ("code",)

    def __init__(self, name, behaviour):
        self.name = name
        self._behaviour = behaviour

    def handles(self, task_type):
        return task_type in self.domains

    async def verify(self, candidate, *, context=None):
        return await self._behaviour()


def _registry(*engines) -> VerifierRegistry:
    reg = VerifierRegistry.__new__(VerifierRegistry)
    reg._verifiers = list(engines)
    return reg


def _no_foundry(monkeypatch):
    monkeypatch.setattr(VerifierRegistry, "_foundry", staticmethod(lambda: None))


@pytest.mark.asyncio
async def test_a_crashed_engine_is_verifier_failed_not_unsupported(monkeypatch):
    _no_foundry(monkeypatch)

    async def boom():
        raise RuntimeError("engine exploded")

    out = await _registry(_Engine("code", boom)).verify("x = 1", task_type="code")
    assert out.outcome == "verifier_failed"
    assert out.infrastructure_failed is True


@pytest.mark.asyncio
async def test_a_timeout_is_named_and_does_not_escape(monkeypatch):
    _no_foundry(monkeypatch)

    async def late():
        raise TimeoutError("sandbox did not answer")

    out = await _registry(_Engine("code", late)).verify("x = 1", task_type="code")
    assert out.outcome == "timed_out"


@pytest.mark.asyncio
async def test_an_oserror_does_not_escape(monkeypatch):
    _no_foundry(monkeypatch)

    async def disk():
        raise OSError("scratch directory is gone")

    out = await _registry(_Engine("code", disk)).verify("x = 1", task_type="code")
    assert out.outcome == "verifier_failed"


@pytest.mark.asyncio
async def test_a_deadline_the_caller_gives_is_kept(monkeypatch):
    _no_foundry(monkeypatch)

    async def slow():
        await asyncio.sleep(5.0)
        return _r(checked=True)

    out = await _registry(_Engine("code", slow)).verify(
        "x = 1", task_type="code", context={"verifier_deadline_s": 0.05}
    )
    assert out.outcome == "timed_out"


@pytest.mark.asyncio
async def test_a_task_no_engine_covers_is_not_applicable(monkeypatch):
    _no_foundry(monkeypatch)

    class _Logic(_Engine):
        domains = ("*",)

        def handles(self, task_type):
            return True

    async def nothing_to_check():
        return _r(engine="logic")

    out = await _registry(_Logic("logic", nothing_to_check)).verify(
        "a sentence", task_type="poetry"
    )
    assert out.outcome == "not_applicable"


@pytest.mark.asyncio
async def test_a_covered_task_with_nothing_checkable_is_unsupported(monkeypatch):
    _no_foundry(monkeypatch)

    async def nothing_to_check():
        return _r(engine="code")

    out = await _registry(_Engine("code", nothing_to_check)).verify("prose", task_type="code")
    assert out.outcome == "unsupported"


def test_the_receipt_carries_the_outcome():
    from core.brain.reasoning_amplifier_v2 import ReasoningReceipt

    receipt = ReasoningReceipt(
        mode="normal", strategy_used="s", task_type="code", num_candidates=1,
        verifiers_run=[], valid_candidates=0, winning_candidate_id=None,
        confidence=0.5, agreement=0.0, epistemic_status="unverified",
        verification_outcome="timed_out",
    )
    assert receipt.to_dict()["verification_outcome"] in SIX


def test_escalation_does_not_adopt_on_a_check_that_could_not_run():
    """Source-level: the adoption gate reads whether verification was possible."""
    import inspect

    from core.brain import reasoning_amplifier_v2 as v2

    source = inspect.getsource(v2)
    at = source.index("esc_verdict = await self._verify(cand")
    gate = source[at: at + 600]
    assert "verification_was_possible" in gate
