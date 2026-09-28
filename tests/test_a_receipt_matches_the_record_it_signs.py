"""The post-action receipt is refreshed on every field it carries.

LIVE 2026-09-28 00:45, on top of a page run that had already failed:
"post-action receipt outcome contradicts terminal transaction on
manual_reconciliation_required (receipt False against True)", raised twice, each
time a degraded degradation, an incident and a MARGINAL fault, plus a resilience
hit that took frustration to 0.30.

`_execute_final_error_msg` rebuilds the receipt from the settled result and sets
six fields, but it decided whether to rebuild by comparing four of them.
`retry_safe` and `manual_reconciliation_required` are settled AFTER the receipt
is first built — a terminal failure whose transport succeeded needs reconciling —
so when one of those was the only thing that changed, the receipt kept the value
it captured before the decision and the linker found its own two halves
disagreeing.
"""
from __future__ import annotations

import pytest

from core.runtime.action_executor_steps import _RunsTheActionSteps
from core.runtime.post_action_receipt import PostActionReceipt

pytestmark = pytest.mark.unit


def _receipt(**over) -> PostActionReceipt:
    fields = dict(
        receipt_id="post-1", will_receipt_id="will-1", executor_name="sovereign_browser",
        actual_outcome="failure", output_hash="h0", error_status="no_progress",
        welfare_transaction_id="tx-1", status="failed_recoverable",
        effect_verified=False, transport_succeeded=True, retry_safe=False,
        manual_reconciliation_required=False,
    )
    fields.update(over)
    return PostActionReceipt(**fields)


def _settled(**over) -> dict:
    result = dict(
        ok=False, status="failed_recoverable", error="no_progress",
        retry_safe=False, manual_reconciliation_required=False,
    )
    result.update(over)
    return result


def test_a_late_reconciliation_flag_reaches_the_receipt():
    receipt = _receipt()
    result = _settled(manual_reconciliation_required=True)
    _error, refreshed = _RunsTheActionSteps._execute_final_error_msg(
        False, "failed_recoverable", True, receipt, result,
    )
    assert refreshed.manual_reconciliation_required is True, (
        "the receipt still says no reconciliation is needed while the record it "
        "signs says one is"
    )


def test_a_late_retry_safe_flag_reaches_the_receipt():
    receipt = _receipt()
    result = _settled(retry_safe=True)
    _error, refreshed = _RunsTheActionSteps._execute_final_error_msg(
        False, "failed_recoverable", True, receipt, result,
    )
    assert refreshed.retry_safe is True


def test_a_record_that_did_not_move_keeps_its_receipt():
    """Rebuilding for nothing would change the output hash for nothing."""
    receipt = _receipt()
    _error, refreshed = _RunsTheActionSteps._execute_final_error_msg(
        False, "failed_recoverable", True, receipt, _settled(),
    )
    assert refreshed is receipt


def test_the_rebuild_trigger_covers_every_field_it_sets():
    """A seventh field added to the rebuild must be added to the test above it."""
    import inspect

    body = inspect.getsource(_RunsTheActionSteps._execute_final_error_msg)
    head, _, tail = body.partition("post_receipt = PostActionReceipt(")
    for name in (
        "status", "effect_verified", "error_status", "transport_succeeded",
        "retry_safe", "manual_reconciliation_required",
    ):
        assert name in head, (
            f"{name} is set by the rebuild but nothing above decides on it"
        )
