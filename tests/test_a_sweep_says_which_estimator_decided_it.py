"""A sweep records which estimator decided its cuts, and the line refuses a mismatch.

The estimator is preregistered — the addendum of 29 September, afternoon, says
seed 23 is run to the paired displacement and that seed 7, read under Fisher-Rao,
cannot be the decisive reading. But the sweep report carried no estimator and
`isc_v5._conforms` checked none, so a sweep decided by the old estimator was
scored by `isc_v5.lines` exactly as one decided by the new one. The same held for
the one-signal rule, where a channel written twice counted twice.

A declaration nobody checks is the defect class this project keeps rediscovering.
"""

from __future__ import annotations

import pytest

from core.subject.isc_v5 import ALPHA, DRAWS, ESTIMATOR, LOOKS, ONE_SIGNAL, design, lines
from core.subject.v25_cut import CutVerdict, SweepReport

pytestmark = pytest.mark.unit


def _conforming(**over):
    plan = design()
    base = {
        "looks": plan["looks"], "draws": plan["draws"],
        "alpha_per_look": plan["alpha_per_look"], "paired": plan["paired"],
        "estimator": plan["estimator"], "one_signal": plan["one_signal"],
        "deciding": True, "screened": False, "shard": "",
        "cuts_tested": 511, "cuts_in_full": 511, "cuts_decided": 511, "undecided": [],
    }
    return base | over


def _irreducibility(sweep):
    return next(
        row for row in lines(sweep, playback_decided=0, nulls_that_pass=[], null_sweeps={})
        if row["key"] == "partition_irreducibility"
    )


def test_a_sweep_run_to_the_design_passes():
    assert _irreducibility(_conforming())["passed"]


def test_one_decided_by_the_old_estimator_is_refused():
    row = _irreducibility(_conforming(estimator="fisher_rao"))
    assert not row["passed"]
    assert "fisher_rao" in row["why"] and ESTIMATOR in row["why"]


def test_one_that_recorded_no_estimator_is_refused():
    row = _irreducibility(_conforming(estimator=""))
    assert not row["passed"]
    assert "did not record" in row["why"]


def test_one_that_counted_a_duplicated_channel_twice_is_refused():
    row = _irreducibility(_conforming(one_signal=not ONE_SIGNAL))
    assert not row["passed"]
    assert "written twice" in row["why"]


def test_the_report_carries_both():
    report = SweepReport(
        tau_seconds=2.0, cuts_in_full=511, looks=list(LOOKS),
        alpha_per_look=ALPHA / len(LOOKS), draws=DRAWS, paired=True,
        estimator=ESTIMATOR, one_signal=ONE_SIGNAL,
        verdicts=[CutVerdict(left=("P",), right=("I",), anchors_used=8)],
    )
    written = report.as_dict()
    assert written["estimator"] == ESTIMATOR
    assert written["one_signal"] is ONE_SIGNAL


def test_a_merge_refuses_shards_decided_differently():
    from core.subject.v25_cut import merge_sweeps

    def shard(index: str, estimator: str) -> dict:
        return {
            "tau_seconds": 2.0, "shard": index, "looks": list(LOOKS),
            "alpha_per_look": ALPHA / len(LOOKS), "draws": DRAWS, "deciding": True,
            "paired": True, "estimator": estimator, "one_signal": ONE_SIGNAL,
            "cuts_unscorable": 0, "undecided": [], "anchors_spent": 8,
            "verdicts": [
                {
                    "cut": f"P|{index[0]}", "anchors": 8, "excess_rate": 1.0,
                    "lower_bound": 0.5, "p_value": 0.01, "raw_rate": 1.0,
                    "sham_rate": 0.0, "decided": True, "note": "",
                    "playback_decided": False,
                }
            ],
        }

    with pytest.raises(ValueError, match="different designs"):
        merge_sweeps([shard("0/2", ESTIMATOR), shard("1/2", "fisher_rao")], cuts_in_full=2)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
