"""Three whole reports runs died partway on 28 September and each lost everything.

One died at anchor 21 of 24 after three hours of her cortex, killed by the
operating system with no traceback. Every anchor is now written as it is
measured and `--resume` carries them forward.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_report_grounding import ANCHOR_FILE, _carried_forward, _note_anchor


def _anchor(index: int) -> dict[str, object]:
    return {
        "anchor": index,
        "arms": {
            "raised": ["0.4", 0.4, True],
            "lowered": ["-0.1", -0.1, True],
            "sham": ["0.2", 0.2, True],
            "control": ["0.1", 0.1, True],
        },
        "answered": {"raised": [], "lowered": [], "sham": [], "control": []},
        "steered": {"raised": 0.7, "lowered": 0.45, "sham": 0.6, "control": 0.55},
    }


def test_nothing_to_carry_from_a_fresh_run() -> None:
    assert _carried_forward(None) == []


def test_a_resume_that_names_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        _carried_forward(tmp_path)
    assert "nothing to resume" in str(raised.value)


def test_every_anchor_is_on_disk_as_it_is_measured(tmp_path: Path) -> None:
    for index in range(3):
        _note_anchor(tmp_path, _anchor(index))
    lines = (tmp_path / ANCHOR_FILE).read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["anchor"] for line in lines] == [0, 1, 2]


def test_they_come_back_in_the_order_they_were_finished(tmp_path: Path) -> None:
    for index in range(5):
        _note_anchor(tmp_path, _anchor(index))
    carried = _carried_forward(tmp_path)
    assert [item["anchor"] for item in carried] == [0, 1, 2, 3, 4]
    assert carried[2]["steered"]["raised"] == 0.7


def test_a_half_written_last_line_does_not_lose_the_ones_before_it(tmp_path: Path) -> None:
    """What a kill in the middle of a write leaves behind."""
    for index in range(3):
        _note_anchor(tmp_path, _anchor(index))
    path = tmp_path / ANCHOR_FILE
    path.write_text(path.read_text() + '{"anchor": 3, "arms": {"rai')
    with pytest.raises(json.JSONDecodeError):
        _carried_forward(tmp_path)
    # The three whole ones are still there to be read by hand or by a repair.
    assert len(path.read_text().splitlines()) == 4


def test_the_run_skips_what_it_carried_and_records_what_it_measures() -> None:
    import inspect

    import tools.run_report_grounding as harness

    source = inspect.getsource(harness.main)
    assert "_carried_forward(args.resume)" in source
    assert "if index < len(done):" in source
    assert "_note_anchor(" in source
    # The carried anchors are written into the new run too, so its own record is
    # whole rather than a continuation nobody can read on its own.
    assert source.index("for item in done:") < source.index("for index, anchor in enumerate")
