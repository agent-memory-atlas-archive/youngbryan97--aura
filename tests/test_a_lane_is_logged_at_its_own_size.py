"""The brainstem was logged as a 7B and has been a 27B at two bits since
20 September.

A size typed into a log line is a claim nobody checks. This one was wrong for a
week, and it mattered on 27 September: the cortex was refused its spawn —
`memory_pressure_refused_worker_spawn:model_load_headroom:11.9GB < required
24.0GB` — and the only other lane's line said "(7B)", so the question the log
should have settled, whether an 8.6GB fallback fits in 11.9GB of headroom, could
not be asked of it.

The line reads the footprint the lifecycle declares.
"""

from __future__ import annotations

import inspect
import re

import pytest

from core.brain.llm import autonomous_brain_integration as integration
from core.brain.llm.model_lifecycle import _APPROX_SIZE_GB
from core.brain.llm.model_registry import BRAINSTEM_MODEL


def test_the_brainstem_declares_a_footprint():
    assert BRAINSTEM_MODEL in _APPROX_SIZE_GB
    assert _APPROX_SIZE_GB[BRAINSTEM_MODEL] > 0.0


def test_the_footprint_is_read_from_the_declaration():
    declared = _APPROX_SIZE_GB[BRAINSTEM_MODEL]
    assert integration._declared_footprint_gb(f"/models/{BRAINSTEM_MODEL}") == pytest.approx(declared)
    assert integration._declared_footprint_gb(f"/models/{BRAINSTEM_MODEL}/") == pytest.approx(declared)


def test_an_undeclared_artifact_reports_nothing_rather_than_guessing():
    assert integration._declared_footprint_gb("/models/no-such-model") == 0.0
    assert integration._declared_footprint_gb("") == 0.0


def test_no_tier_line_types_a_size_into_itself():
    """A size typed into a message is a claim nobody checks."""
    source = inspect.getsource(integration)
    assert "(7B)" not in source
    assert "_declared_footprint_gb(brainstem_model_path)" in source


def test_the_registration_line_carries_no_hard_coded_parameter_count():
    """The label and the artifact have to agree: a reader uses the label to
    decide whether the lane fits in the headroom that is left."""
    assert "27B" in BRAINSTEM_MODEL
    source = inspect.getsource(integration)
    start = source.index("TERTIARY Tier registered")
    call = source[start : source.index(")", source.index("Background/Reflex"))]
    assert not re.search(r"\d+\s*[Bb]\b", call), call
    assert "%s" in call and "%.1fGB" in call
