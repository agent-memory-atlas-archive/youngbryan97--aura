"""A shorter native bound may reuse measured scores, never invent them."""

import copy
import json

import pytest

from core.learning.semantic_native_grammar import decode_native_grammar
from core.learning.semantic_native_relative_program import REGISTER_ENCODING
from tools.replay_semantic_native_depth import project_decisions


def _source_trace(preferred=None):
    if preferred is None:
        preferred = ("add", "input:0", "input:1", "continue",
                     "mul", "result:0", "input:2", "continue",
                     "sub", "result:1", "input:3", "finish")

    def score(choices):
        ordinal = score.index
        score.index += 1
        return tuple(10. if str(choice.value) == preferred[ordinal] else 0.
                     for choice in choices)

    score.index = 0
    return decode_native_grammar(("integer",) * 4, score, max_steps=8,
                                 register_encoding=REGISTER_ENCODING)


def test_projection_reuses_the_original_score_prefix():
    original = _source_trace()
    trace = json.loads(json.dumps(original.trace))
    program, status, forced, consumed = project_decisions(
        trace, ("integer",) * 4, max_steps=3,
        register_encoding=REGISTER_ENCODING)
    assert status == "completed"
    assert forced is True
    assert program == original.program
    assert consumed == len(original.trace) - 1


def test_projection_rejects_missing_or_mutated_scores():
    original = _source_trace()
    trace = json.loads(json.dumps(original.trace))
    with pytest.raises(ValueError, match="exhausted"):
        project_decisions(trace[:2], ("integer",) * 4, max_steps=3,
                          register_encoding=REGISTER_ENCODING)
    altered = copy.deepcopy(trace)
    altered[0]["scores"] = [0.] * len(altered[0]["scores"])
    with pytest.raises(ValueError, match="scored choice"):
        project_decisions(altered, ("integer",) * 4, max_steps=3,
                          register_encoding=REGISTER_ENCODING)


def test_projection_retains_disconnected_depth_failure():
    original = _source_trace(("add", "input:0", "input:1", "continue",
                              "add", "input:2", "input:3", "continue",
                              "add", "result:0", "input:0", "continue",
                              "add", "result:2", "result:1", "finish"))
    program, status, forced, consumed = project_decisions(
        json.loads(json.dumps(original.trace)), ("integer",) * 4,
        max_steps=3, register_encoding=REGISTER_ENCODING)
    assert program.depth == 3
    assert status == "disconnected_at_depth_bound"
    assert forced is False
    assert consumed == 11
