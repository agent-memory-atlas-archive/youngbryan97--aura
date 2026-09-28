"""How operation charts are calibrated and their length penalty chosen.

Lifted whole out of `semantic_program_transducer_fitting`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .semantic_program_transducer_fitting import (
        LinearPointerHead,
        MultiViewClassifierHead,
        SemanticTransducerTrainingExample,
        _OperationNode,
    )


def _select_operation_length_penalty(
    validation: Sequence[SemanticTransducerTrainingExample],
    *,
    pointer: LinearPointerHead,
    classifier: MultiViewClassifierHead,
    max_steps: int,
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[float, list[dict[str, Any]]]:
    from .semantic_program_transducer_fitting import (
        _best_nonoverlapping_nodes,
        _calibrate_operation_charts,
        _operation_nodes,
    )

    cached: list[
        tuple[
            SemanticTransducerTrainingExample,
            tuple[tuple[float, tuple[_OperationNode, ...]], ...],
        ]
    ] = []
    for item in validation:
        nodes = _operation_nodes(
            pointer=pointer,
            classifier=classifier,
            hidden=item.hidden_states,
            input_spans=item.ir.input_spans,
            max_span_tokens=max_span_tokens,
            hidden_channels=hidden_channels,
            hidden_channel_widths=hidden_channel_widths,
        )
        by_count = tuple(
            _best_nonoverlapping_nodes(nodes, count) for count in range(1, max_steps + 1)
        )
        cached.append((item, by_count))
    return _calibrate_operation_charts(cached)


def _calibrate_operation_charts(
    cached: list[tuple[SemanticTransducerTrainingExample, tuple[tuple[float, tuple[_OperationNode, ...]], ...]]],
) -> tuple[float, list[dict[str, Any]]]:
    """Calibrate source-ordered charts independently of execution order."""
    from .semantic_program_transducer_fitting import (
        _OPERATION_PENALTY_POINTS,
        _best_penalized_operation_chart,
        np,
    )

    average_scores = [
        score / count
        for _item, by_count in cached
        for count, (score, _selected) in enumerate(by_count, start=1)
        if math.isfinite(score)
    ]
    if not average_scores:
        raise ValueError("compositional operation chart has no validation candidates")
    penalties = np.linspace(
        min(average_scores) - 5.0,
        max(average_scores) + 5.0,
        _OPERATION_PENALTY_POINTS,
    )
    rows: list[dict[str, Any]] = []
    for raw_penalty in penalties:
        penalty = float(raw_penalty)
        span_exact = 0
        operation_exact = 0
        graph_exact = 0
        for item, by_count in cached:
            selected = _best_penalized_operation_chart(by_count, penalty=penalty)
            expected = sorted(item.ir.instructions, key=lambda instruction: (instruction.operation_span.start, instruction.operation_span.end))
            expected_spans = tuple(
                instruction.operation_span for instruction in expected
            )
            expected_operations = tuple(instruction.op for instruction in expected)
            observed_spans = tuple(node.span for node in selected)
            observed_operations = tuple(node.operation for node in selected)
            span_exact += int(observed_spans == expected_spans)
            operation_exact += int(observed_operations == expected_operations)
            graph_exact += int(
                (observed_spans, observed_operations) == (expected_spans, expected_operations)
            )
        rows.append(
            {
                "length_penalty": penalty,
                "graph_exact": graph_exact,
                "span_exact": span_exact,
                "operation_exact": operation_exact,
                "validation_examples": len(cached),
            }
        )
    winner = max(
        rows,
        key=lambda row: (
            row["graph_exact"],
            row["span_exact"],
            row["operation_exact"],
            -abs(row["length_penalty"]),
        ),
    )
    return float(winner["length_penalty"]), rows
