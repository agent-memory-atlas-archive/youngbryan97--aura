# G03 complete development windows, 2026-09-28

The full-prefix native evaluator previously selected only the first N sources
and limited the entire process to 14,400 seconds. It had no source-offset
contract. An interrupted large run could retain rows without producing a
complete report; those rows did not permit a verified continuation.

The new `--source-offset` path partitions the same complete 500-request
`retained_validation` population. It neither changes the scorer nor reads
answers to select a window. Each plan binds its offset, count, and the digest
of all 500 source IDs in their existing cohort-round-robin order. The v12
verifier reconstructs that exact window from the pinned source report and
manifests. Historical schemas cannot acquire a window. Every arm still holds
an exclusive, non-evicting model lane.

`tools/adjudicate_semantic_native_development.py` independently replays each
window's fitted, base, and fitted source-erasure arms. It rejects missing or
reordered windows, overlaps, changed candidates, changed search or time
budgets, and any incomplete source coverage. It recomputes outcomes from all
500 per-source comparisons. Advancement requires 500 exact fitted procedures,
500 correct public values, no lost baseline success, no forced completion,
and source-dependent gain. This is a development stage, not a fresh-family,
broad-reasoning, fusion, or serving verdict.

## Checks

The CPU source check reconstructed 500 distinct validation sources. Their
ordered digest is
`9cd1d8d3dec0391ce1aed6fff780c54a2886a51bb0dcd2d368f75624b3af5f40`.
The final six-source window at offset 494 exactly matched entries 494-499.
No model was loaded by this check.

The retained validation cohort counts are arithmetic 128, cataphoric 48,
fork-join 192, natural-alias-source 12, natural-source 24, reserved-alias 48,
and role-binding 48. These are exposed development sources. Calling them
fresh transfer would misstate their status.

`tests/test_semantic_native_development_windows.py` checks complete ordered
coverage, historical-schema isolation, independent source reconstruction,
paired identity and budget mismatches, and failure without stage advancement.
The adjacent registered invariant attacks malformed population bounds.

The v7 fit and micro controller remain pinned in their original checkouts.
This change was built elsewhere and does not reset, replace, or re-decode a
passed micro stage. No window decode or G03 closure is claimed here. Freeze
window sizes and identical arm budgets before the full run, using measured
micro latency without inspecting labels to choose budgets or populations.
