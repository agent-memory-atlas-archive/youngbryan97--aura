# G03 v7 micro-probe protocol, 2026-09-28

This protocol was fixed before the v7 model-active fit or any v7 generated
answer. It tests whether a source-only joint-graph fit can produce one new,
exact, source-dependent procedure without losing a baseline success. It is a
small development probe, not an independent transfer or promotion test.

## Frozen inputs

- Training plan: `7fff4f3aa03da3254b3bc5e177714e5acfc0d5a72037f6623c3480babdd51780`
  in `semantic-native-joint-graph-v7-partial-final-20260928`.
- Source report bytes: `30c955d0655bb087c5f8081e4b13fb9935d2078e334e5fa64447a470119d18e9`.
  Use the eight pinned feature manifests in that report. Do not substitute a
  rebuilt or relabeled source population.
- Split: `retained_validation`, first six sources in
  `cohort_round_robin_source_digest` order. This order ignores answers and
  model scores. It was checked against the training plan: zero overlap with
  its 303 fit, 185 calibration, and 50 held source IDs.

| Order | Source SHA-256 | Construction |
| --- | --- | --- |
| 1 | `005ccc967d8b4588f4d2b58e66839da796a049d0fe6d8ba7ea9ebcd30556789a` | `sequential_obtained_afterward` |
| 2 | `041682cce3dc244504989f04531772a51e939aac8cda24230057db847e6de5f4` | `sequence-cataphoric-4` |
| 3 | `021b4921101b1639b18a9ece23a8c80cb3e19ef969d31a56468531e75845f1da` | `fork_derive_elsewhere_finish` |
| 4 | `3f357a5a88b3f6300eaaafca0e4e73e3b9dd60f9bb865ccd131415d3a927e51b` | `natural-alias-source-count_alias_linear_two-4` |
| 5 | `0a59c82618773467c54e3fd77ce03e7876a19e4005d0b7ec1be06dfe401d6a2c` | `natural-source-scalar_linear_two-4` |
| 6 | `0b5a8a692cb34625ea9ce2fffdee57bc39b982bfde69d273f64169a4142dacdb` | `sequence-reserved-alias-5` |

A model-free structural check used the plan's register encoding and equal
finite choice scores. Each input signature yielded four connected completions
within 43-72 expanded nodes under the 256-node cap. This only checks that
the grammar can traverse the cohort at this budget. It does not measure
model latency, candidate reach, ranking, or correctness.

## Run after source-only checkpoint selection

Use `tools/evaluate_semantic_native_grammar.py` on the selected fitted
checkpoint and the frozen base model. Both arms must use `--canary 6`,
`--max-steps 8`, `--search-completions 4`, `--search-nodes 256`,
`--search-score-mode native_nonpositive`, `--prefix-strategy full`,
`--source-evidence source_text`, the same source report and bundles, and a
finite `--max-seconds 3600` per arm. Change only `--weight-mode` and the
output directory. Save immutable plans before either decode. Do not change
the search budget after seeing an answer. Run a fitted
`--source-evidence source_token_erasure` arm with the same budget and cohort.

Independently verify each arm with `tools/verify_semantic_native_grammar.py`,
then compare fitted and base with
`tools/compare_semantic_native_grammar_fit.py`. The target program may be
used for grading only after target-blind generation and complete-graph
selection. Inspect the six per-source outputs and the erasure rows; a process
exit code or aggregate answer count is insufficient.

## Interpretation fixed in advance

An exact new procedure means a fitted `completed` graph independently judged
equivalent to the annotated procedure on a source where base is not exact.
Its observed public answer must also be correct. Any base-exact source made
inexact is a regression. A bounded source-transfer instance needs at least
one exact new procedure, zero exact-procedure regressions, no invalid plan or
verification receipt, and no disconnected or forced result counted as a win.
If base has no exact successes here, non-regression is uninformative and must
be reported that way.

Source dependence requires at least one gained source to change its selected
procedure or lose exactness under source-token erasure. If every gain survives
erasure, this probe cannot attribute it to the request text. Report that
separately from exactness. An erasure effect alone does not prove the learned
semantic relation; the broader source-pair and held-family controls remain.

These six cases were previously exposed as development material and do not
hold out whole construction families. Even a clean result cannot close G03,
establish G04 transfer, justify fusion or serving, or predict a 500-case
result. A failure is useful: inspect reach, graph ranking, and native
termination per source before changing the fit or the search budget.
