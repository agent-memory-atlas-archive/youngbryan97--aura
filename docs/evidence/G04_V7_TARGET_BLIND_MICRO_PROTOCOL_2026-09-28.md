# G04 v7 target-blind micro-probe, 2026-09-28

This source-only protocol was fixed while the v7 model-active fit was running,
before its checkpoint or any v7 generated answer was available. It tests a
small held-construction instance with the existing native grammar and search.

The seed is the first eight hexadecimal digits of the frozen v7 training plan
SHA-256, interpreted as an unsigned integer: `2147438394`. With
`grammar_examples(dataset="natural_request", seed=2147438394, count=3)`, the
first three requests provide one scalar, one lookup, and one count problem.

| Order | Source SHA-256 | Construction |
| --- | --- | --- |
| 1 | `748d5cc1f3f502994994fb88246e79ef92b6f3461f9ac6fef5d2e0eff847f78b` | `natural-scalar_linear_three-0` |
| 2 | `e4c29aa33d0e0f5960d5c96669ca62c60b10c9b1fe902fc61cbe72eeb7d5a6b1` | `natural-lookup_linear_three-0` |
| 3 | `08b36b08dae62300ed3775594c0521474f122e42178601761c26010fed816613` | `natural-count_linear_three-0` |

These source IDs have zero overlap with the v7 fit, calibration, and held
source IDs or the earlier 72-case fresh-schema bank. The three construction
IDs are absent from every v7 training partition. All three three-step
targets have admitted paths with explicit finish decisions under the existing
role-relative grammar. With equal choice scores, each input signature yields
four connected completions within 43-77 expanded nodes under a 256-node cap.
That checks structural feasibility, not learned ranking or target reach.

After source-only checkpoint selection, run
`tools/evaluate_semantic_native_grammar.py` with `--dataset natural_request`,
`--seed 2147438394`, `--canary 3`, `--max-steps 8`,
`--search-completions 4`, `--search-nodes 256`,
`--search-score-mode native_nonpositive`, `--prefix-strategy full`,
`--source-evidence source_text`, and `--max-seconds 3600`. The fitted and base
arms must differ only in weight mode and output directory. Freeze both plans
before decode. Run a fitted source-token-erasure arm under the same budget.
Independently verify each arm and use
`tools/compare_semantic_native_grammar_fit.py` for the matched comparison.
Do not use target programs to construct or rank candidates.

One fitted exact procedure that base misses, with a correct public answer,
zero lost base-exact procedures, and complete independent verification would
show one bounded held-construction gain. If base has no exact success, the
non-regression result is uninformative. A gained procedure must also change
or lose exactness under source-token erasure before source dependence can be
claimed. Report target reach, requested top-k proof, disconnection, and
forced completion for every case; an exit-zero run is not a success verdict.

These constructions were used in earlier development studies, although this
seed's sources are new to the frozen v7 fit and earlier 72-case bank. A clean
three-case result cannot close G04 or prove general transfer across families,
vocabulary, depth, or domains. It cannot justify serving, fusion, a broad
reasoning claim, or frontier performance. Those require larger, independent
and matched comparisons after this diagnostic.
