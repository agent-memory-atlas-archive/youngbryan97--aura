# Cached grouped native search, 2026-09-29

The active 256-node micro campaign uses causal grouping with a full frozen
prefix forward for each group. The new opt-in `v15` evaluation path combines
those groups with the existing frozen-prefix trie. It computes the common
source anchor once per request, reuses causal continuation states across
branches, and leaves suffix scoring and whole-graph selection unchanged.

The saved plan binds the model, checkpoint, implementation, source population,
search budget, `prefix_strategy=trie`, and
`decision_score_execution=causal_groups`. The independent verifier checks
group counts, trie branch accounting, source-anchor identity, search replay,
and matching execution across fitted, base, and source-erasure arms. Relation
controls and 500-request development windows cannot inherit a different
prefix strategy.

Small frozen-model tests cover dense and hybrid layers, tied and untied output,
and 4-bit quantization. Short-sequence grouped/trie scores matched full-prefix
scores exactly in those tests. On 132-token quantized hybrid sequences, the
observed differences were below `1e-5` log-probability and preserved the
winner in the tested competitions. That is a bounded numerical observation,
not bitwise equivalence or a guarantee for near-tied choices.

No real-27B timing or ranking comparison has been run for `v15`. The current
source-matched campaign remains on its frozen full-prefix implementation.
The read-only `tools/probe_semantic_native_causal_groups.py --cached-groups`
plan-only replay recovered seven shared-decision cases across all three
completed fitted reference sources, bound to plan digest
`2524355071814759ca8505998fe62e5b2d2599f055a29a015ac19c220456a06c`.
The saved plan is under
`/Users/bryan/.aura/rlc-evidence/semantic-native-v15-cached-probe-plan-20260929`.
It has not loaded the model or scored the cached arm. The probe records
full-prefix, grouped, and cached scores, winners, and elapsed time without
granting promotion authority.
Before using the cached path in a candidate campaign, compare its decision
scores, selected programs, and elapsed time against full-prefix execution on
the same 27B checkpoint and sources. Then run the matched reference screen and
all later controls under a newly frozen plan. This implementation grants no
serving authority and closes no G-ledger item.

The CPU micro adjudicator now reads the finite bound recorded in the hash-bound
plan, up to the evaluator's 14,400-second maximum. Its former 3,600-second
constant would have rejected the running campaign's valid 14,400-second
plans after decoding, even if all three arms completed and verified.
