# CAA contrastive development path (not a qualification)

The current 27B public steering comparison is negative:
[G10 public result](G10_PUBLIC_STEERING_RESULT_2026-09-14.md). It used real
model-bound vectors; the assertion that the 27B has only bootstrap vectors is
stale. This checkpoint adds an upstream candidate-generation and development
selection path. It does not replace that result or grant serving authority.

`training/caa_contrastive_corpus.py` supplies paired positive/negative
statements across two fit template families and one disjoint development
family, with separate topics. The corpus is exploratory and small. It is not
evidence that the contrast spans the full target behavior. The design carries
the active model descriptor; duplicate prompts, partition leakage, and missing
topic coverage are rejected.

`tools/capture_27b_steering_vectors.py --contrastive-corpus --out <fresh-dir>`
captures only fit pairs in one model load and writes separate `raw/`,
`purified/`, and `polarity_nulls/<index>/` generations. Purification removes
explicitly measured nuisance directions, not guessed positional PCs. Raw and
purified remain competing candidates because projection may remove target
signal. The null flips polarity labels within pairs; reordering pairs would
leave a mean-difference CAA vector unchanged. Incomplete layer or prompt
capture fails instead of silently reducing sample size. Every vector carries
the 27B descriptor and corpus fingerprint, and no output is installed live.

`tools/run_caa_steering_campaign.py --development --vectors <arm> --layers
<attention-layer-list> --alpha <value> --polarity-null-vectors <null-arm> --out
<new-result>` runs disjoint target-blind development prompts through the same
public-channel decoder and hook path as the sealed campaign. The existing
matched baseline, zero, random, layer-shuffle, terse, and rich-text controls
remain. Development adds the polarity-label null and the forced-choice
capability battery, with substrate refresh before each steered item. Multiple
layer/alpha/arm cells are development measurements, not independent sealed
tests. A short `--calibration-only` development run is diagnostic and cannot
enter selection.

`tools/select_caa_development.py --result <result> ... --out <new-selection>`
rehashes vector generations, recomputes target scores and public completion,
and requires a positive target effect beyond the rich-text comparator, null
equivalence within a declared band, and no capability loss beyond the
declared allowance. It freezes one eligible cell or records that none exists.
It cannot turn a failed cell into a serving artifact. No model-active
development run, candidate selection, or sealed campaign has been performed
for these new candidates yet.

The attached proposals contain useful experimental controls but also claims
that do not follow. Entropy thresholds cannot guarantee correctness or zero
regressions; a confident model can be wrong. Top activation PCs are not
necessarily position, and unconditional erasure can destroy binding. A CAA
direction need not be layer-assignment specific when adjacent layers share an
axis; layer shuffle is reported, while the random/zero/polarity controls test
the direction. Exact zero degradation across all unseen constructions is not
a scientifically sound precondition. The proposed system-wide causal state
bus and hidden-state-only semantic decoder are separate research changes,
not substitutes for the G03 source-conditioned program mechanism.

Implementation checks: 26 focused tests, smoke 164 passed and one skipped,
ruff, compile, governance lint, and layering passed. These are code checks,
not behavioral steering results. G10 remains open.
