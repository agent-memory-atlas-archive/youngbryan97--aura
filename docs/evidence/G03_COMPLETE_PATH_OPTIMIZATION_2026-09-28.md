# Complete native paths and optimization

G03 and G04 remain open. No model is promoted or fused by this work.

## Full calibration replay

The replay measures every teacher-path competition on all 185 source
calibration requests at steps 0, 101, 202, and 303 of the paired fit. It
does not inspect the exposed held-bank outcomes to select a checkpoint.

| Step | Exact paths | Operations / 434 | References / 868 | Terminations / 434 | Lost baseline paths |
| --- | --- | --- | --- | --- | --- |
| 0 | 51 | 311 | 744 | 424 | 0 |
| 101 | 51 | 395 | 674 | 434 | 36 |
| 202 | 63 | 426 | 700 | 434 | 38 |
| 303 | 27 | 411 | 638 | 434 | 51 |

The 202 checkpoint improved operations while losing 38 baseline-correct
paths. Reference correctness fell even when conditioned on the correct
earlier choices. This loss cannot be explained solely by an earlier
operation error leading the decoder onto a different path.

Evidence: `semantic-native-path-calibration-full-v1-20260928` under
`/Users/bryan/.aura/rlc-evidence`. Report receipt:
`b242dea26397af719fbd333c4281d2538b030e9ad59029ec08f1a577ca9643b0`.
Independent verification before the redesign reported no source drift:
`dbb159fbe95d81bb0e4259813a0d66ca69ca645d72724d84e9fcd77f21e0d324`.
The original fit and its minimum-loss selection remain historical evidence.

## Training change

The v5 opt-in objective aggregates every conditional decision loss and the
declared source-pair interaction into `log(mean(exp(losses)))`. The old
objective added an operation-only interaction at the weight of an entire
path, whose individual bindings had been averaged down.

For finite losses, Jensen's inequality and monotonicity establish:

`mean(losses) <= log(mean(exp(losses))) <= max(losses)`.

Its gradient weights are `softmax(losses)`. A weak reference therefore
receives more gradient weight than an already-solved operation. Equal
losses reproduce the ordinary mean's gradient scale. This is an
optimization hypothesis; it does not establish correctness on unseen inputs.

Every checkpoint now retains all source-calibration choice scores. The
reader verifies each choice against the bound supervision, reconstructs
complete-path totals, and refuses a checkpoint that loses any exact path
from the unfitted baseline. Among eligible checkpoints it chooses the
most exact paths, then lowest calibration loss, then earliest step.
Step zero remains eligible. This protects measured conditional paths,
not arbitrary generated outputs or fresh-family performance.

## Capture reuse

The annotation pass `e9d2e740c` changed eight capture/supervision file
hashes. Their archived sources match the original capture's hashes at
`caf311832f09cf460f36a8ca7eeba9af0df7e3e3`. After erasing postponed
function signatures and import-only static typing blocks, their numeric
ASTs match the current files. Class fields, operation order, defaults,
mask logic, and dtype expressions remain in the comparison. Annotation
introspection equivalence is explicitly not claimed.

The bound archive receipt is
`50bd5e7b3d9f431a76919b2937e86a823af70313a8c44189bf3d0b11fc927206`.
Reuse still requires the exact model, arithmetic, source partitions,
sequence digests, complete shard inventory, and a fresh optimizer schedule.
It never relabels old fit results as new evidence.

## Prepared run

`semantic-native-path-risk-fp32-full-v5-20260928` has a frozen plan
`ba7a5785515f2fdc5072e45d077def977c5930c82edc16c8356ebcfac30cd73a`.
Preflight reconstructed 21,733 sequences for 303 fit and 185 calibration
sources. Their lossless prefix states are reused; model weights were not
loaded during preparation. The schedule is 303 updates with checkpoints
0, 101, 202, and 303, rank 8, one suffix layer, and FP32 arithmetic.
The 50 previously exposed held-bank requests are development observations.

The model run, independent fit verification, and generated comparison
are pending at this checkpoint. No broader validation follows unless the
measured candidate justifies it. Focused checks: 158 passed, three optional
tokenizer checks skipped.

## Completed v5 fit and residual measurement

The prepared run completed. Its independently reconstructed calibration is:

| Step | Exact paths / 185 | Operations / 434 | References / 868 | Terminations / 434 | Lost baseline paths |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 51 | 311 | 744 | 424 | 0 |
| 101 | 104 | 426 | 754 | 434 | 12 |
| 202 | 34 | 423 | 704 | 434 | 51 |
| 303 | 42 | 425 | 649 | 434 | 51 |

Baseline-preserving selection chose step zero. The exposed 50-case bank
then scored 39/50 against incumbent 41/50, with three gains and five
regressions. The fit is rejected. Its independent verification receipt is
`d5e6cf311b55e04387dbf83b984d34811da67d0f3c1f3b563b8fb23a4a4e13e1`.

The separate residual measurement loaded checkpoint 101 and evaluated all
185 source-calibration paths at scales 0, 0.08, and 1. Both endpoints
reproduced the saved checkpoint scores and path profiles. Scale 0.08
reached 60/185 exact paths: nine new paths and no lost baseline path.
Scale 1 reproduced 104/185 and the twelve losses. The measurement changed
neither training selection nor serving authority. Its independent receipt
is `c97833e01e24c9997f7c62ffdaa7011de84c474890148df12752db244605f03c`.

Target-blind generation on fourteen retained development requests did not
carry this source gain forward. Residual and freshly replayed base both
scored 1/14 exact procedures, 3/14 observed answers, and two proven
output/domain equivalences. Paired gains and regressions were zero for
each correctness measure. The residual completed two additional wrong
graphs; completion cannot count as a semantic gain. Both runs ended with
empty process lineage and were independently replayed without source drift.

Evidence under `/Users/bryan/.aura/rlc-evidence`:

- `semantic-native-residual-source-cal-v1-20260928`
- `semantic-native-residual-retained-14-v1-20260928`
- `semantic-native-risk-base-retained-14-v1-20260928`
- `semantic-native-residual-retained-comparison-v1-20260928.json`

Paired receipt:
`0174eddbf5d368914f51d031fa5384599b114d039e96d3888d3e94a9453711c0`.
G03 and G04 remain open. No broad run, promotion, or fusion follows.
