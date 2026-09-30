# G03 counterfactual acquisition preflight, 2026-09-29

The fit-only fork/join counterfactual corpus built with seed 41 contains 2,241
sources in 648 distinct construction/topology/operation cells. Feature
selection refuses a cap below one example per cell. The materializer's default
cap of 576 was therefore invalid for this corpus. The single-cohort CLI now
checks selection before it configures the MLX device or contacts the resident
model. Its error reports the requested and required counts.

A CPU-only dry run selected 648, 1,024, and 1,536 examples with the existing
balanced selector. Using source-text bytes as **diagnostic** token IDs, the
typed source-pair planner found witnessed operation and reference peers for
every selected source at each cap. Decision 9, the late fork/join reference,
had 390, 614, and 872 paired sources respectively. No termination pair was
available. These counts prove only that the selected program/source labels
can form contrastive pairs under the diagnostic tokenizer. They do not prove
model-tokenizer pairing, learned binding, held-family transfer, or a public
answer gain.

The smallest declared feature cohort that preserves all 648 cells uses
`--max-examples 648`. It still requires exclusive resident-model feature
acquisition, a new source report and frozen fit bank, then training and
independent development/transfer evaluation. No new features or model
weights were acquired for this record. The existing G03 promotion bar remains
unchanged.
