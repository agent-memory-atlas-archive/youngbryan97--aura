# Paired source-choice fit on the exposed bank

The FP32 `grammar_source_pairs` fit completed all 303 scheduled updates over
303 distinct source identities. Checkpoint selection used source calibration
and chose step 202. The independent CPU verifier reconstructed the source
partitions, 4,416 grammar-decision groups, 21,733 captured sequences, checkpoint
digests, and every held-bank comparison. It verified the 61,148,761,124 stored
prefix bytes and their sequence bindings, but did not independently recompute
the frozen model states.

| Exposed held bank | Correct / 50 |
| --- | ---: |
| Incumbent | 41 |
| Unfitted native scorer | 39 |
| Paired source-choice fit | 47 |

The fit gained six answers over the incumbent with zero regressions on this
bank. This is the first non-regressing complete-bank fit in this lineage. It
is a development result, not a fresh-family or serving result. The bank
contained a correct candidate on exactly 47 sources. All three absent targets
belonged to `arithmetic:nominal_nested`. Each source exhausted both its
eight-chart and four-graph allowances without completing operation search.
The fitted scorer could not select a program the bank did not contain.

An earlier proposer generated equivalent programs for these same sources,
but its model and source-training lineage differ. Its candidates cannot be
imported into this held comparison as independent transfer evidence. They do
show that the typed grammar is expressive enough; proposal acquisition and
model-specific ranking remain separate problems.

No source-erasure intervention was run on this checkpoint. The verifier's
`source_control_verified` field is false. The fit does not yet prove that
the gained choices depended on source meaning rather than output-wire or
candidate-inventory effects. Target-blind generation, causal source controls,
fresh construction transfer, and the serving boundary remain open. G03 is
not closed.

Evidence directory:
`~/.aura/rlc-evidence/semantic-native-grammar-source-pairs-fp32-full-v4-20260927`.
The independent receipt is `independent-verification.json` in that directory.
Selected checkpoint receipt SHA-256:
`3c1cffbea25bafa4b90d6b32bda52cce1215bc8a27d817c35f33d9807c175408`.
