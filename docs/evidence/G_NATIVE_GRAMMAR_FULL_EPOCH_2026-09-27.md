# Native grammar complete-epoch result

The declared 303-source, one-update-per-source FP32 grammar-choice fit
completed in 3,944.29 seconds. It reused 21,733 immutable frozen-prefix
sequences from 488 source shards, but initialized a fresh optimizer and
executed every update. The selected checkpoint was step 202 by the frozen
source-calibration rule (loss 0.16561141299825977); the final step 303 had
loss 0.26284509545769197 and was not selected. Checkpoints 0, 101, and 202
matched the interrupted source run bit for bit, including weights digests.

On the exposed 50-source held bank, the selected fit got 44 correct versus
41 for the incumbent and 39 for the unfitted native scorer. It gained five
over the incumbent and regressed on two. Only 47 target programs were in the
candidate bank. The arithmetic cohort was 2/6 correct, with only 3/6 targets
reachable. The two incumbent-to-fit regressions were an arithmetic sequential
result and a cataphoric sequence. This is not a non-regressing result, a
target-blind decode result, or a general-transfer result.

The independent CPU verifier reproduced the held totals, reconstructed the
supervision and source partitions, checked all 4,416 decision groups, and
verified every durable shard byte and sequence binding (61,148,761,124
bytes). It did not independently recompute model hidden states. No
contradictory identical scoring inputs were found; that excludes one
artificial impossibility, not shortcut learning or poor optimization.

Evidence directory:
`semantic-native-grammar-choice-fp32-reuse-full-epoch-v2-20260927` under
`~/.aura/rlc-evidence/`. Plan SHA-256:
`d4d9c74484fe60a8c11559f3c9c80e77cec8fbd27b905092fd5246b0d1577714`.
Fit receipt:
`4521b75cd8697bccefb506d42775f33f39d79eeca13f8fe121a1f28fc031ebb7`.
Independent verification receipt:
`241728424460c91c08e8c0e350ff1d6656c3ffe38a7cc6021f3acf451b7587b7`.

The next experiment must discriminate source-conditioned decisions from
output-wire adaptation using target-blind, paired source interventions and
matched unfitted controls. This result grants no serving, fusion, G03, G04,
G06, or broad-gain closure.
