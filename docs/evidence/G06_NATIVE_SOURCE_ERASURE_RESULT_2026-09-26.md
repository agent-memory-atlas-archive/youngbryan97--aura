# Matched native fitting-source erasure

The fit-only erasure control completed all 320 updates, 1,952 frozen-prefix
captures, 185 intact-source calibration requests, and 50 held-wording decisions.
Calibration selected step 288; the intact-source reference selected step 128.
Neither checkpoint was chosen using held labels.

| Same source bank | Intact fitting source | Erased fitting source |
| --- | ---: | ---: |
| Correct selected programs / 50 | 46 | 46 |
| Gains over incumbent | 6 | 6 |
| Regressions against incumbent | 1 | 1 |
| Gains over unfitted native scorer | 7 | 7 |
| Regressions against unfitted scorer | 0 | 0 |

All 50 paired correctness outcomes agree: 46 common successes and four common
failures. Five selected program identities differ despite those outcome ties.
There are no discordant correctness pairs; the existing exact paired test
returns p = 1. This is a measured null for the incremental contribution of
fitting-source content on this bank, not proof that requests carry no meaning.
Both arms still receive intact calibration and held requests. Source length,
template position, and output-prefix correlations also remain available.

The separate CPU verifier reconstructed every erased and intact supervision
sequence, checked all checkpoint weights, independently re-executed both arms'
385 bank comparisons, and checked matching schedules, hyperparameters, source
partitions, proposal inventories, and baseline identities. Unknown comparisons
are excluded from paired scoring rather than counted as failures.

Control directory:
`~/.aura/rlc-evidence/semantic-native-source-erasure-v2-fold0-20260926/`.
Plan: `6f6ab3b64b07dfef7e5fea563c4d88abe37ff9bfd1b47ce88093a8a6065cd73d`.
Report: `10d473aeeed0755b0210cf9d7d45d7a4d8c86c8e9f516e53a2dbfeb02aa3f5c8`.
Paired independent receipt:
`b06f0aac8675b309885e8d8b80d75ce8cc9b5d1a23e449dbbd693a1b380051f7`.
Elapsed control time: 3128.794098792001 seconds.

The reference's historical trainer hash differs from the current trainer that
added the erasure mode. Its original independent receipt remains intact; the
paired receipt explicitly reports that drift rather than relabeling it green.
The control's bound implementation has no current drift at verification.

The gain over the unfitted model cannot now be attributed to learning source
meaning rather than output-wire adaptation and existing frozen-model semantics.
This prevents that stronger claim. It does not erase the bounded measured gains
or establish broad gain, fresh transfer, fusion, or serving authority. G03-G06
remain open.
