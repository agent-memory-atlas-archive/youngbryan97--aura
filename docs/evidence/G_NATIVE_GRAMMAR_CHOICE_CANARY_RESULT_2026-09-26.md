# Grammar-choice canary: fit verified, generated answer regressed

The eight-update canary completed in 863.0386184170056 seconds. It selected
checkpoint four on four source-hash calibration requests: conditional loss
0.2554021403193474, against 0.3073514308780432 at checkpoint zero. The four
held-bank requests score 2/4 for the fitted decoder, the unfitted decoder, and
the incumbent. There are no held-bank gains or regressions in this small set.

Independent CPU verification rebuilt all 519 sequences and 108 decisions from
the bound training corpus. It regraded all 35 held-bank proposals: eight proven
equivalent and 27 witnessed different. The fit and its artifacts are complete;
they do not establish a better generated program or general transfer.

Directory: `semantic-native-grammar-choice-canary-v3-20260926`.

- Plan: `174049278f02df77faf58939cc2cf652084e1448da23815dde14611af36e0212`.
- Checkpoint four: `2b1017ee5d7427aed71a58503b7cb2e1aa03623a9528ecfb53c1db1950693467`.
- Report: `624e6b4412b3476093f6fbc3545d72a3dc96eaf1ca2402832c20c91a6635cd69`.
- Independent verification: `d8debf567bb97b5a67e62410137d979fc565e6daa043c808d738bb2c7aaa9cd0`.

## Generation rejection

The same source-hash ordered, fourteen-request retained development canary was
started with this checkpoint. Its first completed request produced
`add(add(in1, in2), add(in1, in2))`. The previously measured checkpoint produced
`add(add(in1, in2), in0)`, which the independent floor comparison proves
output/domain equivalent to the requested procedure. The new result is
witnessed different and fails the observed answer as well.

The owned replay process was interrupted after this regression. Only one of
fourteen requests completed; the next request had not produced a durable row.
There is no completed replay report or full-population accuracy. The independent
early-rejection receipt checks the plan, public source coordinates, both saved
programs, all recorded greedy decisions, and the proof/witness distinction.

Directory: `semantic-native-grammar-choice-retained-canary-v3-20260926`.

- Plan: `2a732eb14e127617a9ed3f04419efad58e24dad35d77c03289022229738d0aa5`.
- Partial rejection: `cbb313fb91c32a09a4a2af2748fbdeb2ba440f436ef7629a4c027c521d11b277`.
- Source: `005ccc967d8b4588f4d2b58e66839da796a049d0fe6d8ba7ea9ebcd30556789a`.

The manual stop was not a preregistered stopping rule. The two checkpoints also
have different objectives and training exposure: eight updates here versus 320
in the older fit. This is a candidate rejection, not a controlled estimate of
the grammar-choice objective's effect. The objective remains a usable training
mechanism; this checkpoint is not promoted. G03, G04, G05, and G06 remain open.
