# Role-relative native fit

The single-row role-relative fit completed 320 source-only updates. It captured
all 303 eligible fit sources and all 185 source calibration cases. Calibration
alone selected step 128. The 50 held-wording rows were graded after selection.

| Same frozen proposal bank | Correct / 50 |
| --- | ---: |
| Existing proposer selection | 41 |
| Unfitted native scorer | 39 |
| Role-relative fitted scorer | 46 |
| Correct proposal observed anywhere in bank | 47 |

The fitted scorer gains six cases over the incumbent and regresses one. It
gains seven over its unfitted control with no paired losses. Three nominal
nested-arithmetic cases have no observed correct proposal. The remaining miss
is a reachable sequential-arithmetic program: the incumbent selects the
correct `idiv(0,1); sub(2,result:0)` graph, while the fitted scorer selects
`idiv(2,1); sub(0,result:0)`. Both are typed and executable. Their difference
is source-to-argument binding, not serialization or arithmetic execution.

Directory:
`~/.aura/rlc-evidence/semantic-native-relative-metric-single-fold0-20260926/`.
Plan: `b742298c544c2a0f38e4a47513915a2745f0f193053b5a3f75130351dbde2150`.
Selected checkpoint: `d58d34d4dfef264a2e7cd0374b5ea00a9892c9a70d0f6e49cae4e622bdd899a2`.
Fit report: `521a286ff2b66768b7560173b844b9664c34fe048f779f5410299b0ed62ea7db`.
Elapsed: 3853.5425925840027 seconds.

A separate CPU verifier recomputed every receipt and checkpoint digest,
re-executed all 385 unique held-bank program comparisons against source
features, checked proposal scoring and selection, and recomputed all paired
counts. It found 81 equivalent, 304 different, and zero unknown comparisons;
current bound implementation drift is empty. The independent verification
receipt is `independent-verification.json` in the fit directory.

This is an exposed held-wording development comparison. It is not evidence
of fresh-family transfer, target-blind proposal reach, freely decoded public
gain, or broad reasoning gain. The single regression prevents unconditional
promotion. The role-relative wire and full calibration also differ from the
older fold-0 protocol, so this comparison alone cannot attribute the gain to
the coordinate change. Serving and fusion are unchanged; G03-G05 remain open.
