# V7 source-calibration scale ceiling, 2026-09-29

This is a CPU analysis of the six-scale, 185-source fit-matched residual sweep
under `/Users/bryan/.aura/rlc-evidence/semantic-native-joint-graph-v7-residual-step101-v4-fitmatched-20260929`.
Its independent verification reports `artifacts_verified=true`,
`current_implementation_drift=[]`, and report receipt
`8c943c106246214707d1e1c8aa2d699050a802ea4961c765fc7ef65306a95adb`.
The source, decision order, choices, and teacher index agree across all six
scales for each of the 185 sources. This is calibration evidence, not a
generated or held-out result.

| Teacher-forced calibration criterion | Exact sources |
| --- | ---: |
| Frozen selected scale 0.5 | 112/185 |
| Oracle chooses one of the six measured scales per source | 120/185 |
| Oracle chooses one of the six measured scales per decision | 126/185 |

For the per-source oracle, a source counts when at least one complete saved
scale path is exact. For the per-decision oracle, it counts only when every
teacher-forced decision has a correct top-scoring choice at at least one scale.
These are optimistic ceilings for **this measured six-scale family**, not
achievable selectors or limits on a different representation. Neither oracle
may inspect a target in deployment.

Across 1,736 aligned decisions, 90 have no correct top-ranked choice at any
tested scale: 78 reference, 11 operation, and one termination. They affect
59 sources. The median best margin among the 78 reference misses is -1.122
log-score units; 44 remain at or below -1. Thus most residual misses are not
fixed by choosing a better global scale, and many are not near-tie numerical
effects. In contrast, an oracle over the six whole-graph rankings reaches
184/185; graph selection and reference binding are distinct bottlenecks.
At selected scale 0.5, reference has the largest saved conditional
decision-or-graph loss in 147/185 sources. This comparison omits the typed
source-pair interaction losses in the training objective and is not a gradient
attribution. It does not support simply removing the graph objective as a
reference-binding repair.

The next model change should increase source-conditioned reference evidence
or the learned representation of role binding, then test whole-path effects.
Selecting alpha by source or decision alone cannot clear the 59 sources whose
correct teacher choice never wins in this measured family. This audit grants
no G03 pass, no causal general-transfer claim, and no serving authority.
