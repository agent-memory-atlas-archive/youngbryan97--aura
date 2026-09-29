# G03 native budget frontier, 2026-09-29

The full-prefix residual reference arm generated three exact procedures and
correct answers. Its independently verified rows contain every decision score
and complete-graph score used by the 256-node best-first search. The
`replay_semantic_native_budget_frontier.py` diagnostic re-executes that same
search at smaller node bounds using only a prefix of those saved model scores.
It refuses a changed choice inventory or proposal order and reproduces the
measured full-budget program and verdict. No target enters the replay scorer;
target programs and public values grade the selected graph afterward.

The receipt is
`/Users/bryan/.aura/rlc-evidence/semantic-native-v7-reference-fitted-budget-frontier-v2-20260929.json`
(`91bb9f4510d579d1a864d9af742e0422c62cfc52f0450f724551bedcc4ccb8c7`).

| Node bound | Exact procedures | Correct answers | Scored decisions | Scored alternatives |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 3/3 | 3/3 | 45 | 255 |
| 32 | 3/3 | 3/3 | 93 | 495 |
| 64 | 3/3 | 3/3 | 179 | 963 |
| 128 | 3/3 | 3/3 | 206 | 1,100 |
| 256 | 3/3 | 3/3 | 206 | 1,100 |

At 16 nodes, each request retained one correct completed graph; requested
top-four proof was false. The 16-node replay scored 76.8% fewer alternatives
than the full arm. This is a compute-count reduction, not a measured wall-time
speedup or a prospective model run. The replay records a historical grammar
file-byte difference, but every full-budget source decision and verdict
reconstructs from the saved transcript; the search implementation bytes match.

One completed baseline row was also inspected locally before its full
three-request verification: it had no candidate at 16 nodes, then an exact
program at 32 and 256. That incomplete baseline observation is not a paired
cohort result. The frozen 256-node campaign remains the promotion comparison.
A separate prospectively frozen 16-node screen can reject weak later candidates
quickly; success there would advance to, not replace, the full matched search
and held controls. G03, G04, general transfer, fusion, and serving remain open.

## Paired baseline replay after its verified completion

The baseline arm subsequently completed and passed independent verification:
one exact procedure and public answer among three at 256 nodes. Its saved
scores reconstructed all three full-budget outputs under the same search code.
The baseline budget receipt is
`/Users/bryan/.aura/rlc-evidence/semantic-native-v7-reference-base-budget-frontier-20260929.json`
(`f15704edba9f1064124036e179500e92cc27ea2dbd2c4500a8af7f4a83f9218c`).

| Node bound | Fitted exact and correct | Base exact and correct | Fitted alternatives | Base alternatives |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 3/3 | 0/3 | 255 | 162 |
| 32 | 3/3 | 1/3 | 495 | 342 |
| 64 | 3/3 | 1/3 | 963 | 1,009 |
| 128 | 3/3 | 1/3 | 1,100 | 2,322 |
| 256 | 3/3 | 1/3 | 1,100 | 4,108 |

At 256 nodes, the baseline reaches the first exact graph and misses the
second and third. At 16 it reaches none; all three fitted graphs are exact.
Both arms have the same historical grammar file-byte difference in the replay
environment, while their full-budget decisions and outcomes reconstruct from
the saved scores. This is bounded paired development evidence. The source-
erasure and relation-control outcomes are still pending, so no mechanism or
general-transfer claim follows from this table.
