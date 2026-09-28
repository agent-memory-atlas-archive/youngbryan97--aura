# Native parameter isolation by decision type

G03 and G04 remain open. This path has no serving or qualification authority.

The v5 fit improves operation choices while losing reference bindings. A
single residual scale suppresses both effects: its generated development
canary has zero correctness gains. The new opt-in path isolates the adapter
scale used for each of the existing grammar's three decision types. It reads
operation atoms, register identities, and finish/continue atoms. It receives
no source ID, construction label, expected program, or answer for routing.

Selection uses the already measured scores at scales 0, 0.08, and 1 on all
185 source-calibration requests. It enumerates all 27 type-wise combinations;
no score interpolation is used. A combination is eligible only if it preserves
every exact baseline path. Selection then maximizes exact paths, minimizes
conditional loss, and breaks ties by total scale and declared type order.
The independent CPU checker reconstructs every combination from the bound
source rows and checks source coverage, complete supervision, and receipts.

| Scoring path | Exact source paths / 185 | Lost baseline paths |
| --- | ---: | ---: |
| Base | 51 | 0 |
| Global scale 0.08 | 60 | 0 |
| Global scale 1 | 104 | 12 |
| Selected type-wise scales | 120 | 0 |

The selected scales are operation 1, reference 0.08, and termination 1.
They give 426/434 correct operation choices, 775/868 reference choices, and
434/434 termination choices. Sixty-nine previously inexact source paths
become exact. These are conditional teacher-path measurements; correctness
on generated paths or unseen families is not established.

Runtime scoring applies the declared scale to every suffix LoRA site before
the competition, evaluates its scores, and retains a type/scale/site receipt
for every decision. The frozen prefix and model identity remain shared.
There is no second model or separate family router. The independent decode
verifier reconstructs each grammar competition and rejects missing or changed
parameter receipts. Mixed grammar types fail the registered invariant.

The factorized source receipt is
`803a00b8e7c17122114b7e09e608652d05bba0b5717b54d63502ff0935ded435`,
under `semantic-native-factorized-residual-v2-20260928` in
`/Users/bryan/.aura/rlc-evidence`. The earlier v1 preparation preceded the
scoped invariant-import repair and is not used for model evaluation.

Focused checks: 88 passed; the scoped dependency repair and its consumer
checks: 68 passed. Smoke: 164 passed, one skipped. Lint, compile, governance,
layering, writing, and document-reference gates pass after the scoped imports
were repaired. Target-blind generated measurement remains pending at
this checkpoint. No broad 500-request run, promotion, or fusion follows from
the source-calibration result.
