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

## Completed generated canary

The frozen type-wise policy completed all fourteen retained development
requests without a forced completion or depth-bound exit. It recovered eight
exact procedures and eleven observed answers. Independent replay established
nine output-and-domain equivalences, four witnessed differences, and one
unknown equivalence. Numeric agreement on the other two answers is therefore
not a semantic proof. The scorer did not receive targets or target depth.

The first detached launch was refused before loading weights for insufficient
physical headroom. An orphaned test chunk at PID 63562 had lost its runner;
its log at `/Users/bryan/subject-core-runs/port_verify_0928.txt` was preserved
and that exact creation-time-checked process was stopped. The retry completed
in 150.77 seconds with empty process lineage. The subject campaign was not
stopped. No admission threshold changed.

Evidence under `/Users/bryan/.aura/rlc-evidence`:

- `semantic-native-factorized-retained-14-v2-20260928`
- Decode report: `9b7bf4461607422ee793e1cd94e9dbb471bd0b1fe4992b762be8b37de1709102`
- Independent replay: `3573941b6e0eaea5bbe3c3a6229cced2a70b3d5905b145d4e8e29ae3cd4eede9`
- Detached completion: `16c9958cc5ca2fa63cc52e90768c8bda11e7cd262a4c9795ac1b102819e657b2`

Independent replay measured zero implementation drift. A newly planned base
arm uses the same current implementation, source cohort, arithmetic and bounds;
its paired result is pending at this observation. The older base measurement
is retained but cannot replace this current-implementation comparison.

## Fit-only contrast inventory

The current training plan selects 261 explicit source contrasts, all of them
operation decisions. A CPU audit of its 303 fitting sources found 1,984
ordered source pairs whose first shared-grammar divergence is a reference
choice; 1,648 have an execution witness of different meaning. Reference
divergences cover 151 sources before the witness filter. No held labels or
model weights were used for this inventory. These contrasts are available
learning material, not evidence that training on them succeeds.

G03 and G04 remain open. The complete development matrix, paired regressions,
source interventions and fresh-family measurements are still required.

## Completed current-implementation pair

The matched unfitted arm completed all fourteen requests: one exact procedure,
three numeric answers and two proven meanings. Independent comparison finds
seven procedure gains, eight answer gains and seven new proven meanings, with
no losses on any of those three measures. Its comparison receipt is
`94a6b7b407f3c2e18068f1029b825acfd1d31f33b3f0bead53be2308cbba9e3a`
at `semantic-native-factorized-retained-comparison-v2-20260928.json`.
The base report receipt is
`d078169d96cad8ca7c9b7e0d194d68945120b2b6174c3c41d10ff3c1889c6805`.

This base is the unfitted native-format scorer, not ordinary natural-language
LLM reasoning. Wire-format adaptation remains a possible explanation; neither
broad reasoning gain nor general transfer is established by this pair.
The fifty-request development run uses a clean frozen measurement checkout
at revision `9db3c9cb3`, separate from ongoing objective implementation.

## Completed fifty-request development observation

The type-wise policy completed all fifty retained requests in 673.61 evaluator
seconds: 31 exact procedures, 35 numeric answers, zero forced completions and
two disconnected depth-bound exits. Independent replay establishes 33 proven
output-and-domain equivalences, fourteen witnessed differences, one unknown
and two unavailable meanings. Implementation drift is empty. This is not a
promotable candidate and no 500-request evaluation or fusion is authorized.

- Decode report: `ebf1fee9cbbf4944ef919debc80ee572df6b06a742c0bb32b7c9c1b790944ab2`
- Independent replay: `a15feb67e3910740035340c72f5f9578a2e3d4f3337dc9b04f96fb29c06007ad`
- Detached completion: `ee3c0e129aedaf45740d67c19e880b231d3d95145497ea86edb0098668cbc293`

The supervisor completed with empty process lineage after 679.74 seconds.
These receipts do not establish a fifty-request gain over a matched base;
only the separately completed fourteen-request pair supports that comparison.
