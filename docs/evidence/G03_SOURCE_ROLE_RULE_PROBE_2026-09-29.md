# G03 source-role rule probe, 2026-09-29

This is an opt-in first-instruction reference prior for the existing native
semantic decoder. It is not a second language substrate, a serving policy, or
G03 closure. The role rule fitter reconstructs the native v7 plan's exact 303
fit sources from its eight pinned feature manifests, excludes calibration and
held source identities, and retains only consistent relations observed in at
least two construction groups. Runtime evidence contains operation and argument
spans, their local relation frame, and a role-to-surface permutation. Public
literal values identify source-local names, but values, absolute positions,
construction labels, target programs, and answer correctness do not enter the
learned frame or the decision adjustment.

The fit receipt is
`/Users/bryan/.aura/rlc-evidence/semantic-native-role-rule-fit-v4-20260929.json`
(`f91c1c909f414ade875c5995bd57b90741eee176cdf9ebb0f560bfca0f7c6b35`).
It contains 670 fit observations and seven consistent rules. The source-only
plan at
`/Users/bryan/.aura/rlc-evidence/semantic-native-role-rule-plan-only-v4-20260929/plan.json`
has plan SHA-256
`34a6b13bdc39510dd229fbfd32c07caae3daa93df555bcdb13c28a4f32c89604`.
It is plan-only: no model-generated answer has been measured with this prior.

## Small probes

- On 72 `natural_request` development sources (seed 3141592), the fitted bank
  proposed the annotated first operation's two input references correctly in
  72 cases, with zero wrong and zero abstentions. This compares source-only
  first-role proposals after generation against annotations; it is not full
  program recovery or a model result.
- On all nine `relation_transfer_controls` sources at that seed, it abstained.
  Their nominal operation forms, including `sum of`, were not among the fit
  operation forms. This is a concrete zero-coverage result for those unseen
  constructions, not a pass. The source-only rule cannot establish G04.
- In the synthetic held-construction corpus, first-input roles were 544
  correct, zero wrong, 96 abstentions. These examples are generated from the
  same corpus machinery and are weaker evidence than an independent bank.
- The focused native-decoder tests exercise subtraction's operand reversal,
  source-local named inputs, relative references, ambiguity abstention, source
  erasure, malformed receipt structure, and receipt ancestry. The current
  `make smoke` passed 164 tests with one skipped; 36 focused role/replay tests
  passed after the independent v14 verifier was added.

The 2048 demo suggested a useful design pattern: observe an unknown rule under
interventions, retain only an explanation that predicts across situations, and
use the explanation to guide later choices. This implementation transfers that
pattern to a bounded language role permutation. It does not inherit the game's
full state-transition learner or demonstrate open-ended program induction.

The source-only preflight at
`/Users/bryan/.aura/rlc-evidence/semantic-native-role-rule-coverage-v1-20260929.json`
(`4013f7e5379a93728bf084a62072a4d40d9e3602cb24ad6105f8f227ceba4098`)
reconstructed three frozen cohort plans in 2.1 seconds. It measured 3/3
correct first bindings in the reference cohort, zero correct and nine
abstentions in the relation-control cohort, and 3/6 correct with three
abstentions in the retained cohort. This candidate therefore has no observed
role coverage on the nine-case transfer-control surface. Do not spend a
model-active decode to promote this rule alone. The preflight uses target
annotations only for its post-fit diagnostic and grants no scorer access to
them.

The saved-score calibration probe at
`/Users/bryan/.aura/rlc-evidence/semantic-native-role-decision-impact-v1-20260929.json`
(`81cd549fff17229630519c324bc3ff076d415a5079b1224bd9f3b5eccd59f287`)
independently verifies the historical 185-source residual calibration and
reconstructs the same source population. At the predeclared role strength 1,
the rule changes first-step reference competitions in 118 sources, gives two
new exact teacher paths, and loses zero exact teacher paths. This replays
saved conditional scores, not a generated search. Four historical training
files differ from their fit-time bytes; the probe records that drift rather
than claiming a fresh score measurement. Combined with zero coverage on the
nine relation controls, this is a measured local gain but still a no-go for
another model-active role-rule decode.

## Promotion boundary

The rule only changes first-step reference scores when one supported local
binding is unambiguous. All other scores stay with the model. Its v14
independent verifier now checks fit ancestry and recomputes every saved
adjustment from the scored source, including source erasure. A future
model-active test must establish generated procedure and answer gains against
an unchanged baseline, source-erasure and target-blind relation controls, and
no lost baseline-exact procedures. The current detached residual campaign is
a different frozen candidate; its results must not be assigned to this plan.
The 500-request development bank and fresh-family controls remain substantive
tests, not formalities inferred from these small probes.
