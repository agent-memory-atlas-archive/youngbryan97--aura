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
`/Users/bryan/.aura/rlc-evidence/semantic-native-role-rule-plan-only-v3-20260929/plan.json`
has plan SHA-256
`08bec2b616342db926a1382b397a20b4328a26ef1b082cf4ad923f54e4c6ea56`.
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
  erasure, malformed receipt structure, and receipt ancestry. `make smoke`
  passed 164 tests with one skipped before the final named-reference and
  malformed-receipt test additions; 35 focused tests passed afterward.

The 2048 demo suggested a useful design pattern: observe an unknown rule under
interventions, retain only an explanation that predicts across situations, and
use the explanation to guide later choices. This implementation transfers that
pattern to a bounded language role permutation. It does not inherit the game's
full state-transition learner or demonstrate open-ended program induction.

## Promotion boundary

The rule only changes first-step reference scores when one supported local
binding is unambiguous. All other scores stay with the model. The pending
model-active test must establish generated procedure and answer gains against
an unchanged baseline, source-erasure and target-blind relation controls, and
no lost baseline-exact procedures. The current detached residual campaign is
a different frozen candidate; its results must not be assigned to this plan.
The 500-request development bank and fresh-family controls remain substantive
tests, not formalities inferred from these small probes.
