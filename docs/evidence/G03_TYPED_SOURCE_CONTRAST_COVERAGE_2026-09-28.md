# Typed source contrast coverage

## Missing learning signal

The previous lineage-only objective selects 261 source contrasts, all at
operation decisions. Its fitting corpus contains witnessed reference
contrasts that this selection cannot admit. Conditional operation improvement
therefore does not show that the source-by-reference interaction was learned.

The new opt-in `typed_choice_complete_v1` policy selects at most one peer per
source and decision kind. Peers must belong to the complete fitting partition,
share input types and the same grammar prefix before their first differing
choice, and have a witnessed difference in program meaning. Equivalent
commutative rebindings and unknown comparisons cannot be negatives. Selection
prefers the existing contrast lineage, then token overlap and source hash.
Those selection fields are not features presented to the native scorer.

Every ordinary conditional path loss and each admitted source-by-choice
interaction enter the same log-mean-exponential objective. The interaction
`S(x,i)-S(x,j)-S(x',i)+S(x',j)` cancels additive source-only and choice-only
bias. That algebra does not guarantee that the learned interaction represents
the intended relation or transfers to a new family.

## Frozen preparation

The v6 plan at
`/Users/bryan/.aura/rlc-evidence/semantic-native-typed-contrasts-fp32-full-v6-20260928`
has 303 fitting sources, 185 source-calibration sources and fifty development
bank requests. Its complete 303-update schedule visits each fitting source
once. Selected contrasts cover 303 operation sources and 135 reference
sources. Termination coverage is zero, not silently declared measured.

Plan: `2fec4af45c0b0805d3bf55183088b8aa4478a8c3fabbce4043de8e7f1b2fe59f`.
Supervision: `e690fa6821f6894e9f3357a39477dc7b101894c89cbd18e60b6cea72332f5aa2`.

All 21,733 sequences match the prior bound frozen capture, covering 488
fitting and calibration sources. Numeric prefix behavior, model identity,
tokenization and output supervision remain unchanged. Reuse does not inherit
optimizer state or old outcome evidence. The source archive proves only
annotation erasure on its enumerated numeric paths; it does not waive other
implementation changes.

## Checks and authority

The v6 reader and independent fit verifier require the typed inventory and
reconstruct partners from the fitting corpus. Historical v1-v5 plans cannot
acquire typed-coverage authority. The existing baseline-preserving complete
source-path selection remains in force; held outcomes cannot choose a
checkpoint. The missing-coverage invariant is registered with its owner.

Focused and consumer checks: 187 passed, two tokenizer-dependent checks
skipped. Smoke: 164 passed, one skipped. Lint, compile, governance, layering,
writing and document-reference gates pass. CPU preparation loads no model
weights. Model training and generated validation remain unmeasured at this
implementation checkpoint. No serving, fusion, G03 or G04 closure follows.
