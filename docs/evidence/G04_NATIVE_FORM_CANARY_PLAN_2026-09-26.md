# Native source-form canary plan

Two six-row development canaries change the source expression while preserving
the executable programs and public values in the September 26 operation pairs.
One uses named definitions; the other uses equation-style expressions. Each
contains scalar, lookup, and count pairs whose final operation changes while
the operands and public inputs remain fixed. These are new expression forms,
not new underlying task identities or new domains.

The v4 evaluator scores native operations and arguments from the source without
receiving a target graph, answer, construction label, or candidate bank. It
recovers input values and types from literal source text. A program-first
renderer supplies the independent grading annotations, including argument and
definition spans. The verifier reconstructs the population separately from
the evaluator's ordering logic and replays the saved score decisions.

- Definition plan: `34601102cce563c60df79444be2ab990aab82f86cfaff1a44d4a1989a52b5e4b`.
- Equation plan: `a83adcbcab46812cfcd89f18630aff82c4b9760f57dc37c41f7ba12f0005e0a3`.
- Seed: `2718283`; population: six rows per form; grammar bound: three steps.
- Checkpoint receipt: `d58d34d4dfef264a2e7cd0374b5ea00a9892c9a70d0f6e49cae4e622bdd899a2`.
- Artifacts: `~/.aura/rlc-evidence/semantic-native-definition-intervention-v4-canary-20260926/`
  and `~/.aura/rlc-evidence/semantic-native-equation-intervention-v4-canary-20260926/`.

Before model scoring, all 24 pairs in each form passed program preservation,
distinct outcomes, source-grounded public-value recovery, and noncommutative
argument attribution. Twenty-seven focused tests passed. The definition
canary was checked against the exact fitted source report and eight bound
manifests: 1,764 source examples, 105 constructions, and zero text,
construction-ID, or topology-ID overlap. Disjoint identifiers do not establish
that every linguistic feature is unfamiliar to the model.

The plans were frozen before these form-specific model scores, but their design
uses earlier development results. Neither canary is powered confirmation.
Both supply the known three-step bound. Matched wire-learning controls, fresh
task identities, broader forms, uncertainty, and public/live translation remain
separate obligations. This plan closes no G item.
