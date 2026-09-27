# Native grammar-choice training

## Failure boundary

The retained fourteen-request canary produced ten requested procedures and
eleven correct sampled answers. Two graphs reached the depth bound while
disconnected. The first-operation and termination scores in the retained
receipts locate failures inside the native decoder's own decisions.

The previous fit optimized token likelihood over complete programs, with
four witnessed contrasting graphs per source. Runtime decoding instead
compares every type-admitted operation, reference, and termination choice
at each partial graph. The new opt-in objective trains those competitions
directly. It does not change the request or the decoder's grammar.

## Mechanism

For source target decision `y` and the decoder's admitted set `A`, let
`s(a)` be the sum of native token log probabilities over the decision span.
The objective is `logsumexp(s(A)) - s(y)`, averaged over the source's
decisions. Its score gradient is `softmax(s(A)) - one_hot(y)`. A choice with
only one admitted alternative has zero loss and zero gradient.

Teacher forcing traverses `decode_native_grammar` itself. It retains every
alternative encountered along the requested procedure, including the final
finish/continue competition. It never uses a forced depth completion as
supervised stopping. Input types and register order come from the public
source; a target whose input order differs is refused.

The positive denotes the requested procedure. An alternative may compute
an equivalent result; its exclusion from this exact procedure does not
certify a false meaning. Output/domain equivalence remains a separate
grading claim.

Fit plans and reports use version three. Historical version-one and
version-two evidence cannot acquire this objective. The existing replay
and grammar evaluation loaders accept a source-selected version-three
checkpoint. Independent verification rebuilds every alternative, target
index, decision span, and token from the retained source corpus. Fit-only
source erasure remains available under its explicit control contract.

## Bounded preflight

The canary keeps the original source-hash shuffled update schedule but
performs eight updates. Calibration and held-bank subsets are limited to
four source hashes each, before scores or outcomes are observed. These
bounds are recorded in the plan; they cannot support full-population
qualification. Checkpoint zero remains eligible.

Directory: `semantic-native-grammar-choice-canary-v3-20260926`.

- Plan: `174049278f02df77faf58939cc2cf652084e1448da23815dde14611af36e0212`.
- Supervision: `6d7f8231c4c3fe6ad0ce5ee598efe99c1c415709fdc2ae24721ecc7247fa539c`.
- 519 prefix sequences, lengths 97 through 193 tokens.
- 108 decisions: 27 operations, 54 references, 27 termination decisions.
- Tokenization/materialization: 36.53 seconds, without model weights.

The focused suites pass 85 tests with two optional tests skipped. A
separate run with the resident tokenizer passes 52 tokenizer and real-MLX
prefix tests. The new source loss changes the inference scores in the
real-MLX gradient test. No resident-checkpoint fit outcome has been recorded
at this checkpoint. No promotion, fusion, serving change, or G-ledger
closure follows from these implementation tests.
