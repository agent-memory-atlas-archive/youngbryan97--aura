# Native supervision identifiability, 2026-09-26

The full grammar-choice epoch has a CPU-only audit of one possible artificial
blocker: identical causal scorer inputs requiring different exact teacher
decisions. This audit does not change its frozen producer, objective, source
partition, numerical allowance, or model ownership.

## Scoring Inputs

The verifier groups the actual supervision by source and decision, validates
complete choice indices and a unique valid teacher index, then fingerprints the
ordered alternatives' causal token prefixes and scored target positions.
Source identities, target-program digests, and tokens after the last scored
target do not distinguish scoring inputs. Those future tokens cannot affect
the causal logits being measured.

If one identical input group demands multiple teacher indices, a deterministic
scorer can satisfy at most the most frequent index in that group. Summing those
maxima supplies an exact-label upper bound for this particular collision class.
It is not a general learnability bound: reordered alternatives, representational
capacity, optimization, source ambiguity, and distribution shift require other
checks. Exact teacher labels also need not exhaust semantically equivalent
programs.

## Measured Population

The frozen epoch supervision contains 4,416 decision groups across 488 sources:
1,104 operations, 2,208 references, and 1,104 termination decisions. Of these,
273 have only one admitted choice. All 4,416 scoring fingerprints are distinct;
this audit finds no contradictory identical-input groups. It therefore detects
no reduction of the exact-label ceiling from this collision class. It does not
prove that the resident model can learn all decisions or transfer them.

The durable audit is `identifiability-audit.json` under
`semantic-native-grammar-choice-fp32-trie-epoch-shards-v1-20260926` in the local
RLC evidence root. Its bindings are:

- Plan: `2ccc7d5f23dae9f99827e56003479619c52bff23417c5b597335be3ece19fa6d`.
- Supervision: `1915ff910dec59bfce7fd333e7c5c7f27c7e8a9114ca324edf2370f3f4c81da9`.
- Audit: `31e6a7d0299256f112704248e1cf9dc73f13738617f86c4e05b54839d93facf5`.

## Integration And Boundary

The independent native-fit verifier now includes this audit after reconstructing
every grammar choice and source token. Fourteen new tests cover contradictory
labels, consistent duplicate inputs, causal context, unscored future tokens,
scoring positions, and malformed groups. Ten existing paired-fit tests and
sixteen CPU-only supervision reconstruction tests also pass.

The model epoch is still running at this observation. No fitting accuracy,
fresh transfer, serving, fusion, broad reasoning, or G03 closure follows.
