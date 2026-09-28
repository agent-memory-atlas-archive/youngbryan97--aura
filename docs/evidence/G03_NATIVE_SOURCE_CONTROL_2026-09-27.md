# Native source-content controls, 2026-09-27

The fitted native grammar was decoded on six target-blind, three-step operation
intervention requests (three answer-changing source pairs, seed 2718283). The
checkpoint, grammar, public inputs, population, and decode limits were fixed.
Only the scorer's source content changed between arms: intact text, same-length
token erasure, or the natural-language request from the paired source.

| Source supplied to scorer | Exact target programs | Correct public answers | Exact pairs |
| --- | ---: | ---: | ---: |
| Original request | 5/6 | 5/6 | 2/3 |
| Token-erased request | 1/6 | 1/6 | 0/3 |
| Paired request, graded against original | 0/6 | 0/6 | 0/3 |
| Paired request, graded against partner | 5/6 | 5/6 | 2/3 |

All six swapped programs equal the programs decoded with their partner's intact
source. The erasure comparison has four intact-only correct programs, no
erasure-only gains, and five changed programs. The intact-to-swap comparison
has six changed programs. The single incorrect partner decode is the same
premature finish after two steps on a count request; following that partner's
source reproduced its wrong program, not the requested three-step target.

The independent verifier reconstructed all source identities, choice sequences,
source masking, public inputs, executable programs, and answer totals. It
reported `current_implementation_drift=[]` for all three arms. The comparison
receipt is `9ab805070bbc77f2e31deed1e845bea46a722c26555433345adbc076146cf8d5`.
Evidence lives under `~/.aura/rlc-evidence/semantic-native-source-control-*v6-20260927`
and the intact arm under
`~/.aura/rlc-evidence/semantic-native-source-control-operation-full-v5-20260927`.
The detached erasure and swap supervisors both exited `passed`.

This is evidence that source content causally changes native choice on this
constructed cohort, not proof of general semantic correctness or transfer.
The six requests share known computational topologies, one member still fails,
and neither role/dependency binding nor held-out families are measured here.
The fitted checkpoint's 261 contrastive source-pair updates all target
operation choices; none target reference or finish/continue decisions. Ordinary
grammar-choice supervision includes those decisions, but source-pair coverage
does not. Role/dependency controls and a depth contrast are the next tests.
G03 remains open; this evidence grants no serving or fusion authority.
