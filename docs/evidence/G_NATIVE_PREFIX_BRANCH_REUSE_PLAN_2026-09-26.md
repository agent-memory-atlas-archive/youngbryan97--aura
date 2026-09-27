# Frozen native prefix branches, 2026-09-26

The native evaluator executes the same public request through the frozen
decoder prefix for every operation, reference, and termination alternative.
This repeats work without adding evidence. Reusing a causal source anchor can
remove that repetition, but changing execution shape can also change numerical
results on the hybrid resident model. Earlier grouped capture failed the
single-row equivalence threshold and was rejected.

## Mechanism

`FrozenPrefixBranches` uses the loaded decoder's own cache factory and Aura's
existing layer-mask contract. It evaluates the source/template anchor once,
then gives each continuation a deep-copied cache, as the runtime prompt cache
does. Returned hidden sequences contain both the full anchor and the continuation.
Every suffix layer and supervised vocabulary projection still runs. Token
identity, frozen parameters, evaluation mode, cache inventory, and sequence
bounds are checked. No prompt, source, or native wire is modified.

The anchor ends before the earliest offset-bound assistant continuation.
Branch alternatives must share those exact tokens. Distinct source text cannot
silently use a previous request's cache. Branch receipts count both executed
and avoided token positions; they contain no hidden states or private reasoning.

## Admission and Measurement

Thirty-one focused MLX tests pass. They exercise dense and hybrid layers, both
attention and recurrent cache kinds in the same prefix, tied and untied output
projections, four-bit quantization, repeated branches, anchor-only requests,
invalid token sequences, source drift, prefix training, and returned-state
ownership. Causal states and
native logits agree with full-sequence single-row inference on those models.

The resident probe binds the selected checkpoint and reconstructs its retained
source alternatives. It measures both execution paths on the same model, using
the existing allowance of two output-dtype epsilon units per supervised target.
Every alternative's rank must be identical, not merely its winning choice.
Any larger target-log-probability difference or rank change rejects reuse.
Neither tolerance nor a speed result grants semantic or serving authority.

The CPU-only plan is at
`~/.aura/rlc-evidence/semantic-native-prefix-branch-probe-v2-20260926/plan.json`,
digest `04b4af5630f64f2f031aa0323fdd46228cbf2ec26053b1d6961c2c463b050a64`.
The three-source resident measurement is pending. The evaluator and serving
defaults are unchanged until equivalence is actually measured.
