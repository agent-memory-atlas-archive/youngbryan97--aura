# G03 fit-only termination contrast, 2026-09-29

The v1 fork/join counterfactual cohort supplied witnessed operation and
reference contrasts, but no stop/continue contrast. That absence matters to
the native grammar's termination choice; it does not by itself explain a
particular generated error. The completed retained fitted arm of the v7
six-source micro probe returned 4/6 correct public answers and 3/6
structurally equivalent programs. Its paired base arm was not complete when
this acquisition change was made, so there is no paired gain verdict here.

`counterfactual_fork_join_stop_source_v1` is a new fit-only corpus. It keeps
the deterministic first source in each of v1's 648 construction/topology/
operation cells and renders one four-step continuation for each three-step
graph. The continuation consumes the same four public inputs, keeps the first
three operations and arguments, and adds the former terminal result to one
existing input. The builder requires a witnessed output change and unique
source identities. The selection path refuses any cap below all 1,296 paired
sources; the generic one-per-cell minimum is only 1,224 because some
continuations share a cell, so that weaker minimum would silently drop pairs.

The selected example-ID list has SHA-256
`e5173fafc0bc9701dbb8e16a7bd4310c8a373ae6ed0eec1a9af6db46774d6ed1`.
The active 27B tokenizer from the
[`real-tokenizer audit`](G03_REAL_TOKENIZER_PAIR_AUDIT_2026-09-29.md), without
loading model weights, projected every source annotation. Maximum source
lengths were 125 tokens for depth three and 144 for depth four. On these
actual token IDs the fit-only typed planner measured 1,296 paired sources,
with 1,296 sources by each of operation, reference, and termination. The
first post-prefix stop/continue decision is paired in all 1,296 sources.
No calibration or held label supplies a peer or execution witness.

The CPU-only audit and 87 focused tests establish the training input and
planner contract, not learned termination, native generation, non-regression,
held-family transfer, serving authority, fusion, or G03 closure. Acquiring
hidden states, fitting, and independent generated comparisons still require
the resident-model lane after the current campaign releases it.
