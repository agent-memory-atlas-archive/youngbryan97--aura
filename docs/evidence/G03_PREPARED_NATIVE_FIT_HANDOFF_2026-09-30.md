# Prepared native fit continuation, 2026-09-30

`tools/run_semantic_native_fit_handoff.py` advances the supervised native
preparation into training and independent fit verification. It removes only
the plan-only and supervision-only modes from the common prepared command.
The source bank, source inputs, schedule, objective, adapter sites, storage,
precision and numerical bounds remain unchanged. Relative input paths keep
their meaning in the original preparation checkout.

The continuation waits for successful preparation and proven process cleanup.
It then checks the actual plan, supervision and strict identifiability
preflight, plus each bound implementation file. Existing checkpoints or a
training report prevent another fit in the same directory. The existing
trainer still owns exclusive model-lane admission; this handoff cannot evict
an unrelated owner or bypass the memory envelope.

The detached broker admits two commands, each at most once: the actual fit,
then `verify_semantic_native_fit.py`. Failed fitting does not start verification.
The final receipt requires independently verified selected weights. Selection
of checkpoint zero remains a completed fitting experiment with no learned
candidate and returns a rejection status. Neither a completed fit nor a
positive-step checkpoint establishes generated correctness, transfer, broad
gain, promotion or serving authority.

A rehearsal against the current queued preparation plan found that its
receipts use `fit_identity`, while native plans use the JSON evidence digest.
The reader now checks each artifact with its own existing hash contract.
There is no fallback accepting another digest. Tests exercise this distinction,
changed protocols, wrong partitions and preflights, stale code, prior fitting
evidence, and ordered refusal after failed prerequisites.

The policy-only rehearsal succeeded against the actual
`semantic-native-preparation-v1-20260930` supervisor. No model weights were
loaded by that rehearsal. The affected continuation suites passed 94 tests;
smoke passed 164 tests with one skipped. Lint, compile, governance-lint,
layering, writing and doc-drift passed at their existing baselines.
The source fit has completed eight of eleven heads;
the source bank, native preparation, native fit, and generated validation
have not completed at this observation. G03 remains open.

## Detached continuation launch

The continuation was launched from a clean detached checkout at `6c683c98e6`:
`.claude/worktrees/codex-g03-native-fit-handoff-frozen`. Its evidence directory
is `/Users/bryan/.aura/rlc-evidence/semantic-native-fit-handoff-v1-20260930`.
The native output remains `semantic-native-stop-source-v1-20260930`.

Independent status inspection found supervisor PID 74759 alive with parent
PID 1, child PID 74762 alive, a moving heartbeat and wait log, and sleep
inhibitor PID 74787 bound to the child. The supervised plan hash is
`9a5d112021e227f98943825fb43176a197b935c77716aa87fad2412e242541fc`;
the command hash is
`1b9ee91e8df2a5801b642797dc870870932be4a41034bd1335154711bd97a13a`.
The supervisor owns a 70,000-second outer bound, two exact broker commands,
and no automatic restart or resume contract. It was waiting for preparation
at this observation; model training and verification have not run.
