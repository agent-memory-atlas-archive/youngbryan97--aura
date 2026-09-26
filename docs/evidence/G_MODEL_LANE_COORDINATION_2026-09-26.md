# One model-memory ledger for live-profile research

The target-blind native grammar run isolated its evidence under
`AURA_STATE_ROOT`. Its `standalone_model_lane` reservation consequently went
to the run's private `state/run/model_lane_control.json`, while Aura's resident
owner used `~/.aura/run/model_lane_control.json`. The private record showed
only the research process as an owner. Host headroom still guarded admission,
but the two processes did not share ownership or preemption state.

The default lane path now uses the live state root for live-profile processes
even when their ordinary state root is isolated. Explicit
`AURA_MODEL_LANE_STATE_PATH` overrides remain authoritative. Test, bench, and
development profiles retain their private defaults. A process probe after the
change resolved an isolated research state root and the shared lane path in
the same process. Seventy-one focused lane tests and smoke (164 passed, one
skipped) passed; lint, compile, layering, governance lint, and writing passed.

The 72-source run was already admitted before this repair, so its existing
lease is not retroactively moved. Aura separately received SIGTERM at
11:48:43 local time and completed graceful state-preserving shutdown by
11:48:45. Its logs do not identify the sender. This record does not attribute
that shutdown to the lane split. Explicitly isolated model-lane paths in
contained launchers still require a separate ownership audit.
