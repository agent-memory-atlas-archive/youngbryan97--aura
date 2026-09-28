# Model-lane handoff on 28 September

The role-guided base probe first failed before loading a model because
the reports campaign owned the exclusive lane. Its detached failure
receipt remains in
`semantic-native-role-guided-base-six-v1-20260928/detached/` under
`~/.aura/rlc-evidence/`. No model outcome follows from that attempt.

The competing command was `tools/run_report_grounding.py --whole
--anchors 24 --rounds 8 --seed 7`, running at `b8b004b72` in
`.claude/worktrees/reports-0928`. Parent PID 81425 and its registered
cortex worker PID 81440 were checked against process creation times and
the model-lane registry. The campaign reached anchor 21/24 at 08:16:31.
Its later log repeatedly reported 94.5% memory pressure, 3.5 GB available,
primary-lane downgrade, empty fallbacks, and deferred model admission
under `critical_thermal_pressure_3`. No later anchor was recorded before
the interruption.

Bryan had authorized this task to take the lane. SIGINT to the exact
parent did not end it. SIGTERM to the verified wrapper, parent, and
descendants ended PIDs 79573, 81425, 81427, 81438, 81440, 81981, and
81982. The process check found no survivor. Stopping the wrapper also
prevented its unconditional `DONE` write from marking an interrupted
run complete. The shared model registry then measured zero owners and
zero committed GB before the second RLC attempt was launched.

The original reports log remains at
`/Users/bryan/subject-core-runs/reports-s7-steering-b8b004b72.log`.
That run is interrupted, not complete or scientifically adjudicated.
Its runner retained arm results in memory until its final export, so
this handoff does not claim a saved or resumable 21-anchor result.
The already-paused load governor was not resumed into the RLC probe.

## Later admissions on the same date

The retained fitted canary also refused its first launch before model load.
The competing `reports-s7-steering-2` lineage was checked: wrapper 1948,
parent 2984 (creation time 1790609485.950774), and descendants 2986, 2992,
3000, and 3352. Authorized SIGTERM ended exactly that lineage; no survivor
remained. The second retained attempt then acquired the exclusive lane.
The interrupted report logs and failed RLC admission remain on disk.

Before the calibration audit, a new `reports-s7-steering-4` wrapper had
started its `whole_dry_run.py` phase. The shared registry identified its
1.5B worker and embedding owner, not an idle/free model lane. The audit
refused admission without loading the 27B. Wrapper 4422 (creation time
1790609903.985479), parent 6720 (1790610384.417092), and their verified
descendants 6722, 6727, 6730, 7088, 7089, and 7546 were terminated under
the same task-specific priority authorization. No survivor remained.
The dry-run log at
`/Users/bryan/subject-core-runs/reports-s7-steering-4-dry-closes.log`
is preserved; that phase is interrupted, not complete. The calibration
audit then acquired its own exclusive lane under detached supervision.
