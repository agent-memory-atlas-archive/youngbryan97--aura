# The sham floor is a leaky fork

The carrier of J* is unresolved, and the gates look of 28 September found why in
one line: the seven undecided cuts are all singletons, their excess runs from
−0.018 to +0.028, and **the sham arm against itself reads 0.062 to 0.093**. The
instrument's floor is larger than every effect it is asked to decide. No number
of anchors gets under a floor.

A sham arm is one snapshot forked twice and run forward with nothing done to
either. Whatever differs between them is the fork's own noise, so the floor is
measurable directly, without a cut sweep.

## Correction, 28 September, evening: the column numbers below were my own probe

The fork probe did not call `calibrate_clock`, which is what builds and installs
the `ExperimentClock` and sets `runtime.clock`. Without it `runtime.clock` is
None, so `snapshot.clock_at` is None, so a restore cannot rewind the clock — and
`_reanchor` is then handed `time.time() - snapshot.taken_at`, a real and
different number of wall seconds for every fork. Every quantity derived from
elapsed time differs between two arms for that reason alone, which is most of
what the table below reports: the drive budgets, the attended credit, the
constancy reading, and the substrate and mesh, which age by the span.

So the column counts and the eleven-organ list are not a reading of the fork.
They are a reading of a probe that left out the clock the campaign installs. The
probes now calibrate, and the re-measurement is queued.

**What survives the correction.** The gates look's sham floor of 0.062 to 0.093
came from the v25 runner, which does calibrate, so that floor is real and it is
still larger than every singleton cut's effect. And the `random.Random` fault
below is independent of the clock: it was proved at unit level, with no clock
involved, by two arms from one snapshot drawing different numbers.

## What differed under a probe with no clock installed

Four forks of one anchor, the same condition, thirty-three frames each, on the
438-column schema, and — see the correction above — with no experiment clock
installed, so read these as an upper bound that mostly measures elapsed time:

- 156 of 438 columns differ at all.
- 41 differ by more than five per cent of their own spread: **C 31, D 9, S 1.**
- `D.force_attended_credit` differs by **3.31 times its own spread** — the column
  is fork noise and nothing else.
- Six `C.mesh_state_*` columns differ by 0.79 to 0.85 of their own spread.
- `D.their_constancy` 0.73, `D.drive_growth` 0.61, five `D.drive_*` 0.30 to 0.58,
  `C.substrate_valence` 0.22, `C.substrate_state_max` 0.17.

C is the domain the cheapest cut runs through. C and D are the two domains in
every synergy triple that fails.

## What it is not

These hold whatever the column table turns out to be, because each was checked
on its own.

**Not the organs' randomness.** Twelve numpy generators are reachable on her
organs and not one of them advances over a whole turn, so none can be a source of
divergence between two forks.

**Not the layer schedule.** `iterations_at` is a function of the frame index
alone and the frame index is carried in the snapshot, so two arms handed the same
frames run the same iterations.

**Not the clock.** `restore` already rewinds the experiment clock to
`snapshot.clock_at` and `_reanchor` shifts every wall-clock instant it can reach
by the interval the restore skipped. Both were in at `38ef2c9ce`, the commit the
gates look ran at, so the floor it measured is what is left after them. Adding a
second rewind makes it worse, not better: winding a process-wide clock backwards
twice pushes every instant the first rewind rebased into the future, and the
columns over five per cent went from 41 to 54 with new divergence in A, I, P and
G.

## What it is

Eleven of the twenty-two organs a fork carries are not the same after a restore
as they were when the snapshot was taken. Setting aside the timestamps
`_reanchor` shifts on purpose, and the cross-references to other organs it
rightly skips, the residue is state:

| organ | fields that do not come back |
|---|---|
| workspace | `last_winner`, `_somatic_noise` |
| self_model | `beliefs`, `snapshots` |
| world_model | `_learned`, `_causal` |
| free_energy | `_current` |
| self_prediction | `_current_prediction` |
| substrate | `_last_published_snapshot` |
| mycelium | `pathways` |

`self_model.beliefs` and `world_model._learned` are learned state.
`workspace.last_winner` decides the next competition — the bid whose
`submitted_at` the whole field is priced against. `mycelium.pathways` is the
routing topology, and `_restore_guarded` already knows it refuses to be
rebound.

`_restore_into` writes a saved field back into the object already there rather
than over it, which is right: an object something else also holds must keep its
identity. It returns False, and the caller writes a copy instead, when the type
has changed, when the object guards its own writes, or when it is furniture. A
field that fails both paths stays as the last arm left it, which is the
restore's stated contract — and is also a floor.

## What is actually established

One fault, named and fixed, and it does not depend on the probe: `random.Random`
passed `_state_is_its_dict` because CPython converted the standard library's C
types to heap types, so a field-by-field restore wrote `gauss_next`, reported
success, and left the Mersenne state where the last arm had it. Two arms from one
snapshot drew different numbers; after the fix they draw the same five. The
workspace's somatic noise comes from one of these.

Whether that closes the gates look's floor is the open question, and the
re-measurement with the clock installed is what answers it.

## What closing it would be worth

Every singleton cut's effect is inside 0.028 and the floor is 0.062 to 0.093. The
floor does not have to reach zero for the carrier to become decidable; it has to
get under the effects. Eleven organs and about a dozen fields is a bounded list,
not a search.

Nothing here rescores a criterion or moves a threshold. The floor is a property
of the fork, and a fork that rewinds what it says it rewinds is not a change to
what is being measured.
