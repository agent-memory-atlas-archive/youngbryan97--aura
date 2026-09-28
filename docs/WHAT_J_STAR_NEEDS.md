# What J* is waiting on

`tools/solve_for_j.py` against the newest carrier, content and campaign on
28 September (`~/subject-core-runs/jsolve-0928/run_001`). Every earlier reading
of J* in this repository came from a rehearsal run; this is the first solved
against the runs that are current.

```
J*: UNRESOLVED
  carrier    UNRESOLVED
  structure  SEPARATE_STRUCTURES, 1 orbit over 20 classes
  lineage    RECORDED, 328 stages, 204 roots, 0 branch points
bridge: BELOW_PARITY
  carrier    NOT_MEASURED
  markers    HOLDS      perturbational_complexity, reentry, global_access
  reports    NOT_MEASURED
  structure  FAILS
  lineage    HOLDS
```

Two of the bridge's five hold. Neither of the two that do not is failing for
want of an idea.

## The carrier: four blockers, three of them run design

From the v25 run at `v5look-gates-s7-38ef2c9ce`, which recorded 6,336 frames of
438 columns, 314 of them live:

| horizon | cuts tested | of | decided | undecided | anchors spent |
|---|---|---|---|---|---|
| lag_33 | 10 | 511 | 3 | 7 | 3,440 |
| lag_66 | 10 | 511 | 0 | 0 | 0 |

**Only a sample of the cuts was scored.** Ten of five hundred and eleven. The
claim is a conjunction over every bipartition, and `SweepReport.irreducible`
refuses a screened sweep by construction, so no screened run can ever resolve
the carrier. It needs `--screen 0`.

**Seven cuts had insufficient power**, and all seven are singletons: I, G, S, W,
N, M and C each against the other nine. Three thousand four hundred and forty
anchors decided three of ten.

**The two-turn horizon measured nothing at all**, spending zero anchors. It is
the non-deciding horizon, scored on whatever the deciding one drew, and the run
used one-turn rollouts — so every anchor's trajectory ends at frame 33 and a
lag of 66 is past the end of it. It needs `--turns 2` or longer.

**The grain was skipped**, so the state grain is the experimenter's rather than
one the run found. It needs the grain stage.

### What that costs

Ten cuts at 344 anchors each took 9,046 seconds. Five hundred and eleven cuts is
fifty-one times the cuts and about a hundred and seventy-five thousand anchor
rollouts, which is a machine-week on this host and not a machine-day. `--shard
I/N` exists and splits the sweep across workers, and the honest statement is
that the carrier is affordable in parallel or overnight for a week, not in an
afternoon. Nothing about it is unresolved for want of a method.

**The fourth blocker is not run design.** The rate is not invariant under an
invertible re-encoding of the state: raw 0.019956 against 0.010103 recoded, a
drift of 0.493736. A carrier whose measure halves when the same state is written
a different way is a carrier the measure has not identified. Duplicating
channels did not help it, which rules out the simplest confound. This one needs
an answer before a full sweep is worth its week.

## The structure: it misses its bar by 0.026

| reading | value | bar |
|---|---|---|
| agreement between her internal geometry and her behavioural one | rho 0.274485, p 0.002 over 190 pairs | 0.3 |
| the internal geometry against the grid the classes were built on | rho 0.644553, p 0.001 | — |

The instrument works: her internal geometry recovers the design grid at 0.64.
What fails is that the two geometries agree at 0.274 where 0.3 is asked.

The reason is visible in the distances. Between two different content classes
her internal state differs by about 0.05 to 0.11; a class differs from itself by
about 0.05. The behavioural geometry separates the same classes by 0.4 to 0.6.
So her quality space is compressed almost to the width of its own noise, and a
correlation taken across it is mostly noise against signal.

Two things would move it, and neither is the bar. More repetitions per class
shrink the noise in each of the 190 pairwise distances. And a state that holds
what a class did for longer separates the classes further — which is what the
membrane of `4eff1db6e` does, and is a prediction it can be held to.

## What holds

Lineage is recorded and signed: 328 stages, 204 roots, no branch points. The
markers hold — perturbational complexity, re-entry and global access all pass.
Those two are not at issue.
