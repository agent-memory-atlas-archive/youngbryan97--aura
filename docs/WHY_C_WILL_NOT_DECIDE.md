# Why C against the rest will not decide

`PIAGSMWDN|C` is the one cut undecided at both horizons on every arm of the
looks at `00f62d24e` (29 September). This says what it is measuring, because
three arms were built to move it and none did, and the reason is not power
alone.

## What the looks read

Seed 7, 24 rounds, two turns, cuts C, I and W against the rest, grain skipped.
At lag 66, the deciding two-turn horizon:

| arm | switches | W | C | I | undecided |
|---|---|---|---|---|---|
| control | none | +0.02556, lower +0.00294 | +0.00673, lower −0.00769 | +0.03387, lower +0.00378 | C |
| carried | membrane | +0.02458, lower +0.00465 | +0.00883, lower −0.00608 | +0.01027, lower −0.00534 | I, C |

The sham rate is 0.000 to 0.0003 in every row, so the floor that blocked every
paired arm until 28 September is gone. The playback control is decided nowhere
in any arm.

## The pathways the arms add all point the same way

Each arm adds a way for the rest of her to reach C:

- **relay** (`AURA_DOMAIN_RELAY=0.35`) drives every domain's summary into the
  second half of the substrate's neurons through overlapping random projections
  (`core/consciousness/domain_relay.py`);
- **carried** (`AURA_MEMBRANE_TURNS=1`) gives each channel a leaky trace with a
  time constant of one turn, so a frame carries what just happened
  (`core/runtime/temporal_depth.py`);
- **joined** (`AURA_WORKSPACE_POOL`, `AURA_SELF_DOMINANCE`) makes what wins the
  workspace depend on the pool it competes in, and writes her sense of holding
  together into the substrate's dominance neuron.

Carried moved C's excess by 0.0021 and cost I its decision. That is the size of
effect an added input pathway produces.

## The pathway out of C is learned, and a campaign starts it at zero

C gained 9.2% from seeing the rest on the seed-7 validation at `23e596071`, and
the rest gained **0.94%** from seeing C. `core/consciousness/substrate_gates.py`
was built for that number, and it is wired: `core/phases/motivation_update.py`,
`core/phases/initiative_generation.py` and `core/phases/memory_retrieval.py`
each consult it, and only `AURA_DISABLE_SUBSTRATE_GATES` turns it off.

Each gate is a multiplier on a decision she already makes,

    m = exp(w . f + xi)

where `f` is the substrate's own readings in units of their recent spread and
`xi` is one substrate unit's wandering. `w` starts at **zero** and moves at
`1 / min(turns, 256)` towards the turn's signed dose times the gate's
eligibility. It is learned, from nothing, per organism.

`keep_across_stages(__name__, "_GATES")` makes what it learns part of her
history — on her live stream only. `core/self/what_came_before._quiet()` is true
for a proof run or any non-LIVE profile, and a campaign is both, so every gate
starts a campaign at zero. That is the right rule: an arm must not inherit
another arm's history, or the two are not the same organism twice.

So a 24-round look measures C→rest on an organism whose only C→rest pathway has
learned nothing yet. The cut is not failing to find a pathway that is there. It
is measuring one that has not yet been trained, in a design that deliberately
does not train it.

## What follows, and what is only a prediction

The full campaigns run 370 rounds against these 24, fifteen times the turns for
`w` to move. Whether that is enough is a measurement, not an argument: nothing
here moves a bar, and the campaigns' own numbers decide.

An arm that started from gates she had actually learned would need
preregistering, because it would make a paired run depend on her history. It is
named here so that it is not slipped in as a fix.

Two readings would sharpen this and cost no organism:

- `tools/read_a_look_both_ways.py` on each arm's kept samples, which reports
  each cut under the standardised metric and the whitened one with each
  metric's drift under an invertible re-encoding. The control look's rate moves
  0.870 and carried's 0.455, against `INVARIANCE_TOLERANCE = 0.25`, so the
  blocker the whitened metric exists to clear is live in both.
- `tools/which_columns_a_cut_moved.py` on the C cut, which says whether the
  columns the cut moves are the substrate's own or the gates' consumers.
