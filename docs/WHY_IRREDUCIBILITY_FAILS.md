# Why the partition line has failed in every run

`partition_irreducibility` has failed in all twenty recorded runs. This is what
it is failing on, measured on `whole-s7-27dc1dda9` at its own cheapest cut
`PD|IAGCSMWN`. Nothing here rescores a criterion or moves a threshold; it names
what an organism change would have to do.

## Her state records instants of a process that integrates over a turn

`phi_do` fits a linear-Gaussian model of each domain's next-step change from the
current state, with a one-hot of the phase that produced the transition given to
both sides of every cut. Three readings of her recording, in that setting:

**The movement is sparse and heavy-tailed.** Per domain, the share of frames with
no movement at all runs 0.77 to 0.92 — C alone is lower, at 0.24. Kurtosis of the
steps runs from 142 (N) to 10,699 (D), against a Gaussian's 3. One column carries
94.6% of A's step variance (`A.delivery_z`) and 91.8% of W's (`W.her_placement`).

**Most of what a linear model can explain is the schedule.** With the phase
one-hot alone: 0.554 of C's next-step variance, 0.434 of P's, 0.381 of G's, 0.378
of D's, 0.360 of M's, 0.178 of N's, and about zero for A, I, S and W. Its own
current state adds at most 0.031 to any domain. **The other nine domains add
0.000 to every one of them.** What no model reaches is 0.437 (C) to 1.000 (S).

**The coupling is in which frames jump.** A hard move in P makes a hard move in I
3.7 times more likely on the next frame; W to S is 3.4, A to N 4.1, D to S 2.9,
M to I 2.8, P to A 2.2, G to S 2.0. Pairwise, one other domain adds up to +0.031
of held-out variance (I from P) where all nine together add 0.000, which is nine
domains' worth of predictors spent on a signal that small.

So the estimator is asked to find amplitude coupling in a signal whose structure
is the timing of rare jumps.

## A time constant makes it visible, and it is real

Each column carried as a leaky integral of itself, at time constants in frames:

| tau (frames) | 0 | 1 | 2 | 4 | 8 | 16 | **33** |
|---|---|---|---|---|---|---|---|
| phi | +0.015 | −0.066 | +0.015 | −0.121 | +0.057 | +0.160 | **+0.240** |
| lower bound | −0.057 | −0.292 | −0.081 | −0.555 | −0.139 | +0.030 | **+0.074** |

33 is `frames_per_turn` from the run's own clock, not a fitted number. At one
turn the lower bound is +0.074, over the preregistered 0.05.

A shared low-pass filter gives every column the same autocorrelation, and an
estimator that read persistence as coupling would score independent parts above
zero once filtered. Five nulls say it does not:

| system at tau 33 | phi | lower bound |
|---|---|---|
| hers | **+0.24020** | **+0.07414** |
| her rows shuffled | +0.00000 | +0.00000 |
| `independent` | +0.00000 | +0.00000 |
| `common_driver` | +0.00000 | +0.00000 |
| `high_dimensional_independent` | +0.00007 | −0.00008 |
| her domains dealt to other turns | +0.00006 | −0.00005 |
| the same, a second deal | +0.00009 | −0.00009 |
| `recurrent`, the positive reference | +0.30206 | +0.29294 |

The last null is the tight one. Each domain's turn blocks are kept whole and
dealt to different turns, one permutation per domain: every domain keeps exactly
the dynamics it had, within a turn and across turns, and no domain is in the same
turn as any other. Only the alignment between domains is destroyed, and the
reading collapses from 0.240 to 0.00006. So what the time constant reveals is
coupling between her domains and not the turn they share.

The filter raises the `recurrent` reference 1.22 times and raises hers 16 times,
which is what a system whose coupling is already visible looks like beside one
whose coupling is not.

## What this does and does not license

Scoring the criterion on a filtered recording would be changing the measurement
after seeing the result, which `P0.17` makes a new campaign and which would be
fitting the estimator to the answer. It is not proposed.

Giving the organism real time constants is a different act. Downstream consumers
read the integrated value, so what she does changes and not only what is
recorded, and the recording of that organism is then scored exactly as written.
Her schema already declares the split — N is the lifetime reservoir, S and M
carry across turns, the rest move within one — and does not implement it as
integration.

## What the design would have to be

Heterogeneous and ordered time constants, not one filter. Cortex has a hierarchy
of intrinsic timescales, tens of milliseconds in sensory areas to seconds in
prefrontal, and integration is computed over those windows; every synapse is a
low-pass filter and every population rate is a filtered signal. A channel with no
time constant has no window to integrate over, which is the state she is in.

The three failing criteria then read as one thing. Irreducibility asks whether
any seam is cheap. `partition_beats_nulls` asks whether the number is high for
the right reason, and the two nulls that beat her — `low_rank` at 0.562 and
`all_to_all` at 0.549 — have effective dimensions of 1.21 and 1.38 against her
11.2: they are bound and empty, she is rich and loose. Synergy asks whether two
domains decide something jointly, and a conjunction needs both of them present at
once, which a channel with no memory of the last frame cannot supply.

Reaching 0.562 from 0.240 is a further factor of 2.3 and is not promised by time
constants alone. What is measured here is that the coupling exists at the size
the positive reference has, and that her recording cannot see it.

## What the time constant costs, measured

A channel that carries what it has been is smoother, and smoothing reduces
effective dimension. `differentiation` is the criterion that would pay for it,
and it is one of the twenty-one that hold, so a membrane that closed the
partition line by flattening her repertoire would be a trade rather than a gain.

On the same recording, carried at one turn against as recorded:

| | effective dimension | live columns | regimes |
|---|---|---|---|
| as recorded | 11.2342 | 293 | 3 |
| carried at one turn | 8.7826 | 293 | 3 |

The floor is 3.0 (`THRESHOLDS["d_eff_floor"]`), and the normalised ratio is
0.030 against a 0.40 bar either way. So the cost is twenty-two per cent of the
effective dimension and it stays near three times the floor, against a partition
reading that rises sixteenfold. No column goes flat and the number of regimes
does not move.

That is the whole of the trade as far as the readings that come off the
recording alone can show it. What the lesion, the rescue and the null table do
under a membrane is not knowable without running them, which is what the arms of
the preregistered campaign are for.
