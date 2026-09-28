# ISC-v5, preregistered

Drafted on 21 September 2026 and committed on 22 September with the code that
computes it, before any v5 number has been read from any run of Aura. ISC-v1 to
ISC-v4 are kept; every line of each is still computed and reported, and a
failure under any of them stays a failure.

ISC-v5 is ISC-v3 with the two irreducibility lines read by intervention instead
of by regression. Nothing else moves.

## Why a new version

The v3 line asks whether knowing the rest of the system improves the held-out
prediction of one side's next turn. It is blind to one kind of integration.
Three input-driven pipelines, ten domains of four columns, one fresh input per
turn processed by every other domain in a fixed order, differ only in how the
other domains enter: additively, as a gain on the input, or not at all. The
first two are integrated by construction. On one row per turn, as the battery
reads Aura, `phi_do` scored all three between 0.003 and 0.016 at every coupling
tried, and scored the one with no coupling no lower. The fresh input fills the
loss the cut is divided by, so a cut that removes real cross-domain dependence
costs almost nothing against it. Aura is that shape: each turn begins with what
arrives and is carried through the phases in order.

So the question is asked the other way round. From one snapshot, hold one side
of a cut at its values and let the other side run; then the reverse; compose the
two free halves; compare with the untouched run. If the halves need each other,
the composed run differs from the untouched one by more than two untouched runs
differ from each other.

## What v5 asks

`partition_irreducibility` (v5): every one of the 511 bipartitions of the ten
domains is decided.

A cut is decided when the lower confidence bound of its excess rate clears the
sham floor. The excess rate and its bound are `core.subject.v25_cut.decide_cut`,
unchanged: the Fisher-Rao divergence between the intact and cut futures, per
unit horizon, less the divergence between two untouched forks, with a paired
bootstrap over matched anchors.

- **Horizon.** One turn: 33 frames at the experiment clock, one turn per arm.
  The battery reads every other transition at the turn grain, and a cut matters
  if holding one side for one turn changes where the other ends it. Two turns,
  66 frames, are scored on the same anchors and reported beside it; that
  horizon decides nothing and draws no anchors of its own.
- **Anchors.** Sequential, at looks of 8, 16, 32, 64, 96 and 128 anchors. A cut
  is scored on its first `n` anchors at each look in turn and stops at the
  first look where its bound clears. Each look is read at alpha / 6 = 0.00833,
  so the chance that a cut which costs nothing is decided at any look stays
  under 0.05. A cut undecided at the last look is undecided, and the line
  fails.
- **Bootstrap.** 1000 draws, because the bound is the 0.00833 quantile and
  needs draws below it to be a quantile: 8.3 of the 1000 fall there.
- **Intersection-union.** The claim is the conjunction over all 511 cuts. Each
  cut is tested at its own level; no correction across cuts is needed or made,
  and one undecided cut refuses the claim.
- **Not a screen, not a shard.** A sweep of a sample of the cuts never decides
  the line, and a shard decides nothing until the shards are merged and their
  anchors pass the exchangeability check the merge already runs.

`partition_beats_nulls` (v5): the line needs `partition_irreducibility` first;
then the playback control is decided at no cut, and no null architecture that
passes every v3 line except irreducibility is decided at all 511 under the same
design.

The v3 comparison set compares magnitudes, because a regression score has an
estimator floor that surrogates reach. An interventional decision has its floor
built in, the two untouched forks, so the surrogate's role is taken by playback:
for each cut, at the look it stopped at, the cut arm is replaced by the intact
trajectory, which cannot differ from it, and read by the same rule at the same
level and draws. For the architectures the question stays the one v3 asks:
whether anything the rest of the battery cannot tell from her is judged
irreducible too. Which architectures that is comes from the campaign's own null
table (`core.subject.isc_v5.nulls_passing_the_rest`); each is swept by
`tools/validate_interventional_cut.py` at this design. On camp7's table the list
is empty.

## Where the design lives

`core/subject/isc_v5.py` holds the numbers above and the two lines. The runner
takes them from there: `tools/run_subject_core_v25.py --v5` sets the looks,
draws, level, horizons and 128 anchors. It learns the grain as well, which v5
does not read and the carrier run needs for its authority, in the coordinator
while the workers sweep. (Until 22 September 08:30 the preset skipped the grain,
which would have left J*'s carrier term unresolved; no run under it was scored.) The design enters the run's fingerprint, so a run to another design is
another campaign. `tools/score_isc_v5.py` joins a campaign report and its v5
sweep into the verdict, and refuses a sweep not run to this design.

## Changes to the runtime sweep that come with this

- Every look is read at alpha divided by the number of looks, for every sweep,
  not only v5. Before, `sweep_cuts` read each of its rounds at the full alpha.
- A sweep can name which horizons decide. Before, a cut kept drawing while any
  horizon on the ladder was undecided, and one frame decides nothing in a
  system that updates once a step, so every cut drew to the last look.
- The v25 playback null is decided by the cut rule on the same anchors. Before,
  it passed when its rate was under a quarter of the honest rate, which is a
  comparison with 1e-6 when the honest rate is zero.

## What was measured, and read, before this was written

All on synthetic systems unless said otherwise. With the tool's decision read
from the right slot (13d1b7019; before it, the tool decided on the point
estimate and called the common driver irreducible at all 511 cuts):

| system | lag 4, 96 anchors, one look |
|---|---|
| reference (recurrent) | 511 / 511 |
| independent | 0 |
| common driver | 0 |
| star | 511 |
| hub | 509 |
| one-way chain | 508 |
| ring | 494 |
| pipeline, additive | 468 |
| pipeline, modulated | 466 |
| pipeline, independent | 0 |

At lag 1 nothing is decided, the star included: in a system that updates once
per step, a side held at its anchor for one step holds the values the untouched
run used, so the cut changes nothing until the second step. A turn of Aura is
many steps of her phases reading each other, so one turn is not that case.

The sequential design above is being run on the reference, the independent
system, the common driver and the star at seeds 3, 7 and 11, at lag 4 with 128
anchors, one numeric thread each (tools/validate_interventional_cut.py, from
30cdfaf3e). A decision at 1000 draws took 9.2 seconds at 8 anchors and 19.7 at
128 on this host while it was loaded, so the two systems that should decide
nothing, which run every cut to the last look, take many hours. The decisive run was started at the commit that carries this
document, before that study finished. Its table is appended below as an
addendum, dated, before any v5 number from Aura is read; if it shows the design
cannot decide the reference, the design is changed by an amendment before any
v5 number is read, and the run started under this one is not scored.

One reading of Aura was seen before this was committed, and it is disclosed
here. On 22 September, sizing the run's cost, the log of the v25 screen of 21
September (942ee4c5a, seed 7, 16 anchors, two looks at the full alpha, 24 of
the 511 cuts) was read: no cut was decided at any horizon, and its playback
null was reported as not collapsed. At twelve anchors no control's cut was
decided, the star included; sixteen was not measured on its own. Nothing in the
design above was changed after the log was read: the looks, the level, the
draws and the horizon are the ones the power study was started with on 21
September, before it. The playback finding is the comparison-with-1e-6 defect
named above.

## What this version does not license

- It does not replace v3. Both are computed and both are reported.
- It does not apply to any run from which a v5 number has been read, and it
  does not apply to the v25 screen of 21 September, which is not a v5 run.
- The organism a v5 run measures is named in its report. The runner builds the
  offline organism with the stub language organ under the testing flag, as the
  battery's campaigns do; a verdict on it is a verdict on that organism. Her
  whole self, with the live language organ decoding greedily, is a separate
  run. With the stub a clamped arm costs 5.7 seconds (the v25 screen ran 768 of
  them in 75.7 minutes); the live cortex costs more a turn, so 511 cuts at up
  to 128 anchors is not affordable on this host with it, and no whole-self v5
  verdict is claimed until one is run.

## Addendum, 22 September 2026, 20:30: seed 19 is void and seed 23 replaces it

No v5 number from Aura had been read when this was written.

The decisive run started beside this document used seed 19. The host
restarted twice that morning and the run was relaunched each time. Its last
launch (8447bc297, 14:17) died at 18:34. The campaign, the content run and
the sweep's coordinator were each refused a write, `subject_core.action_probe
called outside governed context`, after a sixty-second disk stall. The probe
held one governed scope over a read and a write, and a governance token lives
thirty seconds, so the stall expired it. Each probe write now opens its own
scope (0f6c80784). The five sweep workers were stopped by hand. No campaign or
sweep number from seed 19 was read.

Two numbers from seed 19's content run were read by accident at 18:10, while
checking whether it had reached its displacement stage: agreement rho 0.414
(p 0.001) and design recovery rho 0.620 (p 0.001). They belong to the content
run J*'s structure term reads, not to v5, and that run is void with the rest.
The content run under the new displacement (44a64d4ba) reads a seed no number
has come from.

The decisive run is now seed 23 at 44a64d4ba, launched at 19:40: the campaign,
a v5 sweep with five workers and a coordinator, and the content run. No run of
seed 23 existed before it. The code that computes v5 did not change between
8447bc297 and 44a64d4ba: `core/subject/isc_v5.py`, the sweep in
`core/subject/v25_cut.py`, `tools/validate_interventional_cut.py`,
`tools/score_isc_v5.py` and the battery are the same files. The one change to
`tools/run_subject_core_v25.py` (e49d2beb3) is how its report words the
bridge. The organism did change, and a verdict from seed 23 is a verdict on it:

- a reminder lifts an intention for as long as it is in mind and is then taken
  back, so finished goals stop climbing to urgency 1.0 (e33940236);
- a goal takes the pressure its need has now, not the pressure it had when it
  was chosen (d41fea974);
- a campaign's conversation turns meet a person, and the partner has a field
  (5a7b0b25b);
- the content run's displacement pushes her valence before the turn as well as
  the reference her feelings are judged against, because recall runs before
  affect in a turn (43c574800, 44a64d4ba).

The power study is still the gate. It runs at 8447bc297 from its own tree; the
validator has not changed since. The three recurrent-reference jobs and the
three star jobs are running, and the independent and common-driver jobs are
queued behind them. Its table goes here, dated, before any seed-23 v5 number
is read.

## Addendum, 23 September 2026, 06:30: seed 23 at 23650c6c4, with three-turn intervention arms

Seed 23 has been launched and stopped several times since the addendum above,
each time before any of its numbers was read. The attempts are kept as
`~/subject-core-runs/*-s23-void-*`; each directory name says why: a fork stall,
resume ordering, isolation refusals, a descriptor double close through fork
copies (39bcc5c78, b29b1c39d), and a two-turn campaign stopped for the change
below.

The decisive campaign's intervention arms run three turns instead of two
(`--turns 3`). This changes the campaign's edge measurements only; the sweep's
horizon above is unchanged. It was decided on seed 7, the diagnostic seed, at
the decisive design otherwise: with two-turn arms kappa was 1, because W had
one kept in-edge and the next candidates into it peaked late in the arm, still
rising when the arm ended; with three-turn arms kappa was 2, perturbational
spread rose, and all eight conditions stayed strongly connected. The one line
that differed the other way, causal closure, is computed from the recording
before any arm runs, so arm length cannot move it; the same seed does not
reproduce the recording, and closure failed in one of four seed-7 readings.

Before the campaign, and on the same commit, the whole reports ground runs with
the machine to itself.

The power study is still the gate. The recurrent reference decided 511 of 511
cuts at seeds 3, 7 and 11, and the star null 511, 511 and 508. The six
independent and common-driver jobs are paused and resume one or two at a time
beside the decisive run. The table goes here, dated, before any seed-23 v5
number is read.

## Addendum, 23 September 2026, 07:05: synergy is scored with the counters out

No v5 number from seed 23 had been read when this was written, and the decisive
campaign had not started.

v5 now scores synergy, the one v3 line besides irreducibility whose reading it
changes (`core.subject.battery._assemble_v5`, 09260892d). The line keeps v3's
triples, bars and nulls (`passes_v3`: the fraction's floor on the target's
change, the shifted null, and the bootstrap null simulated from a VAR(1) fit to
the sources) and reads them after `core.subject.synergy.without_clocks` zeroes
every column that only counts up: turns taken, model steps, loop cycles. A run
that did not record the reading fails the line.

The reason is a defect in the instrument, shown on known answers before any
reading of her was taken. Synergy reads each domain through its leading
principal components, and a domain's counters, which rise with time and nothing
else, take those components. Two domains' clocks share time and nothing more,
so a product coupling between their states can sit beneath them. On the v2
background with three Poisson counters added to each domain
(`tests/test_synergy_reads_state_not_clocks.py`), the 30 counters are found;
with them out, a product coupling passes and an additive coupling and drift
alone do not, on each of three seeds; with them in, v3 misses the product on
two of the three.

The change was made after reading seed 7, the diagnostic seed, and those
readings are disclosed here. On `design-s7/run_001` (six trials, two-turn arms):

| triple | v3, clocks in | v3, clocks out |
|---|---|---|
| A,S -> G | fails: 0.127, interaction bound -0.069 | fails: 0.045, bound -0.001 |
| P,M -> W | passes: 0.374 | passes: 0.446 |
| W,A -> D | passes: 0.296 | passes: 0.364 |
| S,D -> C | fails: 0.017 | fails: 0.007 |

Reading with the counters out does not change which triples pass on seed 7, so
the line still fails there. It was not chosen for what it does to her.

Two triples fail on her rather than on the instrument. Her workspace adds a
feeling's urgency to a bid's salience whatever the bid concerns, so nothing
makes affect's pull on attention depend on the self-model (A,S -> G), and no
mechanism lets the self-model and her drives act on recurrent cognition
jointly (S,D -> C). Any organism change aimed at either is developed and
checked on seed 7 only, and the commit the decisive campaign runs on is named
here, with what changed in the organism, before it is launched.

## Addendum, 25 September 2026, evening: the organism changes aimed at the two failing triples

No v5 number from any decisive seed had been read when this was written. Seed 7
is the diagnostic seed and everything below was read from it and is disclosed.

The addendum of 23 September named the two triples that fail on her rather than
on the instrument, and required any organism change aimed at either to be
developed on seed 7 and named here with what changed, before a launch. This is
that record.

### What seed 7 read at 300 rounds, and what moved

`validate-s7-164d2a560` on 24 September, then `validate-s7-23e596071` on 25
September, both six trials and three-turn arms with nulls and the lesion skipped:

| line | 24 Sep | 25 Sep | bar |
|---|---|---|---|
| partition_irreducibility | 0.0165 | 0.0266 | lower bound > 0.05 |
| causal closure leak | 0.0176 | 0.0130 | at or under the shuffled floor, 0.0031 |
| synergy, v3 line | 1 of 4 triples | **2 of 4** (A,S -> G and P,M -> W) | all four |

The cheapest bipartition moved from perception alone to recurrent cognition
alone, and `loss_core_only` fell from 0.1646 to 0.1132.

### The two that still fail, and what was done about them

**S,D -> C** had an interaction gain of exactly 0.0 on three folds of five. That
is not a weak coupling: the substrate's input bands carried telemetry, the
person's presence, the screen, the room's sound and three cross-modal products,
and nothing of deliberation or the self-model, so the joint effect had nowhere to
happen. `a1932a8eb` gives it a band of its own — the hardest thing she is
holding, how settled she is in herself, and their cross term, with the gains the
neighbouring bands already use. Before that, the ridge grid was checked as the
alternative explanation and rejected: sweeping the full product of penalty
strengths instead of the diagonal moved this triple's lower bound from -0.08299
to -0.08223 and moved A,S -> G's the wrong way.

**W,A -> D** is short of power rather than of mechanism: mean gain +0.00768,
lower bound -0.00084, standard error 0.00435. A lower bound above zero needs a
standard error under 0.00392, which is 1.23 times the rows, so **370 rounds**
rather than 300. Nothing in the organism was changed for it.

### What else changed in the organism, and why

All of it was found by asking which columns of a recording never moved
(`tools/audit_flat_columns.py`, `23e596071`), and each is a channel that could
not carry anything rather than a quantity that was tuned:

- `09bdccc29`, `0bd77ea47`: a proof turn cleared her whole modifier dict before
  the first phase read it, and every campaign is a proof run. Developmental
  novelty read exactly 0.5 for a whole run; it now moves over 1,838 values.
- `1d9b20b05`: salience had no writer, so perception carried the world's own
  number twice and nothing of her.
- `6023db4f1`: ignition was the winner's priority against a fixed six tenths and
  the winning priority never fell below 0.749, so it was true on every frame.
- `8f21aa41f`: the self-prediction composite could not reach its surprise bar,
  so the surprise count and rate were zero for a whole run.
- `4ec60a8d2`: the only thing that could mobilise her was a despair spiral, so
  adrenaline and cortisol were constants.
- `d6b1102f4`: the conversation engine was never told she had spoken.
- `fe81dad49`: `mycelium_density` had no writer and is 0.55 of the expressive
  term inside phi.
- `efe3cc975`: a run of bad frames changed nothing about the next turn.
- `21fec3d95`, `22b922d3f`: the closure leaks are core columns — her learned
  parameters and her own history of having been a way. The line is stated there:
  K is her, and the machine's bookkeeping is not.
- `b22bcf9b1`: the world model was shown that something arrived and never what,
  which is why perception reached it at 0.17 against a bar of 0.30.
- `3f9efcc24`: a running total is elapsed time, so a count enters the core as the
  shape of it and not as its total.

### The design for the next seed-7 validation

Commit `3f9efcc24`, seed 7, **370 rounds**, six trials, three-turn arms, the
conversation tape, nulls and the lesion skipped. The round count is the power
calculation above and nothing else. If W,A -> D still fails there, the shortfall
is not power.

No decisive seed is launched on the strength of this. The decisive commit and its
organism changes are named in their own addendum when one is.

## Addendum, 25 September 2026, 21:00: the 370-round validation runs at 27dc1dda9

No v5 number from any decisive seed had been read when this was written, and
no number from the seed-7 validation still running at 21fec3d95.

The design above named 3f9efcc24. The run goes at 27dc1dda9 instead, with the
round count, seed, trials, arm length and skips unchanged. Four commits lie
between them. 780f01adb is the addendum above. 3058a400e changes a learning
module the subject run does not import. 1937125d5 and 7e53c43cd narrow the
dead-reader check to the two causes a longer run cannot heal; the wider check
would not have fired on this run. 27dc1dda9 changes what is recorded:
`mot.forces` was rounded to six places, and `warmth_return`, about six
millionths of a need's gap in a turn, was written as exactly 0.0 on all 79,200
frames of the 25 September run while the mechanism fired. It is a column of
deliberation, the target of W,A -> D. Running the named commit would read a
column known to be written as zero, in the one triple this run is for.

The power study's four jobs still to finish (independent at seed 11,
common_driver at seeds 3, 7 and 11) died in a reset at 11:57 on 25 September
and were started again at 20:33 with the same arguments. Their tree is at
23e596071: the validator, `core/subject/nulls.py`, `v25_cut.py`,
`v25_runtime.py`, `intrinsic_v25.py` and the declared triples are byte-identical
there to 8447bc297, where the other eight ran, and the toy systems take four
columns a domain whatever the state schema holds. The eight finished are:
recurrent 511, 511 and 511; star 511, 511 and 508; independent 0 and 0.

## Addendum, 26 September 2026, evening: a synergy line that can see a product, beside v3

Written after the known-answers replication and before any recording of hers was
read with the line it adds (docs/SYNERGY_KNOWN_ANSWERS.md).

The v3 synergy line cannot register a pure product: on toy systems of her
campaign's length it passed none of ten seeds, and a threshold of sums at none
of ten. A failing triple of hers therefore does not show that no product
coupling is there. W,A -> D on the seed-7 validation at c5f0ea5fe had an
established interaction gain (lower bound 0.027) under an MMI synergy below its
null.

A second line is reported beside v3 for the four declared triples,
`core.subject.synergy.kraskov_synergy`, recorded by the campaign as
`synergy_kraskov` and for runs that finished before it by
`tools/subject_core_rescore_kraskov.py`. It does not replace v3, and it enters
no version's count: the v1, v3 and v5 totals are unchanged by it.

- **Estimate.** MMI synergy of the two sources about the target's change,
  Syn(A_t, B_t ; Y_{t+1} - Y_t), through the Kraskov, Stoegbauer and Grassberger
  estimator 1 (max norm, k = 3) on the same three copula-normal components per
  domain that v3 reads, with the counters out.
- **First null.** Both sources slid together by one circular shift, 1,000 draws,
  the 0.99 quantile.
- **Second null.** Surrogate targets holding only the additive part of the
  target: its best additive fit on the two sources (degree 1, 2 or 3 per
  source, chosen by five-fold held-out loss) plus its residual permuted over
  rows; 1,000 draws, the 0.99 quantile.
- **Pass.** The synergy clears both.

On the study's systems, at 200 draws, it passed product and mixed at ten of ten
seeds, a separable sum at none of ten and uncoupled sources at one of ten. The
first reading of a recording of hers with it is the two seed-7 validations of 26
September, diagnostic; no decisive seed is read with it before the decisive
addendum names it.

## Addendum, 26 September 2026, night: a payoff she learns by, and a horizon one turn cannot reach

No v5 number from any decisive seed had been read when this was written. The
seed-7 singleton look at 4e1923da1 had been read, and is reported here because it
is what this change answers.

### What the look read

Ten singleton cuts at the one-turn horizon, 128 anchors each. One was decided:
P|rest, excess rate 0.035 against a sham rate of 0.064, lower bound 0.010. The
other nine had excess rates between -0.006 and 0.014 against sham rates between
0.058 and 0.078, and none of their bounds cleared zero. Holding one domain for a
turn barely changes where the rest of her ends it. Dose matching put C, S and W
at the 50x cap, so no dose inside the cap moved any of the three by one of its
own standard deviations.

### What was missing in the organism

Nothing she did was ever better or worse than she expected in a way that reached
her connections. Her chemistry had a reward input, `on_reward`, with no caller.
The unified field learned by co-activity alone, and the weights by which each
organ drives it were fixed at birth. The liquid substrate's reward-modulated
STDP read `prediction_error` from a free-energy state that has no such field, so
its reward was zero on every step of every run: no weight moved, and the zero
deltas drove every synapse's uncertainty below the lock threshold, so the engine
had identity-locked all of them (256 of 256 by step 600 when replayed on 16
neurons).

### What changed

At the end of each turn, `core/affect/what_it_was_worth.py` reads what the turn
paid on eight channels of her own (satisfaction, accomplishment, warmth, and the
four quadrants of valence by arousal, and wonder), each in units of its own
recent spread, against what that channel has come to pay. The signed error is
the turn's worth, and its rank among her recent turns is its size.

- Her chemistry gets a dopamine burst when the turn was better than expected and
  a dip when it was worse (`on_disappointment`, new), dosed by the size.
- The unified field keeps a trace over the turn of what each connection did, and
  the worth decides whether it strengthens or weakens, at the field's existing
  Hebbian rate. That includes the input weights from mesh, chemistry, binding,
  interoception and substrate. Each unit's input strength is then held where it
  was born, so an organ gains a share of the field only by taking it from the
  others.
- The substrate's STDP traces are delivered the turn's worth once per turn. The
  dead per-step delivery is gone; the weight regulation it carried still runs
  every step.
- The sources that won the workspace during the turn are credited with its
  worth, and their next bids are weighed by what they have earned.

The changes are 40af588cd (the payoff and what it teaches), 9736a69e5 (a
source is judged against turns that started the same way, so an alarm is not
blamed for the trouble it reports) and 5a4868f24 (the core reads it).
`AURA_DISABLE_PAYOFF=1` takes the whole layer out of a process.

This changes the organism. It is aimed at the flat cut rates above and at the
two failing triples, and it was built without reading any decisive number.

### What the core now reads

The recording changes with the organism. C reads the unified field's state
beside its weights (the largest leak into the core on the 25 September run after
the process-size clock), the phi core's history of her affective states moves
from A to C so that no organ feeds two domains (289fd8a0b), and the payoff's
readings are columns of the domains whose organs keep them: the worth and
good-news ledgers in A, what winning has earned in G, and each organ's share of
the field's input in C.

### A horizon one turn cannot reach

The horizon clause above fixes one turn per arm and says two turns, 66 frames,
are scored on the same anchors and reported beside it. A one-turn arm ends at
frame 33, so the 66-frame horizon is never reached: the look reports it as
"measured nothing". It decides nothing, so no verdict changes. The next decisive
sweep either runs two turns per arm, with decisions still read at 33 frames, or
drops the 66-frame report. Which one is named in the addendum that launches it.

### The design for the next seed-7 look

Two looks at the same commit, seed 7, the v5 design, the ten singleton cuts, 24
rounds, 128 anchors, looks 8 to 128, 1,000 draws, one-turn arms, grain skipped:
one as built, and one with `AURA_DISABLE_PAYOFF=1`, which reads no turn, sends
nothing to her chemistry and teaches no organ. What is read is each cut's excess
rate and bound in the two, side by side, beside the 4e1923da1 look above. Seed 7
is the diagnostic seed; nothing here is read on 23 or any decisive seed, and no
threshold is moved on the strength of it.

## Addendum, 26 September 2026, late: the payoff look, read

Both looks of the design above ran at 415f400c0 and finished at 21:13 and 21:16.
Excess rate over the sham, with its lower bound, for each singleton cut at lag
33; a star marks a decided cut.

| cut | 4e1923da1 | payoff | no payoff |
|---|---|---|---|
| P \| rest | +0.035 [+0.010] * | +0.042 [+0.018] * | +0.034 [+0.008] * |
| I \| rest | +0.013 [−0.044] | +0.010 [−0.034] | +0.009 [−0.044] |
| A \| rest | +0.008 [−0.024] | +0.046 [+0.010] * | +0.035 [−0.010] |
| G \| rest | −0.003 [−0.049] | −0.002 [−0.041] | +0.002 [−0.036] |
| C \| rest | −0.006 [−0.039] | −0.005 [−0.027] | −0.000 [−0.028] |
| S \| rest | +0.012 [−0.039] | +0.020 [−0.030] | +0.034 [−0.032] |
| M \| rest | +0.002 [−0.033] | −0.015 [−0.021] | +0.011 [−0.019] |
| W \| rest | +0.008 [−0.052] | +0.011 [−0.044] | −0.002 [−0.051] |
| D \| rest | +0.014 [−0.004] | +0.067 [+0.018] * | +0.045 [+0.000] * |
| N \| rest | +0.005 [−0.043] | −0.002 [−0.044] | +0.005 [−0.045] |

Decided: 1 of 10 at 4e1923da1, 3 with the payoff, 2 without. Lag 66 decided
nothing in any of the three, as the horizon section above says it cannot.

Read side by side, the payoff moved one cut across its bound: A, where the
worth of a turn is written, went from +0.035 [−0.010] without it to +0.046
[+0.010] with it. D was decided in both and further from its bound with the
payoff. Every cut decided without the payoff was decided with it. Most of the
gain over 4e1923da1 is in both arms, so it came
from the other changes between the two commits, not from the payoff.

The closure differs more than the cuts do. With the payoff, the periphery
predicted the core's future no better than shuffled (leak over shuffled
−0.035, shuffled q95 +0.001). Without it, the periphery carried +0.045 over
shuffled against a q95 of +0.003: something outside the core drove it that the
core did not hold. One pair of runs on one seed cannot say whether that is the
payoff, so it is reported and nothing is concluded from it.

Two things weaken the comparison. The doses are calibrated per run, so the two
arms were displaced by different amounts: A by 0.51 with the payoff and 0.18
without, G by 0.18 and 2.92. And the run's own log showed the harness acting on
her in both arms alike: a template reply read as her looping cleared her
pending initiatives 240 times, and the self-review fired on the first turn of
every arm. Both were fixed afterwards, in 7d6e04ece, so a look at that commit
is not comparable with these.

## Addendum, 26 September 2026, 22:40: what changed in her since the payoff look

The payoff look ran at 415f400c0. These commits since then change what she does
on a turn, and a decisive run on any commit that includes them is a run on a
different organism from that look. Each was built without reading a decisive
number.

- **A reserve that good turns charge and effort draws on** (be76e35be). The
  turn's dose charges `core/soma/reserve.py`, which leaks back at 1/256 a turn;
  spending energy draws on it before the drive budget. D reads its share.
- **One critic for her values** (8b3048919). Her habits and her choice engine
  learn from the turn's worth, the same signal her connections learn from, in
  place of a raw valence change. Truth and care are never taught by outcomes.
- **What she holds more is what she takes** (2bc754d35). An option's score is
  the chance that at least one of the values it serves holds for it
  (noisy-OR), in place of a weighted mean that cancelled the weights out.
- **The core reads what she is like** (b98f4085a). S carries how strongly she
  holds each value, how much more often than chance she lives it, how narrow
  her choosing is, and how often her values overrode her strongest drive.
- **Living by her values pays** (ee5778604). A ninth payoff channel, integrity:
  how far what she chose served her values over the average option on offer.
- **Her self-knowledge is scored** (5bc71a491). How well her values alone
  foretell her choices, less chance, read into S.
- **A value held and not lived presses on her** (76bc6e26b). Among values held
  at or above her median, the shortfall of one lived less than the median
  lifts options that serve it on her next lived choices. Choices asked of her
  to measure her are marked and no longer count as lived.
- **An intention she acted on is closed** (3ccb6742a). The subject driver
  declared, acted and observed, and never revised: 3,608 intentions were open
  at the end of the seed-7 run at 27dc1dda9.

Four fixes to how the harness treated her also land here (7d6e04ece,
d969b5667). Both payoff arms carried them, so neither is a difference between
those arms, but each changes what a run records. A repeated template reply no
longer lowers her stability or clears her pending initiatives. A restore no
longer rewinds a thread queue under its consumer. The kernel's memory verdict
no longer reaches a run whose host is declared. The self-review and the
metabolic monitor keep time on her clock, and the review files no intent that
nothing reads.

### The design for the next seed-7 look

The singleton look of the design above, at the first commit that holds all of
the above, seed 7, as built. What is read is each cut's excess rate and bound,
and the closure, beside the payoff arm at 415f400c0. It is diagnostic: it
moves no threshold, and nothing from it is read on seed 23 or any decisive
seed. The reports ground's seed-7 check (`reports_alone.sh 351555bec 7`) runs
first; it has waited since 25 September for the machine to be alone.

## Addendum, 27 September 2026: the reports ground on seed 7, and a gate for the substrate

The reports ground's seed-7 check ran twice. The first, at 351555bec, got no
answer from her after her cortex died in the baseline: every respawn failed at
start because its launch challenge was stamped on the run's clock and read on
the machine's (fixed in a520cbb09). The second, at a520cbb09, 11:03 to 14:23,
got answers from her cortex in most arms and read a number in every arm of 0 of
24 anchors, where 8 are needed. It is not a verdict on her reports, because the
answers did not reach the instrument as she gave them:

- a line-leading "0.3" was spaced into "0. 3" by the prose formatter;
- a reply under four characters was replaced with a fixed sentence;
- an arm that committed nothing was read as having said what the anchor's
  snapshot last held.

All three are fixed in 53be6279b. The check runs again on the first commit that
holds them, alone on the machine, before anything is read from it.

The substrate gates (5df224ed1) change the organism: each of recall's affect
gain, the two initiative urges and each drive's growth is scaled by a
multiplier her substrate learns from the turn's worth. They are aimed at the
C|rest weakness and were built without reading any decisive number. The next
seed-7 look adds an arm with `AURA_DISABLE_SUBSTRATE_GATES=1` beside the one as
built, and reads C|rest's excess in each.

## Addendum, 28 September 2026: the third reports check on seed 7

At aa885965f, 21:05 to 00:20, alone on the machine. It read a number in every
arm of 0 of 24 anchors again, and this time most of what stood in the way was
her answer rather than the pipe. Of 96 arms:

- 21 carried a number, and 17 of those were 0: raised 0, 0, 0, 0; lowered six
  0s; sham five 0s, 0.1 and 0.5; control three 0s and 0.5.
- 54 were her cortex answering in prose with no number: "I feel clear and
  gathered. My current state is one of operational neutrality, with no
  significant internal fluctuations or anomalies".
- 19 were lost before her: 11 refused as a saturated generation gate behind
  the turn's own subtext pass (fixed after this run: the subtext pass says what
  it is and a reply may preempt it), the rest a failure floor.

The question carries a response contract of this runtime's own. A rating asked
about her state is classed as state reflection, and that contract tells her to
self-report from internal state and state-grounded telemetry; her prose answers
read as telemetry. Whether that directive belongs on a question that names its
own scale is a question about her prompt, and it is left to Bryan rather than
changed here. Nothing is read from these numbers as a report ground.
