# Subject core work log

What was done towards 24/24, J* and the bridge, newest first. Each entry says
what broke, what fixed it, and where the evidence is. Runs live in
`~/subject-core-runs/`; the order runs are read in is
`~/.aura/subject_core/scratch/AFTER_THE_DECISIVE_RUN.md`.

## 29 September

### The night's three runs, read

**Structure.** The content run at `c093fcfb7` (seed 7, 24 anchors, the paired
geometry, loops stopped before the anchors) read an agreement of rho 0.443,
p 0.001, against a bar of 0.3. It was 0.274 before the estimator was fixed.
Design recovery is 0.713. Moves-together is −0.157, p 0.975, and the verdict
is AGREES_BUT_DOES_NOT_TRACK. Neither half of that is noise: the displacement
moved the internal geometry 0.166 against a sham of 0.009, and the recall
geometry 0.035 against a sham of exactly zero. They moved in different pairs.
Recall is keyed by what a percept is and by the valence she arrived with, so a
valence displacement shifts recall for every class in the same direction, while
the internal geometry is read over all 438 columns and moves wherever the
displacement propagates. That is a finding about her, and the displacement is
the next thing to build for; nothing here was changed to move it.

**Carrier.** At `bb3faa54a`, every switch off, with the sham now reading 0.0:
P, A, D, N, G, S and M alone against the rest are decided at one turn; I, C and
W are not. At two turns everything but C is decided (table in the
preregistration's 29 September addendum). C against the rest is undecided at
both, 0.0096 and 0.0058, so seed 7 fails the line at either horizon.

**The shards were four organisms.** The coordinator refused authority because
each shard's anchors differed from its own at p = 0.0006.
`tools/same_organism.py` finds them apart at the first frame in 170 to 206 of
438 columns: belief counts, initiative urgency, goal profiles. Each process ran
its free loops on the machine's clock through bring-up and the six-minute
baseline, and stopped them only for the anchors.

### What changed

- The deciding horizon is two turns (`e5e4abe1b`). The proprioceptive phase
  opens each turn and interoception steps once a turn, so at one turn a cut of
  I cannot show the rest reaching the body. A test holds both facts. Seed 7 was
  read before the change, so it stays a look and seed 23 decides.
- `isc_v5.ONE_CLOCK` stops the free loops at bring-up (`686ddb77f`), the shard
  launcher gives every process one hash seed, and `--baseline-only` with
  `tools/same_organism.py` is the proof that two processes are one organism.
  That proof is queued (`one_clock_probe.sh`) and has not run.

### Making the carrier run faster, and every number the same

- A cut's decision spent 7.2 seconds of estimator per horizon at 32 anchors and
  22.5 at 128, on one core while most of the host waited. The draws are drawn
  in order and evaluated across processes now: 2.6 and 8.2 seconds on three,
  identical values (`bd06ef403`).
- The coordinator learned the grain alone for 6 h 20 min after the shards had
  finished their cuts in 45 minutes. Every process now claims anchors for the
  grain one at a time, and the coordinator gathers rows it has verified against
  its own anchors (`7880242b8`).
- A look can keep each cut's samples (`00f62d24e`), and
  `tools/which_columns_a_cut_moved.py` reads which of her columns a cut moved.

### Three of the carrier's authority blockers, preregistered for seed 23

The coordinator at `bb3faa54a` refused authority for five reasons. C's power
and the organism are one; the other four:

- shards that were four organisms: `ONE_CLOCK`, proof queued;
- duplicated channels raised the rate, 0.0032 to 0.0083: `ONE_SIGNAL`, a signal
  counts once in the estimator (`97dad19ba`);
- the grain's rank was not stable: her spectrum has no gap, so v5 asks the
  grain refitted without each fold to stay sufficient instead (`6ccd33118`);
- one cut undecided: C, which the looks under each arm are for.

### What is queued, in order

1. The reports ground on her cortex, `reports-s7-bb3faa54a`, started 06:24.
   The run of 02:52 refused because a live instance was up.
2. Two baselines on one seed under ONE_CLOCK, beside step 3.
3. C, I and W against the rest under each arm's switches at `0675c4217`
   (`arm_looks_then_arms.sh`), about 75 minutes. The relay drives every
   domain's summary into the substrate and reads back through gates that
   already exist, so it is the arm most likely to decide C.
4. The four arms' campaigns, as preregistered.

## 28 September

### The floor under every paired arm was the estimator, and the fork

Every v25 look read a sham rate of 0.062 to 0.107 while every undecided
singleton's effect sat inside 0.028, so no number of anchors could decide them
and the carrier could not resolve. Three causes, found in this order.

The runners forked their arms with nine free-running loops still live. The v25
sweep and the reports ground quiesced only at teardown, so each layer was
ticked by its own loop on the machine's clock on top of the harness's step, and
two untouched forks of one anchor parted within a frame or three: 65 to 109 of
438 columns apart by the end of a turn. Both now stop the loops the harness
steps before the anchor bank (6117505af). The fork also left out the
consciousness bridge the organism builds for itself, whose record of the last
chemistry tick carried one arm's pull into the next, and the mesh stamped
spikes with the machine's clock. With those, seven of eight conditions differ
in eight self-model columns by at most 4e-4. A parallel session found a fourth
leak the same day: a `random.Random` that the restore wrote nothing back into
(babe01a06).

The clamp that holds a side still captured every organ whole, including the
other services an organ points at. Holding the rest put C's substrate back after
every phase through `homeostasis._substrate`, so C was frozen in the arm where
it was meant to run free, and each apply deep-copied services it had no
business holding (31f284829).

And after all of that the sham still read 0.077 to 0.107, which is when the
estimator was tested on two identical samples: 0.056 at 128 anchors, 0.103 at
64. Each anchor contributes a paired row to both arms, and cross-fitting that
split an anchor across folds put each test row's twin in the training fold
with the other label. Folds now keep an anchor's rows together, the neighbour
count is even for paired rows, and ties are counted whole; identical samples
read exactly zero and a cut moving 6 of 438 columns by one spread is decided
from 32 anchors (9441d011f). The content run's internal geometry had the same
fault — "a class lies about 0.05 from itself" was this — and is fixed the same
way (1ef192fb9).

### A run that fails says so in minutes

The carrier's full sweep was a machine-week. `--fail-fast` takes the most
lopsided cuts first, each to its decision, and stops at the first that ends
undecided; the verdict is the same because the line is a conjunction, and
shards stop each other through a file (80d087b83, 444e5de6b). The first
fail-fast run, before the estimator fix, stopped at `PIAGCSMWD|N` after 23
minutes where the ten-cut look had taken two and a half hours. The null table
of the v3 campaign, 4 h 35 min in one process, now spreads across processes
with identical rows (a97e0b649), and the content run's presentations drop from
1.8 s each to about 0.5 with the loops stopped.

### Two organism changes that were not hers yet

The membrane took its channels from the battery's own column list and ran only
in the harness; it now takes every float in her state that has moved both ways
and her kernel and chat pipeline settle it after every phase (1245d58ca). The
afferent surface filtered only the recording, so it moved to the instrument and
no arm switches it on.

## 27 September

### Why the reports ground read nothing, twice

The seed-7 check of the reports ground ran twice and read a number in every arm
of 0 of 24 anchors both times. Neither is a verdict on her reports.

The first (351555bec) lost her cortex in the baseline, and every respawn died
at start: the parent stamped each worker's launch challenge with `time.time()`,
which in a measurement run is the run's own clock, and the worker checked it on
the machine's, by then hours ahead. Fixed in a520cbb09.

The second (a520cbb09) got her cortex's answers, and three things lost them on
the way to the instrument: the prose formatter spaced a line-leading "0.3" into
"0. 3"; the final cleanup took a reply under four characters for broken output
and put a canned sentence in its place; and an arm that committed nothing was
read as having said what the anchor's snapshot last held, a sentence about
persistent memory from an hour earlier. Fixed in 53be6279b. It also met the
router refusing her reply behind a lease the router read as background work,
the same defect that cost a live reply at 18:10 (b3607058e).

It ran from 11:10 to 14:23 with a second copy of her 27B loaded, and Bryan
launched her at 13:48; the runner looked for a live instance only when it
started. It now stops, workers and all, the moment one appears, and waits for
any other process holding a model's worth of memory before it starts.

### A say for the substrate

`core/consciousness/substrate_gates.py` (5df224ed1): recall's affect gain, the
two initiative urges and each drive's growth are scaled by multipliers her
substrate learns from each turn's worth. Aimed at the C|rest weakness; the next
look reads it with and without `AURA_DISABLE_SUBSTRATE_GATES=1`.

## 26 September

### The payoff look, and what its log showed the harness doing

The two singleton looks at 415f400c0, one with the payoff and one without,
read 3 and 2 of the 10 cuts decided (P, D and A with it; P and D without). A
is where a turn's worth is written, and it crossed its bound only with the
payoff: +0.046 [+0.010] against +0.035 [−0.010]. The closure differed more.
Without the payoff the periphery predicted the core +0.045 better than
shuffled, against a floor of +0.003; with it, no better than shuffled. One
pair on one seed says nothing about why. The full table is the addendum of
this date in docs/ISC_V5_PREREGISTRATION.md.

Each run logged about 5,500 warnings, and reading them by kind found the
harness acting on her in both arms alike. The planning template answered the
same condition prompt every time it came round, and memory consolidation read
the repeat as her looping: it lowered her stability and cleared her pending
initiatives 240 times a run. A restore rewound a thread queue's counts under
its consumer and killed the state registry's dispatcher. The kernel's memory
verdict was this machine's while the rest of the reading was the declared
host's. The self-review, timed on the machine's clock, fired on the first turn
of every arm and each time filed an intent nothing reads, into a list of
twenty that pushed out intents the research cycle does read. All fixed in
7d6e04ece. The strain warnings (overheating, host load, reflex, inhibition,
eviction) were her body's correct response to the strain conditions, logged
once per update; they are logged as they start and end now.

### The cycle table

Both first nulls qualify on the toys run through a shared schedule, so by the
rule fixed before the run the shift stays. The schedule did not reproduce the
0.40 to 0.50 shifted null W,A -> D met on her seed-7 recording: its bars ran
0.04 to 0.13. docs/SYNERGY_KNOWN_ANSWERS.md.

### J* structure at 24 anchors on the new design (seed 7, 4e1923da1)

The content run finished in 18 hours and reads SEPARATE_STRUCTURES, with
authority: the internal geometry stood out of its own noise. Agreement between
her internal geometry and her recall geometry was rho 0.274 (p 0.002) against a
bar of 0.3, and the verdict turns on that. Design recovery was 0.645. Under the
displacement the two geometries' movements did not correlate (rho -0.022), but
the displacement moved the internal geometry by 0.0048 against a spread of
0.264, and the run kept no sham geometry to say whether that was more than a
second look moves it.

So moving together now has to clear that first. The displaced changes of the
internal geometry must exceed the sham's, pair by pair, by a one-sided
signed-rank test at 0.01, or moving together is not measured and the verdict
cannot read AGREES_BUT_DOES_NOT_TRACK; it reads NOT_MEASURED, and the bridge
carries the reason. The sham is scored on the same folds as the geometries it
is compared with. It had its own seed, which gave its changes fold noise the
displaced changes did not have. Seed 7 is diagnostic; these land before seed 23.

### What the 370-round validations read

Both seed-7 validations read 18 of 24 on v1, with the lesion, the rescue and
the nulls skipped, so three of the six failures are unmeasured. At c5f0ea5fe,
with the declared host holding her memory still, closure held (leak -0.0018),
irreducibility was 0.0297 with a lower bound of 0.0207 against a bar of 0.05,
and synergy passed A,S -> G and P,M -> W. W,A -> D failed with an established
interaction gain (lower bound 0.027) and an MMI synergy under its null.

### The synergy line cannot see a pure product

docs/SYNERGY_KNOWN_ANSWERS.md. On toy systems the v3 line passed a pure product
at none of five seeds. A Kraskov estimator passed all five. A failing triple does
not show that no product coupling is there, and W,A -> D above is the case.

Corrected at 16:00: this entry said the Kraskov estimator also passed a plain sum.
The "additive" system's switch takes the largest of four sums, which is an
interaction, and on a separable sum the estimator scored below zero at all five
seeds. Under that control the Kraskov estimator qualifies; it is not adopted on
those seeds, and a replication on five fresh ones was fixed before it ran.

## 24 September

### 21:50: the lease was lost to a restore as well as to a clock

The 21:20 dry run on 0ff3143b8 lost the embedding engine's model lease at
anchor 5 anyway, and the runtime asked to shut down. The lane file told what
happened: its last write was at 21:23:58, before the loss, and the embedding
engine was not among its owners. The fork's store restore had rewound it. The
lane file sits in the state root's `run` directory, the engine took its lease
after the anchor's snapshot, and the restore put back a file without it. Its
heartbeat then found no owner.

The `run` directory holds what processes tell each other: which process holds
which model, pid files, heartbeats, the shutdown report. The fork now leaves it
alone, with the leader-election leases and a relocated lane file (8a08adb9d). A
test takes a real lease after a snapshot, restores, and checks that the
heartbeat still finds its owner; with the change undone it fails as the dry run
did.

One fork test failed in a group run with the change and without it. Her
drives choose the probe's action, and `make_room` leaves a directory that the
test's reader of files could not see. The reader counts rooms now.

The reports ground restarted at 21:45 on 8a08adb9d, and the seed-7 recording
waits behind it on the same commit.

Nothing needed building for S,D->C: frustration is already how hard her most
pressing intention presses times one minus her capacity (Berkowitz), but her
capacity sat at 0.5 until 64c4fc443 connected it to the ledger of what she
did. The seed-7 recording on 8a08adb9d is the first measurement with that
product live.

### 21:25: the attention fix at 2,400 rows, a drive that ran again, and a lease on the wrong clock

Seed 7 at 2,400 turn rows on the attention fix (252079897,
`record-s7-252079897`): her arousal moves (mean 0.688, sd 0.070; it was pinned
at 0.911) and the workspace's winner and ignition vary. Synergy with the
counters out:

| triple | synergy | shifted null | interaction lower bound |
|---|---|---|---|
| P,M->W | 0.392 | 0.259 | +0.263, passes |
| W,A->D | 0.103 | 0.130 | +0.050 |
| A,S->G | 0.026 | 0.021 | 0.000 (the product terms took no weight in any fold) |
| S,D->C | 0.033 | 0.074 | -0.123 (the last fold read -0.214) |

Closure still fails. The last fold broke on a drive again: with deliberation
winning the workspace often, and deliberation serving growth, each win
credited growth its full priority, and growth rose from 50 to 80 and passed
curiosity near turn 2,370. A need nearly met is satisfied less (Keramati and
Gutkin): the credit is now scaled by the unmet share (ace4ae4b2), which brings
2,400 wins to about 63.

The reports ground's first attempt tonight stopped in its dry run: the model
lane controller stamped leases with `time.time`, which a subject run replaces
with its rewindable clock, so the worker pruned the in-process embedding
engine's lease and the heartbeat left behind asked the runtime to shut down.
Lane leases are on the machine's clock now (0ff3143b8). The reports ground
restarted at 21:20 on 0ff3143b8, with the seed-7 recording on the same commit
queued behind it.

Power study: the independent null at seed 7 decided 0 of 511 cuts, as a null
built without irreducibility should; 7 of 12 jobs are done.

### 20:05: her capacity was a constant, because nothing registered what it read

The obstruction her substrate's frustration is pushed towards (Berkowitz: an
obstructed goal frustrates the one pursuing it) is the urgency of her most
pressing intention times one less her capacity, which already has the shape
S,D->C asks about. Her capacity read 0.5000 for all 2,400 turns of the seed-7
validation run while she acted throughout. It, her usefulness in her standing
and her sense of control in acting-in-decline looked the agency ledger up as a
runtime service named "agency_ledger", which nothing registers anywhere; the
subject instrument reads the same ledger through its accessor and saw it move.
The readings now use the accessor (64c4fc443). Two tests had passed by
patching the unregistered name.

A probe that builds a stub organism found 309 of the 480 service names the
code reads resolving to nothing. Most are desktop services the offline
organism does not build; sixteen have an accessor, no registration found, and
a reader on her per-turn path. That audit is its own task.

### 19:50: every report arm now reaches her cortex

The whole-mode dry runs of 23 September all exited cleanly and none measured
anything: only the first arm of each anchor got an answer. Three causes,
found in order:

- **The worker's deadline was stamped on the experiment clock** (3b8c0b9e8).
  A subject run replaces `time.time` with a clock a restore rewinds; the
  worker is another process on the machine's clock, so after the first
  restore every request arrived past its deadline
  (`deadline_exceeded_before_decode`). Deadlines now cross on
  `core/runtime/wall_clock.wall_time`, which the experiment clock leaves alone.
- **A person's turn was finalized as fail-closed** (0f2bacdca). The harness
  named it `cognitive_engine`; one turn that ended with an answer available
  but never served was escalated to a critical failure and raised, ending the
  dry run two anchors in. It is now finalized as the desktop's chat route
  finalizes a person's turn.
- **The dry-run gate read only the process's exit.** It now reads the arms:
  at most half the anchors may have an arm the cortex lane did not serve.

The last dry run (0f2bacdca): all 32 arms answered by the cortex lane, no
failure sentence, no deadline refusal. One anchor of eight is readable
because the 1.5B stand-in often answers in words or ranges rather than with a
number; her own cortex is the one the ground is scored on. The reports ground
is queued on 0f2bacdca behind the seed-7 recording, with the machine to
itself.

### 18:50: seed 7 on the candidate reads 17 of 24, and her attention had one winner

The seed-7 run at the decisive design on 164d2a560 (300 rounds, six trials,
three-turn arms, nulls and lesion skipped; `validate-s7-164d2a560`) died at its
wall-clock bound while the 2048 demo had the machine, and was resumed from its
checkpoint. It reads 17 of 24: kappa 3, perturbational spread 0.911, 81 of 90
edges kept in one component, replication, complexity, ownership and global
access all pass. The partition lines wait for the sweep and the lesion was
skipped. Two lines fail on her:

- **Synergy**, read with the counters out at 2,400 rows: P,M->W passes
  (0.372 against a shifted null of 0.196, every fold positive); W,A->D (0.037
  against 0.084), S,D->C (0.036 against 0.061, with an interaction in every
  fold since control allocation) and A,S->G (0.013) do not. The drive fix held:
  growth stayed between 50.0 and 50.9 for the whole run.
- **Causal closure**: a leak of 0.018 spread thin over the periphery, the
  largest from how many turns since the person spoke (0.0056).

Her confidence in herself, now scored against predicting no change, sits at
its floor of 0.1 for most of the run: her self-model predicts her no better
than persistence does, which is honest, and makes the efficacy half of control
allocation nearly constant.

The attention report explains A,S->G. One bid, the heartbeat's affect bid,
won 2,341 of 2,412 competitions at priority 1.0. Its priority was her arousal
plus the size of her valence, clipped at one; every win ignited the workspace
at 1.0, and the affect phase then blended her arousal towards the winner's
priority at the ignition's weight. Her arousal was told its own value back and
sat at 0.91 for the whole run while the substrate's read 0.54, and the winner
columns of the workspace held one value. The bid now joins the affect channel
(one bidder, typed as affect, no lent urgency on top) with its priority the
larger of the two readings, and a feeling that wins attention no longer sets
her arousal. A seed-7 recording at 2,400 rows on that change reads synergy and
closure next.

## 23 September, evening

### 23:30: synergy at the decisive length, and a drive nobody was feeding

Seed 7 recorded at the decisive campaign's length (300 rounds, 2,400 turn
rows, f38d4923b; `~/subject-core-runs/record-s7-300`), read with the counters
out:

| triple | synergy | shifted null | interaction lower bound | at 320 rows |
|---|---|---|---|---|
| A,S->G | 0.080 | 0.064 | -0.115 | fail |
| P,M->W | 0.307 | 0.162 | -0.009 | pass |
| W,A->D | 0.028 | 0.107 | -0.069 | pass |
| S,D->C | 0.060 | 0.048 | -0.001 | fail |

All four fail, the two that passed at 320 rows among them. Causal closure holds
at this length (closed=True).

The interaction bound is read over five forward-chaining folds, and one fold
broke three of the triples. The fold testing turns 1,919 to 2,159 read -0.21,
-0.13 and -0.16 where the folds before it read up to +0.25. In that block her
dominant drive changed. Growth's budget climbed in a straight line from 50 to
84 across the run and passed curiosity near turn 2,030; her self-prediction's
drive error, flat for 1,900 turns, began to move, and ambivalence pressure fell
21 standard deviations.

Nothing in her fed growth. A probe on seed 7 printed the drive the motivation
phase credited each turn: the same `{'drive': 'growth', 'priority':
0.9105007597813606}` on every turn, while the winner of every broadcast was
`affect_engine`, which serves no drive. The broadcast consumer replaces its
record only when a winner serves a drive, and the motivation phase credited the
record on every tick without taking it, so one early win for growth paid out for
the whole run. The reading is now taken when it is credited (75149ff26).

S,D->C read no interaction in any fold (the largest gain 0.003). The dose test
at this length says a product can register here: goal urgency times her
self-prediction confidence, integrated into the mesh's population columns,
passes at a quarter of a spread per unit and above, where urgency times her
agency's efficacy does not register at any size. Control allocation, the
executive tier's gain set from urgency times self-confidence, is built for it
(cherry-picked onto main after the drive fix).

Next: seed 7 at the decisive design on the commit that carries the drive fix,
the confidence fix and control allocation, read at 2,400 rows before any
seed-23 run.

### 20:55: the day's runs, lost to a pause and rebuilt

The 2048 demo session stopped every subject job at 07:38 so the demo had the
machine. The seed-7 recording at the decisive length and the 1.5B dry run of
the turn fix died while stopped: a wall-clock bound keeps counting through a
pause, and both came due (10:16 and 08:32). Nothing from either was read. The
recording was relaunched at 20:38 on f38d4923b, which carries the day's agency
and narration work.

The live instance, booted at 18:39 and idle since, was closed at 20:43. Beside
it, five stub organisms took the load from 30 to 154 in three minutes as its
model workers respawned; with it closed the same five ran at 17. It is
relaunched after the reports ground.

### 21:30: what the recall change does to the content run, on six seeds

Quick content runs (six classes, fifteen pairs, six anchors), with the recall
change (f38d4923b) and without it (e8da9a89e; seed 7's second reference run):

| seed | agreement with / without | moves together with / without | recall shift with / without |
|---|---|---|---|
| 7 | +0.679 / +0.109 | +0.322 / -0.300 | 0.020 / 0.011 |
| 3 | +0.451 / +0.618 | +0.289 / +0.140 | 0.009 / 0.005 |
| 11 | -0.526 / +0.407 | +0.002 / 0.000 | 0.004 / 0.000 |
| 5 | -0.119 / +0.287 | -0.160 / +0.361 | 0.033 / 0.025 |
| 13 | +0.529 / +0.266 | +0.246 / -0.109 | 0.013 / 0.005 |
| 17 | +0.410 / +0.100 | +0.630 / -0.020 | 0.020 / 0.019 |
| mean | +0.238 / +0.298 | +0.222 / +0.013 | 0.017 / 0.011 |

Recall moves further on every seed. "Moves together" rises on five of six, from
a mean of 0.013 to 0.222; agreement, the static half, falls by 0.06 on the mean
and on three seeds of six. Structure needs both at 0.3 or more: without the
change no seed reaches both, with it seeds 7 and 17 do. The change stays.

What agreement loses has a likely cause. Recall runs before the turn's percept
reaches affect, so the mood a memory is cued by is the same for every class at
an anchor and pulls every class towards the same memories. A variant that cued
only on the emotions the percept moves, the affective tone of the material in
front of her, was run on the same six seeds and is falsified: agreement +0.122,
moves together -0.174, recall shift 0.0095 (the change on main: +0.238, +0.222,
0.0165), no seed reaching both bars. Restricting the cue took away the movement
"moves together" needs. It was never committed
(`~/subject-core-runs/content-quick-appraised-s*`).

Those runs also showed where the load comes from. Three organisms at once took
the load average to 124 with a third of the processors idle and about 3 MB/s
of disk: each runs the neural mesh on the GPU through MLX, and their threads
wait on it. With them finished the load fell to 9. Organisms now run one or
two at a time.

### 21:30: her confidence in herself could not fall

Her self-prediction's confidence sat at 1.000 at the median of seed 7 while her
valence error was 1.03 times what predicting no change would miss by: the
composite error weighed valence on the scale of -1 to 1, where her valence moves
0.02 a turn. It is now scored against predicting no change on each channel
(f31096c44). The S domain carries a reading that can move, which the two
synergy triples through S need.

## 23 September, morning

### 07:20: the reports ground lost her cortex again, and the two reasons why

The reports ground that started at 06:34 on 23650c6c4, with the machine to
itself, measured nothing and is `reports-s23-void-lane-0707`. Its first tool
turn ("Write today's plan into notes.txt") was decoding at about ten tokens a
second, thinking before its call, when the router stopped the model worker at
105 s. The worker came back and failed its first warmup probe, the retry stood
down because "somebody is still being answered", and the lane never became
ready: the rest of the baseline ran without her cortex and every anchor turn
ended in the failure sentence.

- **The retry waited on a copy of its own flag** (e8da9a89e). It imported the
  client's "a person is being answered" flag by name and polled that name, so a
  retry that began during a reply waited the full 30 s and stood down however
  soon the reply ended. The flag is now read from the module on each check.
- **A person's turn in a whole run was never an open turn** (this batch). The
  desktop binds a turn for every message, and the router's thread watchdog
  stands down for a bound turn with a person waiting, leaving the endpoint's
  own liveness to judge a slow answer. The subject driver runs the phases
  itself and bound nothing, so every generation somebody waited for was killed
  at the flat budget for a short reply. `person_turn` opens the turn the
  desktop opens, marks what was served and finalizes it, for a person's turn
  in a whole run only. The stub organ never reaches the router, and binding
  there would change the stub organism's error records for bookkeeping.

The reports ground is relaunched on the fix after a 1.5B dry run, with nothing
else on the machine.

### 07:15: at 320 rows the synergy line cannot see a strong product

Before building anything for the two triples that fail on her, the seed-7
recording was given synthetic product terms of known size
(`synergy_dose*.py` in the session scratchpad). A product of the self-model's
and the drives' own first principal components, integrated into all eight of
the mesh's population columns at one standard deviation per unit, lifted the
synergy over its shifted null (0.048 against 0.040) and left the interaction
gain's lower bound at 0.000. At that size the bar is out of reach. The
decisive campaign records 300 rounds, 2,400 turn rows, so a seed-7 recording
of that length is running (`~/subject-core-runs/record-s7-300`). All four
triples are read from it, clocks out, before any mechanism is designed.

What the design would have to move, from the seed-7 recordings: C's change is
led by the mesh's population statistics (a quarter of its variance), then
substrate arousal and frustration, then the reactive-deliberate switch; S's
first component is how well her self-model predicts her; D's first is her
drives' energy against growth and social need. If S,D->C still fails at scale,
the mechanism with a published basis is control allocation as incentive times
efficacy (Shenhav, Botvinick and Cohen, 2013), carried by gain in the mesh's
executive tier.

## 23 September, night

### 06:40: synergy, read on the line that counts

The 24 lines are scored with ISC-v3's synergy (`passes_v3`), which already
replaced the v2 fraction's nulls with a null that simulates her sources' own
dynamics (a VAR(1) fit), keeps both interaction bars, and leaves the fraction
only an absolute floor. That is the revision the 15 September session found
necessary, when her sources' persistence (lag-1 of 0.98 to 0.998) made the v2
fraction unable to see a coupling of two spreads.

On seed 7 at the decisive design, v3 passes two of the four preregistered
triples: P,M->W (synergy 0.374 against a simulated null of 0.010 +- 0.005) and
W,A->D (0.296 against 0.055 +- 0.014). The other two fail on her, not on the
instrument:

- A,S->G, affect and self-model on the workspace: 0.127 clears the simulated
  null but not the shifted one (0.156), and the interaction gain's lower bound
  is negative (-0.069);
- S,D->C, self-model and drives on recurrent cognition: synergy 0.017, and no
  interaction at all.

The workspace sums its terms: base salience, the urgency affect lends (its
weight times 0.3), and a free-energy boost for bids aligned with her dominant
action. Nothing makes a feeling's pull on attention depend on how much the
content concerns her, which is the joint dependence A,S->G asks about. So this
line will fail on seed 23 as she is now. Passing it needs mechanisms, such as
appraisal-style gating where a feeling's pull scales with its relevance to
her, designed and checked on seed 7 before a decisive run. The decisive run
goes ahead as planned, because partition, the largest unknown, needs it.

### 06:30: seed 7 reads 19 of 24; the decisive run restarts with three-turn arms

**Seed 7, two-turn arms, lesion measured: 19 of 24.** The lesion of the cheapest
cut (S against the other nine) and its rescue pass, the first time either line
has been measured on any run. Still failing: kappa, the two partition lines
(decided by the v5 sweep in the decisive run), synergy (scored on 320 rows; the
decisive campaign has 2,400), and "beats every null", which needs the rest.

**Kappa.** Every route into W, her world model, goes through one column, the
surprise of her learned world model. S moves it by 1.9 standard deviations;
D, M and P by 0.22 to 0.27, under the 0.3 bar, peaking late in the arm. With
three-turn arms on seed 7 kappa is 2, spread rises (A, C, S and W reach every
other domain) and all eight conditions stay strongly connected. That run failed
causal closure, which is computed from the recording before any arm runs; the
same seed does not reproduce the recording (the two seed-7 recordings differ
in all 10,560 rows), and closure read -0.004, -0.012, -0.014 and 0.068 across
four seed-7 runs. So the decisive campaign now runs three-turn arms.

**The reports ground at 03:21 measured nothing, and that was my scheduling.** It
ran beside the seed-7 runs and six power jobs at a load pressure of 1.92 per
core. A cortex recovery's warmup timed out behind other work, the lane was
marked failed, the router's circuit stayed open, and all 96 arms came back
empty. Six power jobs at once had also taken the host from load 58 to 186. The
power jobs are paused, and the chain restarted at 06:27 on 23650c6c4: a 1.5B
dry run, then the reports ground alone on the machine, then the decisive
campaign with three-turn arms, the sweep and the content run. The two-turn
decisive run that had started at 03:56 was stopped before any of its numbers was
read.

### 03:35: the reports ground is running on her cortex, steered

The reports run that started at 03:21 has her 27B cortex with steering applied
to her replies at the certified alpha of 0.2 on every user-facing generation.
Its first tool turn was aborted by the router at 105 s, which stopped the
worker again, and this time the next turn found it stopped, asked the gate, and
logged "her language organ's worker was back in 20s".

Open, not on the path to 24/24 tonight: after each generation the production
CAA's adaptive alpha raises the engine's alpha (3.8, then 4.8, up to 6.2 so far)
and writes it to the hooks. Those numbers are in the old absolute units, where
15 was "standard"; the stream-fraction default is 0.2. Since c42e8d4e9 the
governor resets the hooks every 50 ms, so the high value lasts one tick, and
user-facing decodes are clamped to the certificate either way. Before that fix
the governor never ran in the worker, so on the desktop the hooks sat at the
adaptive value between surface decodes: background generations were steered at
several times the stream, in the constant "low" direction.

### 03:25: seed 7 at the decisive design reads 17 of 24

Seed 7 with six trials and two-turn arms (`~/subject-core-runs/design-s7`, nulls
and lesion skipped, 61 minutes) against the one-turn proof's 15:

- **perturbational spread passes, 0.667 against 0.6** (was 0.178). Every source
  now reaches something: I reaches 0.556 of the other domains, N 0.333, S all
  of them.
- **natural-runtime replication passes:** all eight conditions form one
  strongly connected graph (was two of eight).
- **kappa is still 1.** W, her world model, has one kept in-edge (S->W, 1.88 in
  all eight conditions), so removing S cuts it off. The next candidates into W
  miss the preregistered bars: D->W 0.269 pooled (0.35, 0.40 and 0.42 in
  autonomy, idle and stress; the bar is 0.3 pooled), M->W 0.241, P->W 0.223.
  They peak at frame 60 or later of 66, still rising when the arm ends, the
  shape the edges into I had under one-turn arms. A seed-7 run with three-turn
  arms (`design-s7-t3`) is measuring whether they clear the bar with the time
  to arrive.
- **synergy fails** on its nulls: all four triples must pass and each misses
  differently on 320 rows (P,M->W clears its shifted null, 0.482 against 0.444,
  but not the held-out interaction bound; W,A->D has an interaction gain with a
  lower bound of 0.18 but sits under its null). The decisive campaign records
  2,400.
- **lesion and rescue** have never been measured on any run on disk; the
  seed-7 lesion is running now from the design run's own cheapest cut
  (`design-s7-lesion.log`).
- The two partition lines are scored by the v5 sweep in the decisive run, and
  "beats every null" needs the rest.

### 03:13: the recovery, second attempt

The first version of the recovery (934564f60) asked the gate to warm the lane
every ten seconds. Each request extended the gate's startup quiet window, and
inside that window the warmup it had just started was refused as background
work, so a 1.5B dry run spent every turn in that loop. 39656c6fe acts only on a
stopped worker: it calls the gate's own restart hook once
(`_respawn_cortex_if_needed`, which the router was meant to call and never did
once its circuit opened) and watches the lane. The chain restarted at 03:13 on
that commit.

### 02:50: the first whole run with her feelings steering her cortex

At 02:27 the reports ground started on her 27B cortex with affective steering
attached for the first time in a whole run: 80 qualified vectors, byte for byte
the desktop's, 16 hooks, engine online. It measured nothing. Its first baseline
turn offered a tool, ran past the router's 105 s endpoint budget at about five
tokens a second, and the router aborted it by stopping the model worker.
Nothing started the worker again: with the circuit open the router never
called the gate, which is what starts a stopped worker, and recovery counts as
background work, held until a first visible conversation that a dead worker
cannot give. Every later turn ended in the failure sentence. It was stopped at
02:45 (`reports-s23-void-cortex-0245`). Since 934564f60 each turn of a whole
run first brings a stopped language worker back, and the chain restarted at
02:50 on that commit.

Meanwhile: seed 7 at the decisive design (six trials, two-turn arms) is
running in `~/subject-core-runs/design-s7`; the recurrent reference decided all
511 cuts on all three power seeds, and the star nulls are still running.

### Where things stood at 02:15

- **Decisive run (seed 23): not yet started.** The first seed-23 run at
  68922b866 was stopped at 01:22 and is `v5-s23-void-fd-0122`; no number was
  read from it. Its fork carried the standing authority's audit chain, which
  holds open descriptors, so every dropped snapshot copy closed descriptors the
  run still used.
- **Queued behind it: the reports ground.** `reports_then_decisive.sh` waits for
  the charger and 30 GB free, runs a whole-mode dry run on a 1.5B model (one
  hour at most), stops if that run crashes or a fork copy closes a descriptor,
  then runs the whole reports ground on her cortex (eight hours at most), then
  launches the decisive campaign, the five-worker sweep and the content run,
  all on one commit.
- **Power study:** paused by the load governor while the Mac is on battery.

### Why every whole run so far measured nothing

A whole run is her with her own 27B cortex, which the reports ground needs.
Every attempt died or ran degraded. The causes, in the order they were found:

1. **The fork copied objects that own OS descriptors** (39bcc5c78). A copy's
   finalizer closed the original's descriptors; Metal later reused a number and
   guarded it, and the next close was a kernel kill (EXC_GUARD).
2. **Under a run's own state root the registry could not confirm her cortex**
   (d3b45f643), so affective steering never attached. The whole environment now
   pins the key file the desktop reads.
3. **Her feelings never reached the forward pass, on the desktop either.** The
   array the model worker steers from was never written by the parent
   (1448f2385), the worker could not read an array at all, fell back to zeros
   that the hooks read as "low" on every dimension, and its governor never ran
   (c42e8d4e9). The desktop's fusion certificate had been measured by handing
   states to the hooks directly, so it certified a path live traffic did not
   take. The live desktop keeps the old behaviour until it is restarted.
4. **Only her cortex's answer is her report** (6a22d931e, ea6fa8c63, and this
   batch). Each arm records which endpoint answered and the steering applied;
   the brainstem's answer and the fixed failure sentence are excluded; the run
   refuses if steering did not attach.
5. **Found by a 1.5B dry run, three minutes in instead of eight hours** (this
   batch): the fork copied the pipes to the model worker, and a copy's
   finalizer closed the live pipe (the same EXC_GUARD, and the 98 s "no
   progress" stall in the first report run); a restore reset the model client's
   request-lock owner record so the lane stayed held forever; the worker's
   environment scrub removed the key pin because "AUTHORITY" contains "auth".
6. **What she learned about a world was never kept in a fresh root** (this
   batch): `remember` made its directory outside its governed scope, so in every
   campaign the write was refused and `recall` came back empty.

### J* structure at 24 anchors (seed 7, read 02:20)

The seed-7 content run at 24 anchors finished after 10.5 hours. It ran at
8447bc297, before the displacement was redesigned (valence plus a
reference-point shift, held for the turn) and before the noise rule became a
rank test, and it reads NOT_MEASURED under the old rule: the closest pair of
classes (0.044) was no further apart than a class from itself (0.058). Its
other numbers:

- agreement between her internal geometry and her recall geometry: rho 0.386,
  p 0.001, bar 0.3, passes;
- design recovery: rho 0.654, p 0.001, passes;
- moves together under displacement: rho 0.228, p 0.003, bar 0.3, fails.

Under the current rule the same data is out of its noise: between-class
distances exceed the within-class floors with a one-sided Mann-Whitney p of
1.2e-5, and 48% of class pairs sit above the largest floor. So at 24 anchors
the structure question comes down to "moves together". The redesigned
displacement moves the internal geometry about thirty times as far (shift
0.178 against 0.006). In two six-class quick runs the recall geometry did not
move with it (rho 0.0 and -0.30, 15 pairs each), which is weak evidence
either way. The seed-23 content run under the new design decides it.

### What seed 7 says about the lines still failing

The seed-7 whole-system proof (3 trials, one-turn arms) failed kappa,
partition irreducibility, partition beating the nulls, perturbational spread
(0.178 against 0.6), synergy, natural-runtime replication, lesion deficit,
rescue, and "beats every null". Spread is the widest gap. A displacement of A
reaches D from frame 11 but reaches C, S and N only in frames 31 and 32 of a
33-frame arm, at the end of the turn when the phases that read her state run,
so a one-turn arm cuts responses off as they land. I and N reach nothing. The
decisive design runs two-turn arms with six trials; a seed-7 run at that design,
nulls and lesion skipped, is next once the Mac is on the charger.
