# Can the synergy line see a product?

A study on toy systems, 26 September 2026. It was designed before any estimator
other than the v3 line had been run on a recording of hers, and nothing of hers
was read with any estimator in it. `tools/synergy_known_answers.py` reproduces
the table; `tests/test_the_synergy_line_cannot_see_a_pure_product.py` pins its
answers on one seed.

## Why

Synergy is meant to find two sources that together say something about a target
that neither says alone. The simplest such coupling is a product. The v3 line
reads MMI synergy through a Gaussian copula, which keeps only rank correlations,
and a zero-mean product of two independent sources has none, with either source
or with their sum. So the line may be blind to the one coupling it exists to
find. On the seed-7 validation at c5f0ea5fe, W,A -> D had an established
interaction gain (lower bound 0.027) while its MMI synergy, 0.019, sat under
its null bar of 0.041.

## Design

Four systems, 2,400 turn rows each (a 300-round campaign), five seeds (3, 7, 11,
19, 23). W and A are three slow AR(1) channels each. The target D holds a
four-state switch and four continuous columns, driven by W_i * A_j (product),
(W_i - A_j) + W_i * A_j (mixed), (W_i + A_j) / 2 (additive), or by nothing either
source does (none). Every system is read W,A -> D on the change.

Estimators:

- **v3**: the battery's line, `SynergyReport.passes_v3`, its own shifted null and
  bootstrap included.
- **ksg**: MMI synergy through a Kraskov estimator (Kraskov, Stoegbauer and
  Grassberger 2004, estimator 1, max norm, k = 3) on the same copula-normal
  components, against the same shifted null, 200 draws, at the 0.99 quantile.

An estimator qualifies when it passes product and mixed at four of five seeds and
passes additive and none at no more than one. One that passes additive is
reading a sum as synergy; one that passes none is reading the rig.

### Added after the first rows were read

The Kraskov line passed the additive systems at every seed. That was read at
the time as MMI scoring a sum as synergy, and the correction below shows it was
not. On that reading, and since the v3 line rejects sums through its second leg,
the held-out interaction gain, a third estimator was added after those rows were
read, and held to the same rule:

- **ksg+gain**: ksg, a Kraskov fraction of at least 0.10, and the v3 interaction
  gain positive with a positive lower bound.

## Result, as first read

Passes out of five seeds:

| estimator | product | mixed | additive | none | qualifies |
|---|---|---|---|---|---|
| v3 | 0 | 2 | 0 | 0 | no |
| ksg | 5 | 5 | 5 | 1 | no |
| ksg+gain | 1 | 2 | 0 | 0 | no |

On the products, the v3 line's MMI synergy was between -0.0015 and -0.0010, at
its null. The Kraskov synergy was between 0.152 and 0.188 against bars near 0.05.
The interaction gain on the products was positive at three of five seeds and its
lower bound cleared zero at one.

This version of the page read the additive column as the Kraskov estimator
scoring a sum as synergy, and said it could see a product and not tell it from
a sum. That was wrong, and the correction follows.

## Correction: the additive system was not a sum

In `additive` the switch takes the largest of four sums. Which of four sums is
largest is not a sum of anything: it is a threshold across W and A together, an
interaction. So the negative control carried a real coupling, the Kraskov
estimator was right to pass it, and the v3 line's 0 of 5 there was a miss rather
than a rejection.

A clean sum was added after this was seen, and the rule changed with it:
`separable` gives the switch W alone and keeps the levels (W_i + A_j) / 2, so
nothing D does depends on W and A together. The rule reads separable where it
read additive. The additive-surrogate null named below as the next candidate was
run on all five systems at the same time.

| estimator | product | mixed | additive (a threshold of sums) | separable | none | qualifies |
|---|---|---|---|---|---|---|
| v3 | 0 | 2 | 0 | 0 | 0 | no |
| ksg | 5 | 5 | 5 | 0 | 1 | yes |
| ksg+gain | 1 | 2 | 0 | 0 | 0 | no |
| ksg-over-sum | 5 | 5 | 5 | 0 | 1 | yes |

On the separable sums the Kraskov synergy was negative at every seed, between
-0.18 and -0.067, against bars near 0.04. The additive-surrogate null
(ksg-over-sum: the best additive fit of the target, its degree chosen by held-out
loss, plus its residual permuted) changed no outcome on these systems.

## What it licenses

The v3 line cannot register a pure product at this length, and it missed the
threshold of sums as well. The interaction gain can tell a product from a sum
and lacks the power to establish one at 2,400 rows. For her runs this means a
failing synergy triple does not show that no product coupling is there. W,A -> D
is the case in point.

Two estimators qualify, but only under a control and a rule changed after the
first rows were read. They are not adopted on these seeds. The replication below
was fixed before it ran.

## Replication, fixed before it ran

Seeds 29, 31, 37, 41 and 43, the same five systems, the same code
(`tools/synergy_known_answers.py --seeds 29,31,37,41,43`). The rule is the
corrected one: product and mixed at four of five, separable and none at no more
than one. An estimator that qualifies again is written up as an amendment to the
synergy line, beside v3 and not in place of it, with both tables as its controls,
before anything of hers is read with it. One that does not is dropped.
