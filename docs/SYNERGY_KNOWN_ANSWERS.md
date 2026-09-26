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

The Kraskov line passed the additive systems at every seed. MMI synergy does
that: for a sum of independent sources the joint information exceeds either
source's alone, and the difference is scored as synergy. The v3 line rejects
sums through its second leg, the held-out interaction gain. A third estimator
was added after those rows were read, and held to the same rule:

- **ksg+gain**: ksg, a Kraskov fraction of at least 0.10, and the v3 interaction
  gain positive with a positive lower bound.

## Result

Passes out of five seeds:

| estimator | product | mixed | additive | none | qualifies |
|---|---|---|---|---|---|
| v3 | 0 | 2 | 0 | 0 | no |
| ksg | 5 | 5 | 5 | 1 | no |
| ksg+gain | 1 | 2 | 0 | 0 | no |

On the products, the v3 line's MMI synergy was between -0.0015 and -0.0010, at
its null. The Kraskov synergy was between 0.152 and 0.188 against bars near 0.05.
On the sums it was between 0.044 and 0.105, against bars between 0.037 and 0.053.
The interaction gain on the products was positive at three of five seeds and its
lower bound cleared zero at one.

## What it licenses

None qualifies, so the line stays as it is. The v3 line cannot register a pure
product at this length. The Kraskov estimator can see a product, and cannot tell
it from a sum. The interaction gain can tell them apart and lacks the power to
establish a product at 2,400 rows.

For her runs this means a failing synergy triple does not show that no product
coupling is there. W,A -> D is the case in point.

The next candidate keeps the Kraskov estimate and replaces the gain with a null
built from the additive part of the target: fit the best additive model of the
target on the sources, and score the Kraskov synergy that the fitted additive
target alone produces. A product clears that null and a sum does not. It is a
new estimator, so it gets its own known-answers table here, on these systems and
seeds, before anything of hers is read with it.
