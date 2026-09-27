# Balanced polarity controls for 27B CAA development

The contrastive extraction path in `G10_CONTRASTIVE_DEVELOPMENT_PIPELINE_2026-09-27.md`
made eight independent sign flips over eight paired differences. Those flips
were not necessarily balanced. If every pair has the same difference vector,
an unbalanced flip produces exactly the treatment direction or its negative
after normalization. It is not an informative negative control. A synthetic
eight-pair replay produced treatment cosines of `-1` and `+1` among the
purported nulls.

Polarity controls now use distinct, balanced within-pair label assignments.
For small pair sets, the finite assignment space is enumerated before seeded
ordering; larger spaces use a bounded unique sample. Zero and treatment-collinear
directions are rejected. If the declared number of controls cannot be formed,
capture fails instead of publishing an incomplete generation. Each captured
control records its signed cosine with the unnormalized target direction.

This is a *negative-control construction*, not a permutation-test null
distribution: excluding collinear assignments changes the distribution. The
development campaign still has to measure these controls, the zero/random
vectors, rich-text comparator, and capability battery. No new model-active
measurement, sealed evaluation, or serving authority follows from this repair.

Focused tests cover identical-pair refusal, balanced noncollinear controls,
determinism, complete capture geometry, and development selection. The
current-model lane was occupied by an independent subject-core run during this
CPU repair.
