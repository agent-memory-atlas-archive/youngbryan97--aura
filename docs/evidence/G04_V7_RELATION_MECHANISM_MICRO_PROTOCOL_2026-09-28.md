# V7 relation mechanism micro-probe, 2026-09-28

This protocol adds nine controlled requests to the already frozen
[three-reference probe](G04_V7_TARGET_BLIND_MICRO_PROTOCOL_2026-09-28.md).
It was fixed during the v7 fit, before checkpoint selection or v7 generation.
It tests the proposed mechanism on a small scale, not broad qualification.

The seed remains `2147438394`. Each scalar, lookup, and count reference gets
three controls, in that order: a meaning-preserving definition paraphrase,
one role reversal, and one dependency change. Public values stay identical.
Every changed target differs at exactly one binding, preserves operation
identities, and has a distinct observed answer. Targets are used only for
grading after generation, never for search or selection.

| Domain | Control | Source SHA-256 |
| --- | --- | --- |
| Scalar | Paraphrase | `ecf3cafb16e5a97d2400dbd0f553d18d1565f0355901325513453167c11f65e3` |
| Scalar | Role | `5ea5cffc115e10808a581c10916381170f65ede8befa3afe8a564df405b7f855` |
| Scalar | Dependency | `2da01cb64e1d765c483fe24e7c81c6665993667e10a75404e50f25f963831748` |
| Lookup | Paraphrase | `6092ae46181c78e85d80fc51473b2169ce1e9dc6f60d87817e6f524a3e5dbbbb` |
| Lookup | Role | `d80c17c65f7e244fb792960d795b8bbe56fa29adda00f7eeef8b3e2116ac746c` |
| Lookup | Dependency | `8588803e1485c9b67ee31a59eb9da39dd332890400d58fee5cbfb04632de6c91` |
| Count | Paraphrase | `840b4a8de68d175c4805db4a33ca511d63315bcab19f85efc5f32d4535ac65df` |
| Count | Role | `7ebf31c46530b1df1b0b0f49dd3ccbe8a1045640f5e5ed4d660e066888bf29c1` |
| Count | Dependency | `f5136cf913890800917a959a1e4f97ebe2eb9698e301b87bb8aed2f573300ce7` |

All nine sources are absent from the v7 fit, calibration, and held IDs.
Reconstruction of the eight pinned source manifests found 53 constructions
in those partitions; none is a reference or controlled construction here.
All twelve intended programs have admitted teacher paths in the role-relative
grammar. These checks establish separation and structural feasibility, not
model accuracy or bounded target reach.

## Ordered execution

1. Complete and verify the v7 fit and source-only checkpoint selection.
2. Run the original three-request fitted, base, and fitted-erasure arms under
   their existing protocol. Preserve their plans and verified rows.
3. If all three fitted reference interpretations are exact with correct public
   values and no forced completion, run the nine controls. Use
   `tools/evaluate_semantic_native_grammar.py --dataset relation_transfer_controls
   --seed 2147438394 --canary 9 --weight-mode fitted --max-steps 8
   --search-completions 4 --search-nodes 256 --search-score-mode native_nonpositive
   --prefix-strategy full --source-evidence source_text --max-seconds 3600`,
   with the same training directory and selected checkpoint.
4. Independently verify the nine rows. Run the existing fit comparator on the
   original three arms, adding `--relation-controls-directory` for the nine
   new rows. Do not decode the reference stage again.

The mechanism micro-probe passes only if all nine controls and all three
reference interpretations recover their intended procedure and observed
public answer, with complete verification and no forced completion. Each
control must meet the same frozen search budget. The comparator refuses a
changed model, checkpoint, seed, scoring rule, common code hash, or incomplete
cohort. Failed controls remain attributed to paraphrase, role, or dependency;
a pooled count cannot hide one of those failures.

The original matched base and erasure results still decide whether the three
reference successes include a fitted gain and whether it depends on source
text. The controls do not separately measure a gain over base. Their purpose
is to test whether the fitted interpretation is stable under new wording and
responsive to a changed relation. No G04 closure, fusion, serving, broad-gain,
or frontier claim follows from this diagnostic.

## Why This Is More Predictive

An isolated right answer may be a numerical coincidence or a familiar output
pattern. This check requires the same relational graph under changed wording
and a different correct graph under changed role or dependency, across scalar
and two sequence domains at three-step depth. It can falsify those mechanism
claims quickly and localize their failure. A clean result supports advancing
the same frozen candidate to wider tests; it does not turn the wider task
distribution into a repeated copy of these twelve cases.
