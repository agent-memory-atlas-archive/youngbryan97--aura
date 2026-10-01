# G03 Binding Build Register

This register covers the latest two semantic-binding batches, the LoRA
question, and Bryan's rigid-object-permanence add-on. The exact 28 input files
are recorded in [the input manifest](G03_BINDING_ADVISORY_INPUTS_2026-09-30.json).
They were read in full during this task. This is not a claim that every older
video, report or attachment has been re-read or implemented in this pass.
External numerical reports are advisory claims, not Aura execution receipts.

G03 remains open. Building mechanisms, fitting a checkpoint and proving
implementation contracts are different from learned language correctness,
unseen-construction transfer, broad gain, fusion or frontier performance.

## Integrated Architecture

```text
public source and existing learned operation/span proposals
  -> actual recorded features OR frozen prefix + native suffix depth taps
  -> source-qualified entities, definitions, occurrences and operand roles
  -> shared multi-depth role/mention/candidate product pointer
  -> explicit relational graph refinement
  -> candidate-conditioned mention and definition evidence
  -> the EXISTING full SSA chart solver + contextual constraints
  -> best distinct graph, measured energy margin, alternatives/abstention
  -> the existing typed program and floor execution boundary

source-only local + witnessed whole-graph supervision
  -> balanced source environments and optional nuisance adversary
  -> source-witnessed equivariance and baseline retention
  -> joint gradient into pointer AND actual native suffix adapter sites
  -> one selected source-calibration checkpoint with implementation custody
  -> reload into caller-owned suffix + pointer + ordinary decode chart bridge
```

The bridge is opt-in through `decode(binding_chart_solver=...)`. It modifies
neither the default live runtime nor a frozen launch checkout. A loaded
research engine has no serving or release authority. The native suffix must
be supplied by its model-lane owner; the loader does not allocate another 27B
model. The cached-feature fit CLI never loads model weights.

### Concrete Connections

- `tools/semantic_native_adapters.py` is shared by native fitting, calibration,
  replay and causal probes. Contracts specify depth, per-layer ranks, scaling,
  layer-type-aware sites and optional mixed layer function classes.
- `tools/semantic_native_adapter_layers.py` implements SiLU, bilinear-product,
  routed experts, residual-gated DoRA, square compression/expansion and dense
  updates. LoRA remains available as a control. Every adapter retains an exact
  zero-residual baseline lesion; nonlinear adapters reject fixed-matrix fusion.
- `NativeDecoderSuffix.layer_states` taps real distinct depths without feeding
  normalized taps back into the transformer. Hybrid masks remain native.
- `semantic_grounded_binding_engine.py` trains and checkpoints the actual
  pointer, optionally together with the adapter-only native suffix. Capture
  refresh occurs inside the gradient; projection remains differentiable.
- `semantic_grounded_chart_bridge.py` conditions on every actual chart option,
  including alternative definition spans. It does not pick one old winner
  before the new evidence is available. The ordinary decoder's assignment
  path accepts the bridge; unsafe old-score pruning is disabled on that path.
- `semantic_context_binding.py` adds source identity, scope, time, subtype,
  agreement, alias ambiguity and declared relations to the same SSA solver.
  No blanket one-to-one binding rule or successful-incomplete solve is used.
- `semantic_grounded_binding_acquisition.py` separates inference acquisition
  from teacher-label reading. Source-only graph supervision can optionally
  retain proved equivalent graphs and omit unknown-equivalence negatives.
- `semantic_binding_tensor.py` retains exact role-addressed vectors AND source
  keys. It uses an exact or well-conditioned independent role basis and dual
  unbinding, not normalized random vectors presumed orthogonal.
- `rigid_object_memory.py` reuses `TrackStore`, preserves intrinsic geometry
  under viewpoint/visibility changes, records prediction provenance, and
  supplies retained identities to `BindingContext` through `context_referents`.

## Adapter Resource Calculation

The configured checkpoint is Qwen3.5-text geometry: 64 layers, 5120 hidden
width, 17408 intermediate width, 48 linear-attention and 16 full-attention
blocks. Its final eight contain six linear and two full-attention blocks.
The native projections, including gated dense Q, are taken from the actual
config and installed model topology, not assumed to be identical Q/K/V/O.

For ordinary LoRA at all declared native attention/mixing and dense MLP sites:

| Rank | Trainable Parameters | Five Float32 Copies |
| --- | ---: | ---: |
| 8 | 7,295,488 | 145,909,760 bytes |
| 32 | 29,181,952 | 583,639,040 bytes |
| 64 | 58,363,904 | 1,167,278,080 bytes |

Five copies budget parameters, gradients, two Adam moments and a retained
baseline. They do NOT bound peak activations, backbone residency, temporary
workspaces, caches or the pointer. Unknown geometries return unmeasured, not
zero. Native plans include this projection, compare it with actual installed
parameter counts, and reject even state-only budgets exceeding the observed
host envelope. Sequence length, suffix depth and precision still need actual
peak measurement. This arithmetic does not certify that a fit will succeed.

## Disposition Of The Distinct Mechanisms

Each row is an engineering disposition, not an empirical generality claim.
Related proposals share a row only when their requested mechanism is the same.
Original obligations remain linked through the exact manifest and source map.

| Idea | Disposition / Implementation |
| --- | --- |
| Distribute adaptation across several native layers | Built: suffix depth, heterogeneous rank schedule, exact shared replay contract. |
| Raise rank rather than treating rank 8 as a ceiling | Built: arbitrary positive ranks; prospective 8/16/32/64 source-calibration plan. |
| Attention + MLP sites, including keys and gates | Built: dense Q/K/V/O and MLP gate/up/down. Hybrid blocks use native qkv/z/b/a/out mixing sites. |
| Rank-stabilized scaling | Built: alpha/sqrt(rank), distinct from alpha/r and historical direct multiplier. |
| LoRA+ unequal factor rates | Built: separate AdamW rate only for B factors; single-matrix configurations reject meaningless B-rate requests. |
| DoRA magnitude/direction adaptation | Built with separate residual lesion gate; native frozen base retained. |
| Nonlinear bottleneck | Built: SiLU native adapter and source-fitted tanh residual in existing triadic heads. |
| Bilinear/quadratic interactions | Built: product native adapter and joint operation/mention/candidate pointer features. |
| Routed nonlinear experts | Built: learned routing and real-call balancing loss in native training. |
| Combine mechanisms rather than choose a disconnected method | Built: mixed layer classes within one suffix and jointly trained pointer/graph system; not all settings are assumed beneficial. |
| Higher-rank update under a small budget | Built: square compression/expansion, explicitly not represented as an exact MoRA implementation. |
| Remove affine rank restriction entirely | Built: optional dense trainable residual with state projection and frozen base custody. |
| Multi-depth pointer / learned depth gates | Built on observed native taps or recorded equal-width acquired channels. One depth is not cloned as several. |
| Entity/operation/mention trilinear evidence | Built: candidate-conditioned product features, not unconditioned token similarity. |
| Candidate permutation equivariance | Built: shared scorer, identity-qualified maps, graph node permutation. No candidate index enters the pointer. |
| Joint recurrent semantic graph | Built: bounded 0-8 rounds; no neighbor means no invented message. |
| Local + whole-graph training | Built in one fit objective; complete graph positives remain separate from independent slot marginals. |
| Train difficult aliases/reversals/dependencies | Reuse: source corpus augmentation, counterfactual corpus, graph contrasts and existing operation/span learner; new acquisition includes computed-register candidates, not only first-step inputs. New native acquisition still needs a measured fit. |
| Invariant / counterfactual supervision | Built: explicit source-only role/entity bijections, symmetric aligned choice loss; automatic pairing only where exact program/input alignment is witnessed. |
| Nuisance projection / geometry erasure | Built: measured source-intervention SVD basis and differentiable projection. No fixed claim about 8-16 positional RoPE dimensions. |
| Domain-adversarial nuisance removal | Built: training-only nuisance classifier and reversed representation gradients. Environment labels do not enter served features. |
| Group DRO / weak environment emphasis | Built: source-group count compensation and exponentiated source-risk weights. No theorem of unseen-family success asserted. |
| IRM-like stationary score scale | Built: explicitly named squared scale-risk derivative, not claimed to prove causal invariance. |
| Explicit identity / equal-value distractors | Built: namespace + source identity, never denotation-based merging. |
| Alias / coreference / entity linking | Built constraints and ambiguity preservation; existing parser/retrieval owns evidence discovery. No alias match alone creates a new observation. |
| Scope, types, agreement, temporal eligibility | Built contextual admission and relation constraints before learned ranking. |
| Hard/soft same, different, reachability relations | Built pair factors in the existing constrained optimizer. No independent solver replaces SSA authority. |
| Parser-confidence fallback | Built explicit supplied evidence switching; low confidence never relaxes type/scope constraints. |
| Best distinct graph and uncertainty | Built runner-up exclusion and energy-gap threshold. Gap is not an automatically calibrated probability. |
| Formal perturbation robustness | Built margin bound: retain a choice only if gap exceeds twice summed per-role energy bounds. Applies to declared energies, not unknown language truth. |
| Tensor role/filler storage | Built exact/dual-basis recovery, source-key preservation and serialization. Does not discover an interpretation. |
| Full linear capacity diagnosis | Built observed-space reduced-rank regression with exact inaccessible + spectral-tail error; not a transformer ceiling. |
| Source-only fit/calibration and no held tuning | Built verified bank/fold custody, explicit partition validation, source-calibration selection. |
| Baseline-correct retention | Built source-witnessed retention objective and exact source-capacity selection gate. Candidate retention is measured, not guaranteed by keeping an old answer available. |
| Exact lesion and checkpoint replay | Built shared adapter installation/scaling, zero residual gate and combined adapter/pointer artifact validation. |
| Object permanence / hidden geometry | Built object-relative landmarks, retained observed surface properties and full capsule extent linked to existing tracked identities. |
| Relative/world/observer frames | Built normalized rigid transforms and composition; camera motion does not change object-local coordinates. |
| Hidden motion / changing world | Built explicit predicted pose updates with ordered time, source and uncertainty. Deformation requires a new observation. |
| Collision despite end-on visibility | Built continuous translation solve over complete extent; rotation/translation sweep uses a Lipschitz clearance bound and reports unresolved contacts as possible. |
| Apply persistence to linguistic identity | Built shared source-qualified binding context; word order, aliases and visibility do not erase identities. This does not imply an English parser already understands every transformed description. |
| Learned dynamic rank allocation (AdaLoRA) | Open alternative: a static heterogeneous rank schedule is not AdaLoRA. Requires source-only allocation and checkpoint/optimizer migration. |
| PiSSA / ReLoRA / RandLoRA / Monarch exact implementations | Open alternatives, not copied under false names. The useful capacity/function-class changes have implementations above; algorithm-specific initializers/consolidation still need distinct custody. |
| Partial base/norm unfreezing | Not enabled in this frozen-backbone contract. Dense residual provides a controlled full-matrix alternative; changing frozen base ownership requires a new explicit training/serving contract. |
| Fast/slow consolidation and durable episodic retrieval | Existing Aura subsystems are not automatically wired by adding the pointer. Durable tensor/object storage is built; causal episodic acquisition/retrieval integration remains open. |
| Generic neural frame/span discovery | Reuses the existing learned operation/span proposal path. The new binder does not create spans or solve arbitrary English solely from labeled graphs. Broader acquisition and its learned accuracy remain open. |

## Original Source Map And Corrections

Manifest entries `7559...` and `8ceffd...` motivate larger/deeper adaptation and
grounded relational binding. `4f5c...`, `dadc...`, `ac963...`, `6f9f...`,
`d8ed...`, `ec4c...`, `semantic_binder.py`, `Tesb.py`, `Proof.txt` and their
JSON reports motivate type/scope/time/coreference, exact assignment,
fallback, perturbation bounds, ambiguity and control tests. The manifest
retains both duplicate reports instead of silently discarding an obligation.

The later batch is mapped as follows:

- `c7e127...`: tensor-role storage and type-theoretic checks. Normalized random
  roles are not orthogonal. Independent fillers are unnecessary for vector
  recovery, but vector equality does not establish entity identity. The supplied
  parser is a labeled-frame stub; its result is not general NL parsing. Its
  covariance generalization bound and blanket type-checking undecidability
  assertion are not adopted as established theorems.
- `8769ab...`: distributed nonlinear/product/routed capacity. The supplied
  example has inconsistent dimensions and different random train/test
  encoders; its reported numbers are not valid native transfer evidence.
- `05c696...` and `e23c9...`: nonlinear relational interaction and deeper
  computation. A one-hot conjunction is linearly separable; its majority
  baseline is 98.4%, not a 50% XOR bound. Nonlinear adapters cannot be exactly
  fused by averaging a single fixed matrix over input-dependent updates.
- `ba9ae...`: alternatives to a one-layer low-rank contract, source calibration,
  regression guards and broader target sites. Algorithm names are not treated
  as mandatory synonyms for an unrelated implementation.
- `89ef...`: higher-capacity structured updates. Two fixed-width Monarch
  factors do not parameterize every dense matrix merely by being composed.
  No universal rank-50 requirement for semantic binding is assumed.
- `bfdf9...`: multi-layer projections and explicit relational heads. Its claim
  that Aura's existing final decoder adapter is only an affine vocabulary
  readout is false: native Q/V/O/down updates occur inside nonlinear blocks.
- `c608ba...`: the integrated native/pointer/refinement/source-objective
  architecture, controls and compute-aware escalation. The ten concrete
  build points are represented above; native efficacy remains unmeasured.
- `prove_binding.py` and `binding_adapter_results.json`: controlled scalar
  readout examples. A scalar output matrix has rank at most one; rank-8 is not
  a meaningful upper-bound diagnosis there. Added factors are not necessarily
  destroyed information, and polynomial OLS is not a universal ceiling.
- `aura_binding_upgrade_experiment_full.py`, the two CSVs and executed report:
  candidate matching under fixed synthetic encodings is a useful controlled
  example, not proof of natural-language construction transfer. Truncating an
  unwhitened coefficient SVD is not prediction-optimal for anisotropic inputs;
  the new exact surrogate works in the observed feature column space.

Relevant original methods were checked against primary references:
[LoRA](https://arxiv.org/abs/2106.09685),
[rsLoRA](https://arxiv.org/abs/2312.03732),
[DoRA](https://arxiv.org/abs/2402.09353),
[LoRA+](https://arxiv.org/abs/2402.12354),
[MoRA](https://arxiv.org/abs/2405.12130),
[PiSSA](https://arxiv.org/abs/2404.02948), and
[tensor-product binding](https://www.sciencedirect.com/science/article/pii/000437029090007M).
Their papers do not certify Aura's implementation or performance.

## Historical Custody And Remaining Gates

The completed recovered source fit and diagnostic bank are not repeated.
The bank's bounded selected result was 15/21 versus ordinary 16/21; reachable
was 17/21. This was a negative promotion result. The native environment retry
then failed before model loading because the exact resident descriptor was
unavailable; its contained terminal preparation receipt is
`350faeb5b4cb645c736e1fb7787b72c7176cfcf8eede84e5131244bacf24e6fa`.
Frozen launch source and historical receipts remain untouched.

Before wider testing, finish the integrated build QA, fresh native source
acquisition/custody and explicit adoption decisions for remaining alternatives.
Do not call an API, fit preparation or a zero-step checkpoint semantic success.
The prospective acceptance sequence remains positive-step fit verification,
16-node three-arm reference screen, sequential 256-node reference/relation/
retained decode and controls, larger development, fresh transfer, then later
G-ledger gain/fusion/frontier gates. No broad validation has been restarted by
this build pass.

## Role-Query Batch Addendum

The [eight-source manifest](G03_ROLE_QUERY_ADVISORY_INPUTS_2026-09-30.json)
preserves every later role-query input. The ten-page report, JSON analysis,
both byte-identical V2 implementations, the older implementation and all three
pasted analyses were read fully. Reference scripts were inspected and executed
separately; their reported numbers are not native Aura evidence.

All three analyses and the V2 scripts motivate explicit operation-role-filler
conditioning, source identity, whole-graph competition and role-swap controls.
The integrated pointer now has an explicit role query in addition to operation,
mention and candidate states. Acquisition supplies `operation:operand:slot`
queries at every step, including computed-register candidates. The ordinary
SSA chart consumes this evidence. Per-role and complete-graph losses share the
same fit; source-only balanced sampling records actual visited sources and
does not call a partial epoch complete.

The useful mechanisms were adapted rather than copied with their mistakes:

- A diagonal trilinear term is not an exact recoverable full tensor product.
  Exact event storage reuses the separate dual-basis tensor module.
- Distinct or orthogonal roles can still receive equal learned scores; they
  cannot set the role-confusion probability to zero by construction.
- Valid self-reference, self-transfer and reentrant operands are not globally
  banned. A declared frame constraint must supply distinctness when required.
- Arithmetic results need not be positive. Only an explicit available public
  condition may exclude negative results; a verifier may not invent that goal.
- Cached/synthetic role separation is an implementation check, not a proof of
  raw-language parsing, native held-family transfer or G03 success.

## Nonlinear-Meaning Addendum

The three latest Arrival/meaning attachments were also read fully and hashed
in [their manifest](G03_ARRIVAL_ADVISORY_INPUTS_2026-09-30.json).
[The nonlinear-meaning build record](G03_NONLINEAR_MEANING_BUILD_2026-09-30.md)
maps the viable mechanisms, corrections and explicit remaining boundaries.
It adds whole-graph representation/revision, temporal closure, contextual
symbol acquisition, discourse smoothing, bounded global endpoint admission,
actual shared binder/floor calls, governed retention and independent replay.
No broad or held validation was launched for these additions.
