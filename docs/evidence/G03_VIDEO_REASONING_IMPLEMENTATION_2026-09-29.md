# Source-indexed reasoning review, 2026-09-29

This is an implementation ledger, not evidence of a G03 gain. The user-supplied
recordings are examples of people *reporting* or *demonstrating* reasoning.
Narration, edited scenes, dramatic reconstructions, and an ASR transcript are
not direct access to an internal cognitive process. The timestamped transcripts
under `/Users/bryan/.aura/rlc-evidence/g03-video-reasoning-review-20260929/`
are SHA-256-bound to the local files. Visuals were inspected at selected frames;
this is **not** an exhaustive frame-by-frame visual audit. In particular,
fictional puzzle theories and medical guesses are not Aura answer keys.

## Mechanisms taken into the semantic path

| Source observation | Reusable mechanism | Implementation and boundary |
| --- | --- | --- |
| Spirit operations-readiness test, `969ab5bf3c3648f3`, 1:29-2:04: operators were denied access to the test bed and had to use flight-like telemetry and commands. | Rehearsal must preserve the information and action boundary of deployment; a simulator cannot pass by peeking. | Existing source-only construction folds and target-blind decode remain the gate. The new counterfactual corpus admits only train construction/topology sources; its held sources supply neither pairs nor labels. No rehearsal proves transfer by itself. |
| Spirit ground simulation, `6b899fd077563309`, 0:19-1:22: recreate conditions and test actions that could worsen the rover state. Apollo 13, `5d5df6114d269c48`, 19:38-20:23: novel power-up procedures were simulated before use. | Predict state transitions, record risk and test a discriminating intervention before acting. | `predict_meaning_trajectory` executes every proposed prefix on the canonical floor with stable register identities. `MeaningStageInquiry` now plans discriminating intermediate observations; selection changes only after caller-supplied feedback. Simulated consequences are not observations of intended source meaning. |
| Clinical problem-solving, `a205a6443cfe45ef`, throughout: maintain common and cannot-miss hypotheses, inspect chronology and anomalies, and revise when later culture/autopsy evidence arrives. | Keep competing causal accounts and update only the claim touched by a new observation; severity and probability are different axes. | `SemanticProgramPortfolio` retains proposals rather than replacing the bank. The new stage-inquiry path can reject one step hypothesis without declaring another globally true. Clinical diagnosis and unmeasured probabilities were not copied into semantic selection. |
| 2048 demonstration, `5b8f9a313984873f`, sampled around 10:40: the visible chat tracks two 16s merging into a 32 on the right edge, two 4s into an 8 at top-left, and two 128s into a 256 after a later move. Apollo guidance restoration, `7c67085791f29509`, around 11:29-13:20: local diagnosis precedes replacement. | Keep a tile's value distinct from its changing position; predict the state after an action, then compare it with observation. A local mismatch need not refute every rule. | The 2048 code already has `Arrangement`/`Cell`, a learned transition rule, `CompiledWorld`, and a one-state compiled-versus-rule check. In the semantic path, `RegisterIdentity`, source occurrence spans, trajectories, and stage-aligned contrasts supply the analogous separation. `CompiledWorld` is a line-board optimization and is not a general semantic solver. The 2048 file yielded no ASR segments; this row rests on sampled frames, not a complete dialogue trace. |
| Fermat/Wiles documentary, `c122c40845bc8bc5`, 2:20-2:55 and 6:49-8:28: a bridge between domains was conditional on an unproved conjecture, and a later gap required repair. | Preserve proof dependencies and distinguish a conditional implication from a completed proof. | `semantic_program_floor` verifies executable steps. `MeaningStageInquiry` makes intermediate predictions observable separately; no executable program is treated as proof that the source selected it. |
| FNAF timeline material, `c726cb263bacf2ba`, `1e4bb6bfe0c2c2f0`, `9117887d1ae90d91`, `184599a8a92fcc63`: chronology, aliases, sprite conventions, and explicit revision of an earlier wrong identity claim. | Decompose a theory into atomic dependencies. Strong evidence for chronology must not automatically certify identity; a correction should leave independent subclaims intact. | Source-bound operation occurrences and stage contrasts allow localized disagreement. The current code does not infer fictional canon or learn a reliability value from these edited videos. |
| DDLC/Project Libitina theory, `4e3abb26cdbe551c`, around 3:00-11:30, and Cicada account, `aa1653c27d6e86c7`, 5:25-6:10 and 11:29-11:40: encoded artifacts require an explicit transform chain; a valid signature authenticates a message, not every claim in it. | Keep representation transforms, artifact identity, source authenticity, and semantic truth as separate assertions. | This is a remaining integration item; a transcript or plausible decode cannot be silently promoted to semantic evidence. Existing RLC epistemic state distinguishes evidence provenance and verification. No general artifact-transform service has been connected by this change. |
| Game Theory ARG creator retrospective, `1df521a6b369fa61`, 5:19-7:51: genuine designed typos and Morse blinking coexisted with misleading background detail and accidental smudges. | Search for alternate encodings, but require creator/source-bound corroboration and null controls; novelty alone is not a clue. | The new training variants cross form and graph topology so a form-only rule is less useful. Target-blind held evaluation and nuisance controls remain required. |
| Apollo 11 alarm recording, `bf683bf364c180cd`, around 0:50-1:08: triage under an ongoing operation. CERN Higgs announcement, `a464628574cc0bbf`: a public result followed a separate experimental chain. | Time pressure and announcement confidence cannot replace independent measurement. | Already represented by bounded operations and evidence statuses in `core/brain/llm/latent_cortex/epistemic_state.py`; no direct training label was extracted from either clip. The alarm ASR is noisy. |
| Kevin Buzzard math talk, `5064447119133341`, and tumor-board case, `c6a9e333d61c9ae9`: formal proof and domain evidence have different validators. | Choose a verifier that matches the claim and avoid transferring certainty from a checkable substep to an unchecked conclusion. | Canonical floor proof is scoped to program execution. The independent source-meaning observation remains separate. No medical treatment rule was added. |
| Apollo 13 film power-up scene, `fb529381fd8be9ed`, and brief Ventris clip, `f8b1b82fb535fab3`: edited drama and a biographical introduction are not a full trace of the discovery. | Mark unavailable evidence, rather than reconstructing a process from an outcome. | No new mechanism inferred from these clips. |
| Daniel Everett monolingual fieldwork, `7c1376bb1da77bb1`, 57:33-58:04: change some things and hold others fixed; 43:30-44:12 and 75:06-75:41: correct a lexical hypothesis with a speaker and cross-check other speakers; 59:29-59:36: an apparent left/right failure revealed an up-river/down-river reference frame. | Elicit minimal contrasts, preserve raw observations, and allow the ontology/reference frame itself to be wrong. A speaker's correction is evidence for a local use, not a universal grammar. | `controlled_counterfactual_inputs` and stage inquiries form a reusable minimal-pair test; source spans and immutable proposal receipts keep the original hypothesis available. A new language cannot be learned from these synthetic math sources alone; real cross-speaker observation remains an external input. |
| Boeing 737 accident-investigation documentary, `25e0650754ec3d8d`, 16:18-17:22, 40:26-45:10: nominal valve tests passed, but a later thermal-shock test produced a transient jam and reversal that left no scratch mark. | A negative result is conditional on the test environment. Reproduce the state that could trigger a failure, test the sign of an action-to-outcome mapping, and retain cases without an identified cause. | Fork/join training now crosses wording, graph topology, operation-first/result-first clause order, and witnessed operand-role reversals. The canonical floor verifies what each proposed program would do; it does not establish that a particular source intended that program. This documentary is not a certified source for the engineering history. |
| Lake City Quiet Pills documentary, `64fedcd5ce95650b`, 18:45-21:07 and 46:50-57:51: source code, accounts, shared IPs, writing habits and aliases support several incompatible stories; the narrator later reopens a multi-person account. | Distinguish raw artifacts from interpretations and treat several clues from one upstream actor as dependent. A shared IP or phrase is not an identity proof. | `RelationalGeneralizer` now accepts an explicit upstream `source_group_id` and counts each group once toward principle support. `core/evidence/packet.py` retains exact source identities; it does **not** infer common causes across distinct sources. The video's allegations about real people are not recorded as facts. |
| 4Channer mystery documentary, `5bc0f08843b4cf80`, 9:09-14:55 and 18:33-22:23: coded links connect accounts, videos and a downloadable game, while early claims about real-world harms remain unverified and the narrator later treats the story as a staged ARG. | A verified hyperlink or successful decode establishes a route through artifacts, not the truth of a story told by those artifacts. Source roles and genre may change without changing the bytes. | The portfolio keeps interpretations separate; the shared-source grouping prevents several pages from automatically certifying an upstream story. There is no rule that a decoded reference to a person establishes real-world conduct. The recording's allegations are not used as facts or labels. |
| Kryptos documentary, `ce3857343dcd9fc5`, 8:03-9:15, 17:11-17:17 and 37:07-38:05: an early segmentation by punctuation was abandoned, a useful `Q`-followed-by-`U` guess proved wrong, and an omitted source character still yielded plausible text. The documentary contrasts plaintext recovery with the creator's intended method. | Preserve alternative segmentations and error hypotheses; require source checks before treating a readable output as intended meaning. A correct answer does not uniquely identify the generating path. | `SemanticProgramPortfolio` keeps multiple complete executable proposals; `compare_program_meanings` separates output equivalence from source interpretation. Stage inquiries can target a shared operation despite equal final results, then retain caller-supplied observations. No cryptographic solver or claim about the present unsolved segment was built from the edited video. |
| Brew game/ARG recording, `00f74067188b0eb2`, reviewed transcript and sampled frames: clues from a game and staged story were interpreted as real-world allegations, with consequences outside the game. | Model the boundary between fictional role, creator-controlled artifact, decoded clue, and independent real-world observation. Search can establish an artifact path without establishing an allegation about a person. | The source-bound portfolio and evidence provenance retain candidate interpretations and do not treat a decoded story as observation of conduct. No names or accusations from the video are used as labels. This is a boundary check, not a new fact-finding mechanism. |

The three MatPat sources are not an accuracy oracle: later episodes revise some
earlier proposals, while several hidden-lore conclusions remain speculative.
The distinction between a source-grounded clue, a decoded payload, and a story
about what that payload implies is the useful part.

## Callable changes and focused evidence

- `build_semantic_counterfactual_fork_join_corpus` now crosses three train
  wordings with three train topologies, varies four clause forms and register names,
  and renders witnessed mutations at every step. It excludes all validation
  and test constructions and topologies.
- `native_typed_source_pair_plan` indexes exact shared typed prefixes, then
  finds witnessed partners per decision index. The four-form corpus retains
  144 original sources and adds 2,097 generated fit-only sources. Its
  synthetic byte-token audit found operation and reference partners for all
  2,241 sources, including 1,134 at late join decision 9, in 80.05 seconds
  after 7.02 seconds of corpus construction. Byte tokens affect peer ranking
  and are **not** the model tokenizer or a trained-model result. No
  termination contrast was available in this audit.
- `SemanticProgramPortfolio.plan_stage_inquiries` and
  `reconcile_stage_inquiries` now expose source-aligned intermediate
  disagreements and require caller-supplied observations before eliminating a
  proposal. Pending inquiries and observed results can now be retained and
  reopened through the existing state gateway. A focused test separates two
  programs with the same final output but different first-stage results.
- `RelationalGeneralizer` now has an explicit source-group field for correlated
  observations. In a focused test, three supportive records from two upstream
  groups stay provisional until a third group is observed. The caller must
  identify the grouping; this is not an authorship or causality detector.
- The corpus suite passed 27 focused tests after the four-form change; the
  meaning/inquiry suites passed 37 after durable stage feedback. No full development,
  held-family, target-blind model decode, or G03
  promotion test is claimed here.

## Still to review and connect

The visual-only content of the reviewed files has not been exhaustively
adjudicated. `RelationLanguage` and `IndexProgram` already support bounded
invention and held-out checks for finite index transforms. That vocabulary
cannot be assumed to solve semantic operand binding or general artifact
decoding: those require a source-grounded candidate set and an independent
observation of what the source meant. No such connection is claimed here.

Only after the remaining applicable mechanisms are reviewed and connected
should the frozen G03 progression run: fit-only checks, then unseen-construction
micro probes, controls, full development, fresh transfer, and serving gates.
