# G03 real-tokenizer pair audit, 2026-09-29

The revised postfix corpus was rebuilt with seed 41, then bounded to 648
examples by `select_bounded_semantic_examples`. The SHA-256 of the selected
example-ID list is
`25bd4fba71af08801d6ab1c214ad614cea39485195f28b887613408a83d08130`.
The active model descriptor resolved to
`Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`; its descriptor SHA-256
is `52d313c2c435d343cf6acfa2b2ca61bc70334c01b8b17877df79f32ec3c5283c`.
The tokenizer-file identity from `tokenizer_checkpoint_identity` is
`60d689a68d0e32d27fab74726171d7ab84e49bd2005146f83ecaed3b3f5a3585`.

The model's offset tokenizer, without loading weights, tokenized all 648
sources with at most 125 tokens each. On those token IDs, the fit-only typed
pair planner found witnessed operation and reference peers for all 648
sources. Decision 9 had 388 paired sources. No termination pair was found.
This replaces the uncertainty about byte-token ranking in the earlier dry
run; it does not supply hidden states, training updates, held-family results,
or public-answer accuracy. Those remain required before G03 promotion.
