# G03 v7 partial frozen-prefix reuse, 2026-09-28

The v7 joint graph fit adds 1,952 complete-graph sequences to 21,733 grammar
choice sequences. Its frozen v3 source capture already holds the grammar
states. The three supervision inventories have identical grammar token rows
under SHA-256
`aa3f577257bca16daba91dcc7580dc7f2c6e30a4c0a1190e61f09192da2e83f8`.

The v7-only reuse contract binds that entire row inventory, the frozen model
identity, the trie/precision protocol, all 488 archived source manifests, and
their capture receipts. Training will read each old shard through its byte
digest, capture only graph alternatives, compare their trie and full-prefix
rankings, and write a complete new-plan shard per source. Verification reads
both old and new shard bytes independently. The optimizer and source schedule
remain fresh. This does not grant serving or qualification authority.

Source-only preflight completed without model weights at
`/Users/bryan/.aura/rlc-evidence/semantic-native-joint-graph-v7-partial-final-20260928`.
The plan SHA-256 is
`7fff4f3aa03da3254b3bc5e177714e5acfc0d5a72037f6623c3480babdd51780`;
the supervision receipt SHA-256 is
`12b350d409aa3f7da0508f95335e7d464eb089837e61246c05305c392770c19c`.
It declares 23,685 total sequences, 1,952 requiring new model capture, and
a largest projected float32 shard of 223,457,280 bytes. The 21,733 archived
states are 91.8% of that sequence population. This is a capture-count saving,
not a measured wall-time saving.

The first model-active check can stay small. Freeze the selected checkpoint on
the 303-fit/185-calibration source partition, then generate six source-held
requests with no target available to the scorer. Run matched fitted and base
arms under the same search budget; independently verify exact procedures,
observed answers, baseline regressions, and source erasure or swap controls.
One new exact procedure on an unseen construction with no lost base success
would establish a bounded transfer instance. It would not establish broad
reliability, a 500-request result, or frontier performance.

The partial-reuse path passed 432 native tests with four skips. Lint,
compile, governance-lint and layering passed. Model-active fit, the six-case
check, held-family replication and broader validation have not run for this
plan. G03 remains open.
