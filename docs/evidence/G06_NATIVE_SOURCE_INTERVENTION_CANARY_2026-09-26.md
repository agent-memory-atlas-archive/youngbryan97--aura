# Native source-operation canary

The fitted 27B produced exact three-step programs and answers for all six
requests in the frozen canary. Each of the three original/changed pairs
changed only its final requested operation. Both members of every pair were
correct, and each decoded program preserved its first two instructions and
final operands while changing the final operation. No row hit a depth bound.
The run took 674.33 seconds.

- Plan: `37aaf01233f70bfd9cd965555dcee9c2b461ca29f742b9707ef85799ace779d5`.
- Report: `8f8832cfef879b3e5c1c1c30ed013ba7c41eac04709d8bae211b12f40e34aaa7`.
- Artifacts: `~/.aura/rlc-evidence/semantic-native-operation-intervention-v3-canary-20260926/`.
- Independent replay: `independent-verification.json` in that directory.

The independent verifier reconstructed the requests, recovered their four
public values from source text, replayed every scored grammar decision, and
executed each resulting program. It rebuilt 1,764 bound source examples in
105 constructions; source-text, construction, and topology overlap were all
zero. No bound implementation file changed during the run.

These six requests use one fixed sentence form per scalar, lookup, and count
topology. The canary establishes that this fitted decoder responded causally
to the changed final-operation wording on these cases. It does not establish
transfer to other sentence forms, a gain over a wire-trained source-blind
control, broad reasoning improvement, serving authority, or fusion readiness.
G04 and G06 remain open.
