# Native dependency source-control result

The FP32 native grammar scorer used exact frozen-source trie reuse on six
target-blind, three-step dependency-intervention requests (three paired
source changes, seed 1618033). Each pair keeps the primitive sequence and
public values fixed while changing one type-correct argument dependency.
The fitted arm completed 6/6 exact programs and answers with 3/3 responsive
pairs. The matched unfitted arm completed 2/6 with 1/3 exact pairs. Its four
failures first diverged from the fitted arm at operation choice (`at` or
`count_of` versus `add`), so this comparison alone does not identify a
learned dependency-selection gain.

Erasing only source-content tokens changed all six fitted programs: 0/6
remained exact and 0/3 pairs remained responsive. Independent meaning
audits found a counterexample to each erased program's target meaning.
Substituting the partner's source instead produced 6/6 programs exactly
matching that partner's intact-source program, with 3/3 exact swap pairs.
The existing matched comparator reconstructed the pairs, verified every
arm under the same checkpoint and implementation, and graded swap outputs
against partner targets after decode. This establishes source-conditioned
graph selection on a constructed development cohort, not fresh-family
transfer or broad reasoning gain.

Evidence under `~/.aura/rlc-evidence/`:

- `semantic-native-dependency-trie-fitted-six-v1-20260928/`
- `semantic-native-dependency-trie-base-six-v1-20260928/`
- `semantic-native-dependency-trie-erasure-six-v1-20260928/`
- `semantic-native-dependency-trie-swap-six-v1-20260928/`
- `semantic-native-dependency-trie-comparison-v1-20260928.json`

All four arms have independent replay receipts with no current
implementation drift and counterfactual meaning audits. The comparison
receipt is `d92d4b82efd3911f3725cf0b237d60b8115c75ddd1bc7fba0ac5c44ed6021562`.
The sources reuse three exposed topologies and one definition-style
renderer. G03, G04, G06, and later gates remain open. The next diagnostic
must hold prior operation and argument choices fixed without using target
labels while comparing the fitted and base reference scores directly.
