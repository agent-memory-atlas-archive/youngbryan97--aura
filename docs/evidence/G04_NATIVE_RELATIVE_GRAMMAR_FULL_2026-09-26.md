# Target-blind three-step development result

The role-relative step-128 scorer constructed 72/72 typed programs equivalent
to their targets and produced 72/72 exact executed answers across 24
three-step constructions. The grammar supplied legal operations, register
types, and structural JSON only; the scorer received no target program,
answer, construction label, or candidate bank. There were zero forced
completions and zero depth-bound failures. The model-active run took 4,221.95
seconds.

An independent CPU process reloaded the selected checkpoint receipt,
reconstructed all 72 target requests, verified every row and report digest,
replayed every legal decision from the saved score vectors, re-executed each
program, and checked current implementation hashes. It also rebuilt the
1,764 examples in all eight bound source manifests: no source text,
construction ID, or topology ID overlaps these 24 target constructions.

- Run: `~/.aura/rlc-evidence/semantic-native-relative-grammar-full-20260926/`.
- Plan: `075c3cfb30cf01b01af78600de0a4105e8fed2872d7f658f72ec5b03625acdae`.
- Report: `aa72006af984e1a1ad044cc12521731b81c5b1d2338f885690a726d235ee6f72`.
- Independent receipt: `independent-verification.json` in the run directory.
- Source report SHA256: `30c955d0655bb087c5f8081e4b13fb9935d2078e334e5fa64447a470119d18e9`.

The run is **development evidence**, not a confirmatory transfer result. The
72 requests use three fixed lexical templates, share a left-to-right chain
shape, and were exposed in earlier development runs. Input arity, types,
and values were supplied by the evaluator, not extracted from free language.
No same-grammar unfitted or ordinary-model control has yet been run. G04,
G05, G06, G07, G09, and G12 remain open; this result grants no serving or
fusion authority. The next comparison must hold the grammar, requests,
budget, and checkpoint identity fixed while removing the fitted weights,
then move to newly committed source forms and actual public input grounding.
