# Unfitted native grammar control

The same first 12 source requests used in the fitted 72-row development run
were decoded by the unfitted resident 27B under the typed grammar. The base
arm generated four exact programs and answers. The fitted arm's saved rows
were exact on all 12. There were eight fitted-only wins and no base-only wins
in this prefix. Four scalar requests passed in both arms; the base model got
none of the four lookup or four count requests right. Eight base rows reached
the depth bound, seven without a connected program and one with a wrong
completed program. The base run took 2,111.71 seconds.

The base plan is `76a97f4d4e55c854a3c1d9cc371bf62a2926468c05b4a9523603ed2a67709c37`;
the report receipt is
`765997003d6982d507aa50938a881bbd798c6d031c082ded824372f18c05c947`.
Independent replay at
`~/.aura/rlc-evidence/semantic-native-base-grammar-canary-20260926/independent-verification.json`
regraded all 12 programs and answers, replayed every grammar decision, verified
the checkpoint/model/source receipts and eight source manifests, and found no
implementation drift. The 12 sources belong to six constructions; none of
their source text, construction identities, or topologies overlaps the 1,764
bound source examples.

The fitted rows came from the earlier v1 evaluator; the base arm used v2,
whose new weight-mode switch skips LoRA conversion and loading. Their sources,
grammar, bound, model, and selected checkpoint identity agree, but this is not
a same-version confirmatory replication. The base model has not learned the
native output wire either. Its failures therefore cannot isolate relational
understanding from format adaptation. These requests were exposed in prior
development, and the first 12 are not independent family samples. This
control does not close G04 or G06 and grants no serving or fusion authority.
