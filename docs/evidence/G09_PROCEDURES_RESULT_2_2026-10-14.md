# G09, attempt 2: planning gains, math does not, breadth not met

Plan: G09_PROCEDURES_PREREGISTRATION_2_2026-10-12.md (4cf1055ba, amended
2b61482be and 62927f914). Numbers: G09_PROCEDURES_RESULT_2_2026-10-14_analysis.json,
from tools/g09_analysis.py and tools/g09_verdict.py. G09 stays open.

## Induction

Development run (eight kinds, four rounds, budget forcing): CRUXEval output
prediction was kept in round 2, agreeing with all 400 known answers. Trip
planning reached 153 of the first 156 shown answers and stopped there; no
other development kind wrote a function that agreed with its shown pool.

Held-out run (eight kinds, four rounds), kept:

| Kind | Domain | Round | Known answers agreed |
| --- | --- | ---: | ---: |
| PlanBench Blocksworld | planning | 0 | 250 of 250 |
| PlanBench Mystery Blocksworld | planning | 0 | 250 of 250 |
| DeepMind algebra__linear_1d | math | 0 | 5,000 of 5,000 |
| DeepMind polynomials__evaluate | math | 2 | 5,000 of 5,000 |

Not kept: LiveCodeBench execution (118 of 120 shown answers in round 2; its
functions kept leaving out names the programs under test use, such as
defaultdict, accumulate and prod), Natural Plan meeting,
algebra__linear_2d_composed and comparison__sort_composed. Code therefore
had no held-out kind.

## Drafts and the kept arm

| Kind | Her drafts right | Kept right | Gained | Lost |
| --- | ---: | ---: | ---: | ---: |
| Blocksworld | 149 of 150 | 150 | 1 | 0 |
| Mystery Blocksworld | 16 of 25 | 25 | 9 | 0 |
| algebra__linear_1d | 150 of 150 | 150 | 0 | 0 |
| polynomials__evaluate | 149 of 150 | 150 | 1 | 0 |

No request of the routing-only sets (MATH outside MATH-500, AIME 2024 and
2025, HumanEval+, HotpotQA, BBH, BBEH Mini, CRUXEval, Natural Plan) was
answered by a kept procedure, so none was drafted.

Mystery Blocksworld was stopped after 25 of its 150 drafts, at 10:42 on 14
October, on Bryan's question whether the rest bought anything. It could not
change the verdict: breadth needs two domains, math had one gain and no loss
on its 300 requests, and code had no kept kind, so breadth could fail no
other way. Each Mystery Blocksworld draft took about 9,800 tokens, and the
remaining 125 would have taken about nine and a half hours. The drafts
written are the first 25 of the planned 150 in the plan's order; the run
can be resumed from where it stopped.

## The criteria

| Criterion | Result | Holds |
| --- | --- | --- |
| Primary: pooled gains over losses, alpha 0.05 | 11 to 0, p = 0.00098 | yes |
| Breadth: two of code, planning and math, held-out kinds only | planning 10 to 0 (p = 0.0020); math 1 to 0 (p = 1.0); code none | no |
| No harm: no domain loses more than it gains | no losses anywhere | yes |

G09 closes only if all three hold. It does not close on attempt 2.

## What this shows

Kept procedures beat her own reasoning where her reasoning fails: Mystery
Blocksworld, the obfuscated planning domain, where she was wrong on 9 of 25.
Where her drafts are already near perfect (Blocksworld, one-variable linear
equations, evaluating a polynomial) a kept procedure has almost nothing to
gain, and the breadth criterion cannot be met there. No kept procedure lost
a request.

## What this does not show

* A gain in more than one domain.
* Anything about kinds no procedure was kept for.

## Next

Attempt 3 (G09_PROCEDURES_PREREGISTRATION_3_2026-10-14.md, amended 573e8e9e5,
39b35e323, fced40c42): 52 DeepMind modules, 27 BBH tasks and three further
PlanBench domains (Logistics, Depots, Obfuscated Deceptive Logistics) held
out, breadth two of math, transfer and planning at 0.05 / 3. Its induction
started at 10:44 on 14 October.
