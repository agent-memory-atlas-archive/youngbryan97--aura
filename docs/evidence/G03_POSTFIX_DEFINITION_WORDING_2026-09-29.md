# G03 postfix definition wording, 2026-09-29

Inspection of the generated fork/join training requests found a clause such
as "Compute: A minus B; name that value C" before a later clause defined A.
The program and spans were executable, but that sentence implied an immediate
computation with a missing operand. The postfix style now says "Let A minus B
be named C." It states the relation even when clauses are presented out of
execution order. Operation and operand annotations still identify the same
program, and focused tests check both the text and the spans.

This changes source bytes and example identities. No previously acquired
feature bundle is relabeled or reused for the revised corpus. With seed 41,
the corpus still has 2,241 sources and 648 factorial cells. A CPU-only audit
of the 648-source balanced selection using diagnostic byte tokens found 648
operation peers, 648 reference peers, and 388 decision-9 reference peers. The
earlier 390 count belongs to the earlier wording and remains in its dated
record. Neither count measures model-tokenizer pairing or learned transfer.
