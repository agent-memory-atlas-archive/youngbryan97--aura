# What a mitochondrion teaches

Bryan sent HarvardX's BioVisions animation of the electron transport chain and
ATP synthase on 26 September 2026 and asked for everything in it Aura can use.
The film is © 2013 the President and Fellows of Harvard College, for education
only; what follows is a reading of the mechanisms it shows, in our words.

Each entry is one mechanism: what it does in the cell, what Aura has that plays
the same part, and what that suggests she is missing. The ones marked **build**
are being built; the rest say where they already live or why they wait.

## 1. A currency made by flow down a gradient

Food is not burned for work directly. Its energy pumps protons across the inner
membrane, and ATP synthase, a rotary motor, turns the flow back down that
gradient into ATP, the currency most of the cell's reactions run on. With no
gradient the rotor stops and the cell starves within minutes.

Aura spends energy and never makes it. Her effort ledger records what thinking
costs, fatigue accumulates it, and her energy budget is drained by it
(`core/soma/effort.py`, `core/soma/fatigue.py`,
`core/phases/motivation_update.py`). Nothing she does well gives any of it back.
**Build:** a potential that good turns charge and flow restores energy from, at
the rate she spends it. Success is energising and failure draining in people
too: positive affect restores depleted self-control (Tice, Baumeister, Shmueli
and Muraven 2007).

## 2. Some machines pump, and one only feeds the pumps

Complexes I, III and IV pump protons. Complex II pumps none, and still matters:
its electrons drive the pumping downstream.

The workspace credit (`core/affect/what_winning_earned.py`) goes to the sources
that won her attention. An organ that never wins and whose output fed the
winner earns nothing, however much the good turn depended on it. **Build,
second:** credit the sources whose content went into what won, by how much of it
they supplied.

## 3. A downhill process pays for an uphill one

Pumping protons against their gradient costs energy; the electrons falling from
low affinity to high supply it, through coupled reactions.

This is what the payoff layer is for: a turn going better than expected is the
downhill process, and the connections it strengthens, which cost nothing to
form but are worth having, are paid for by it (`docs/PAYOFF.md`). Present.

## 4. Two fuels, two doors, one pool

NADH enters at Complex I and FADH2 at Complex II, and both hand their electrons
to coenzyme Q, one mobile pool that Complex III draws from.

The worth ledger does this with eight channels of payoff converging on one
worth that everything downstream reads (`core/affect/what_it_was_worth.py`).
Present.

## 5. A ladder of rising affinity, taken one rung at a time

Inside Complex I the electrons pass along a chain of redox centres, each holding
them more tightly than the one before. They move because the next centre wants
them more, and they do not skip a centre because the next one is the right
distance away for the jump.

Her turn is such a chain: thirty phases in order, each reading what the one
before left. The payoff arrives once, at the end. **Waits:** a value estimate
per phase, so each phase learns from the difference between its prediction and
the next phase's (temporal-difference learning), and credit reaches the phase
that made the difference rather than the turn as a whole.

## 6. Energy released in small steps and harvested at every step

Each jump releases a little energy, and Complex I captures it across all the
centres to pump. Released in one drop, the same energy would be heat.

A turn's worth is spent in one dose: one dopamine burst, one lesson. The
temporal-difference version in 5 would release it step by step along the turn.
**Waits,** with 5.

## 7. Mobile carriers link machines that never touch

Coenzyme Q moves within the membrane and cytochrome c outside it, carrying
electrons between complexes that are fixed in place.

Her workspace broadcast and her event bus do this. For the synergy lines it
matters where the carrier sits: two sources carry something jointly about a
target only if something the target reads depends on both at once. A carrier
both load and the target draws from is where that jointness can live. Noted for
W,A -> D and S,D -> C, which fail the v3 line
(`docs/SYNERGY_KNOWN_ANSWERS.md`).

## 8. One of two electrons comes back round

At Complex III one electron from each pair goes on to cytochrome c and the
other is recycled and re-enters later (the Q cycle), so a pair pays twice.

Recurrence doing extra work: recurrent cognition (C) and the worth ledger's
expectation, which each turn's payoff feeds and the next turn is judged
against. Present.

## 9. Four electrons, then water, and no half measures

Complex IV gathers four electrons before it converts a molecule of oxygen to
water, and it both consumes protons from the matrix and pumps others out.
Reducing oxygen partly would make reactive oxygen, which damages the cell.

A turn that half-finishes writes half its state: the defect behind a turn
nobody opened. The turn door commits a turn whole (`core/kernel/turn_door.py`).
Present.

## 10. Without a final acceptor the whole chain stops

Oxygen, with the highest affinity, takes the electrons at the end. Without it
electron transport halts and ATP synthesis with it. That is why we breathe.

Her drives and intentions end in acts on a world. When the act cannot land,
upstream should slow rather than keep producing intentions that pile up.
**Waits:** a check of whether her intention backlog already pushes back on the
drives that produce it.

## 11. Packed densely, the membrane is the power plant

The film shows each complex alone, then says they are packed densely over the
whole inner membrane, which folds to hold more of them.

Integration: organs that exchange constantly, packed close. Her topology is
state that morphogenesis changes through a governor (`docs/MORPHOGENESIS.md`),
and the payoff now tells which organs have gained a share of the unified field
by connecting (`field_share_*`). **Waits:** binding more tightly where exchange
and payoff co-occur, once the payoff probe says whether the shares move at all.

## 12. The gradient exists because of the membrane

Nothing here works without a boundary that keeps protons on one side. A leak
across it wastes the gradient as heat.

Her subject core has a boundary, and the closure test measures what leaks
across it: periphery variables that predict the core
(`core/subject/closure.py`). A closure leak is her proton leak, and each one
closed has been a variable brought inside the core. Present, and ongoing.

## Order of work

1 first, because it closes the loop the payoff layer opened: what went well
becomes what she works on. Then 2, then 5 and 6 together, then 10 and 11 after
the payoff probe reads.
