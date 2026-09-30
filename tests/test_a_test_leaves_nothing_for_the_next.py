"""What one test measures of her is not the next test's starting point.

Q09 run, 29 Sep: two failures that passed alone. An earlier test's failed
actions lowered her capacity, so a goal was set aside one retry early; an
earlier test's broken promise stood on her interior ledger, so a neutral
sentence read -0.105 against a line it clears alone. The conftest now empties
every private slot a module declares `None` once the test that filled it
ends (d0e188d39).

Each pair below checks that the other half's mark is gone, then leaves its
own. pytest-randomly runs them in either order, and either way the second
one to run fails if the slot survived.
"""
from __future__ import annotations

import importlib


def _ledger_probe(mine: str, theirs: str) -> None:
    from core.agency.authorship import get_agency_ledger

    ledger = get_agency_ledger()
    assert theirs not in ledger.by_capability, "the agency ledger kept another test's record"
    ledger.by_capability[mine] = [3, 0]


def test_the_agency_ledger_starts_clean_a() -> None:
    _ledger_probe("leak_probe_a", "leak_probe_b")


def test_the_agency_ledger_starts_clean_b() -> None:
    _ledger_probe("leak_probe_b", "leak_probe_a")


_INTERIOR = (
    ("core.interiority.service", "get_interiority"),
    ("core.interiority.cleft", "get_cleft"),
    ("core.interiority.receptors", "get_receptor_bank"),
    ("core.interiority.attribution", "get_attribution"),
    ("core.interiority.census", "get_census"),
    ("core.interiority.other_minds", "get_other_minds_model"),
    ("core.interiority.interoception", "get_interoception"),
)


def _interior_probe(mine: str, theirs: str) -> None:
    for module_name, getter in _INTERIOR:
        part = getattr(importlib.import_module(module_name), getter)()
        assert getattr(part, "_leak_probe", None) != theirs, (
            f"{module_name}.{getter}() handed over another test's instance"
        )
        part._leak_probe = mine


def test_her_interior_starts_clean_a() -> None:
    _interior_probe("a", "b")


def test_her_interior_starts_clean_b() -> None:
    _interior_probe("b", "a")
