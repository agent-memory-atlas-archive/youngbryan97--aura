"""A group's telemetry declared twice still names its channels the second time.

Each `declare()` declares once per process and returned `[]` on every later
call. `declare_telemetry` registers the phenomena publisher with whatever
`declare()` returns, so a second call re-registered it with no channels over
the registration that had them. Found as an order dependence on 29 Sep:
`test_declaring_the_dispositions_registers_their_publisher` passed alone and
failed after `test_cognition_discipline`, whose boot had declared first.
Conation's `boot()` registers its publisher the same way, and every group's flag
outlived a reset of the dictionary it described.
"""
from __future__ import annotations

import importlib

import pytest

GROUPS = (
    "core.conation.telemetry",
    "core.morphogenesis.telemetry",
    "core.ontogeny.telemetry",
    "core.fsw.phenomena_channels",
)


@pytest.mark.parametrize("module_name", GROUPS)
def test_a_second_call_names_what_the_first_declared(module_name: str) -> None:
    module = importlib.import_module(module_name)
    first = module.declare()
    second = module.declare()
    assert first, f"{module_name} declared nothing"
    assert second == first, f"{module_name} answered a second call with {second!r}"


def test_the_phenomena_publisher_keeps_its_channels_when_declared_again() -> None:
    from core.fsw.telemetry_samplers import samplers_report
    from core.phenomena_wiring import declare_telemetry

    declare_telemetry()
    declare_telemetry()
    entry = next(row for row in samplers_report()["samplers"] if row["name"] == "phenomena")
    assert entry["channels"], "the second registration dropped the channels"


@pytest.mark.parametrize("module_name", GROUPS)
def test_a_group_declares_again_into_a_dictionary_that_forgot(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag set before the dictionary was reset does not stand for channels that are gone.

    The two flakes this caught, 29 Sep: the morphogenesis tick and the bound
    check read `None` from channels their group had declared before an
    earlier test reset the dictionary, because the group's flag said it was
    done and it never declared again.
    """
    from core.fsw import telemetry_dictionary as td

    module = importlib.import_module(module_name)
    names = module.declare()
    monkeypatch.setattr(td, "_DICTIONARY", td.TelemetryDictionary())
    assert not td.still_declared(names)
    assert module.declare() == names
    assert td.still_declared(names)
