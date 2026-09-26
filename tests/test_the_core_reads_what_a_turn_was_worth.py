"""The core reads the payoff: what a turn was worth, what winning earned, and who holds the field.

Each is persistent state that decides what comes next: a payoff is priced
against what its channel has come to pay, a bid is weighed by what its source
has earned, and the field's input is split by what the organs have gained by
connecting. So each is a column of the domain whose organ keeps it: the worth
ledger and the good-news ledger in A, the credit ledger in G, the field's shares
in C. No organ feeds two domains.
"""

from __future__ import annotations

import pytest

from core.affect.what_it_was_worth import CHANNELS, WorthLedger
from core.affect.what_winning_earned import CreditLedger
from core.consciousness.unified_field import UnifiedField
from core.soma.good_news import GoodNewsLedger
from core.state.aura_state import AuraState
from core.subject import state as core_state
from core.subject.state import Organs, read_core_state, schema


def _column(reading, name: str) -> float:
    domain, field = name.split(".", 1)
    return float(reading.values[domain][list(schema(domain).features).index(field)])


def test_the_vocabularies_are_the_organs_own():
    assert core_state._WORTH_CHANNELS == CHANNELS
    assert core_state._FIELD_SOURCES == UnifiedField._INPUT_SOURCES


def test_a_turns_worth_and_what_each_channel_has_come_to_pay_are_in_affect():
    ledger = WorthLedger()
    for _ in range(5):
        ledger.note({"warmth": 0.1})
    ledger.note({"warmth": 0.6})
    reading = read_core_state(AuraState.default(), organs=Organs(worth=ledger))
    assert _column(reading, "A.worth") == pytest.approx(ledger.read().worth)
    assert _column(reading, "A.worth_error_warmth") == pytest.approx(ledger.read().error["warmth"])
    assert _column(reading, "A.worth_expected_warmth") == pytest.approx(ledger._expected["warmth"])
    assert _column(reading, "A.worth") > 0.0


def test_good_news_is_in_affect():
    news = GoodNewsLedger()
    for predicted, actual in ((0.0, 0.1), (0.0, -0.1), (0.0, 0.05), (0.0, 0.4)):
        news.note(predicted, actual)
    reading = read_core_state(AuraState.default(), organs=Organs(good_news=news))
    assert _column(reading, "A.good_news_error") == pytest.approx(0.4)
    assert _column(reading, "A.good_news_jump") == pytest.approx(news.jump())


class _Workspace:
    def get_status(self):
        return {"last_winner": "memory"}


def test_what_winning_earned_the_winner_is_in_attention():
    credit = CreditLedger()
    for turn in range(1, 5):
        credit.note({"memory": turn, "drive": 0}, 1.0)
    credit.note({"memory": 4, "drive": 3}, -1.0)
    reading = read_core_state(AuraState.default(), organs=Organs(workspace=_Workspace(), credit=credit))
    assert _column(reading, "G.winner_earned") == pytest.approx(credit.earned("memory"))
    assert _column(reading, "G.earned_best") == pytest.approx(credit.earned("memory"))
    assert _column(reading, "G.earned_worst") == pytest.approx(credit.earned("drive"))
    assert _column(reading, "G.earned_worst") < 0.0 < _column(reading, "G.earned_best")


class _Field:
    def input_shares(self):
        return {"mesh": 0.4, "chemistry": 0.1, "binding": 0.05, "interoception": 0.05, "substrate": 0.4}


def test_who_holds_the_fields_input_is_in_recurrent_cognition():
    reading = read_core_state(AuraState.default(), organs=Organs(field=_Field()))
    assert _column(reading, "C.field_share_mesh") == pytest.approx(0.4)
    assert _column(reading, "C.field_share_binding") == pytest.approx(0.05)


def test_each_new_organ_feeds_one_domain():
    homes: dict[str, set[str]] = {}
    for domain in core_state.DOMAINS:
        for source in schema(domain).sources:
            if source.startswith("organ:"):
                organ = source.split(":", 1)[1].split(".", 1)[0]
                homes.setdefault(organ, set()).add(domain)
    assert homes["worth"] == {"A"}
    assert homes["good_news"] == {"A"}
    assert homes["credit"] == {"G"}
    assert homes["field"] == {"C"}


def test_the_live_organs_carry_the_three_ledgers():
    organs = Organs.live()
    assert organs.worth is not None
    assert organs.credit is not None
    assert organs.good_news is not None


def test_reading_the_schema_does_not_fix_a_data_path_before_the_run_isolates():
    """The worth ledger's channels are written out in the schema rather than imported.

    Importing the ledger at load pulled in core.self_model and core.utils.paths,
    which fix their data paths at import, and every subject run then refused
    because a module held a path into the shared state root.
    """
    import subprocess
    import sys

    probe = (
        "import sys, core.subject.state; "
        "print(int('core.utils.paths' in sys.modules), int('core.self_model' in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "0 0"
