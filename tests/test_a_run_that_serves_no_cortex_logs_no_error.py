"""A pointer nobody pinned is not an invalid pointer.

Every arm of the battery campaign of 29 September logged
`Active cortex pointer is invalid: migration_authority_key_unavailable` at
ERROR, twice, because a battery serves no cortex and so pins no migration
authority key. An ERROR for a thing nobody asked for teaches its reader to skip
errors, and the neural stream is read by hand.

A key somebody DID pin and that cannot be read stays an error: the registry
cannot then confirm the cortex it has loaded, and her affective steering never
attaches.
"""

from __future__ import annotations

import logging

import pytest

import core.brain.llm.model_registry as registry
from core.brain.llm.model_registry import _pointer_failure_is_worth_reporting
from core.learning.cortex_migration_authority import CortexMigrationAuthorityError

pytestmark = pytest.mark.unit

UNAVAILABLE = CortexMigrationAuthorityError("migration_authority_key_unavailable")


def test_nothing_pinned_a_key_so_nothing_is_an_error(monkeypatch):
    monkeypatch.delenv("AURA_CORTEX_AUTHORITY_KEY_FILE", raising=False)
    assert _pointer_failure_is_worth_reporting(UNAVAILABLE) is False


def test_a_key_somebody_pinned_and_could_not_read_is_still_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_CORTEX_AUTHORITY_KEY_FILE", str(tmp_path / "pinned.key"))
    assert _pointer_failure_is_worth_reporting(UNAVAILABLE) is True


@pytest.mark.parametrize(
    "exc",
    [
        CortexMigrationAuthorityError("migration_authority_key_custody_invalid"),
        ValueError("active_cortex_path_missing"),
        ValueError("active_cortex_pointer_not_object"),
        OSError("permission denied"),
    ],
)
def test_every_other_way_the_pointer_can_be_invalid_is_an_error(exc, monkeypatch):
    monkeypatch.delenv("AURA_CORTEX_AUTHORITY_KEY_FILE", raising=False)
    assert _pointer_failure_is_worth_reporting(exc) is True


def test_the_reader_takes_the_quiet_path_when_it_says_so(tmp_path, monkeypatch, caplog):
    manifest = tmp_path / "active.json"
    manifest.write_text('{"active_model_path": ""}')
    monkeypatch.setattr(registry, "_pointer_failure_is_worth_reporting", lambda _exc: False)
    with caplog.at_level(logging.INFO, logger=registry.logger.name):
        assert registry._read_active_cortex_spec(manifest) is None
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("nothing pinned" in r.getMessage() for r in caplog.records)


def test_the_reader_still_reports_when_it_says_so(tmp_path, monkeypatch, caplog):
    manifest = tmp_path / "active.json"
    manifest.write_text('{"active_model_path": ""}')
    monkeypatch.setattr(registry, "_pointer_failure_is_worth_reporting", lambda _exc: True)
    with caplog.at_level(logging.INFO, logger=registry.logger.name):
        assert registry._read_active_cortex_spec(manifest) is None
    assert [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_no_pointer_at_all_is_silent(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=registry.logger.name):
        assert registry._read_active_cortex_spec(tmp_path / "absent.json") is None
    assert not caplog.records


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
