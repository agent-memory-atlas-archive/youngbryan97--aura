"""SQLAlchemy's ORM is read in a thread, not by whoever first asks for the audit log.

`persistent_state` is a lazily-built singleton, so the first caller pays for
importing `core.db.orm`, and that pulls in SQLAlchemy's ORM: hundreds of modules
to find, read and compile. Four of the 120 newest stall dumps caught the event
loop inside `sqlalchemy.orm`'s own module bodies, reached through
`create_persistent_state`.

An import is idempotent and `sys.modules` is shared, so reading it in a thread at
registration leaves the factory alone and lets its first call find the work done.
The import stays inside the factory on purpose: hoisted to module scope, a test
patching `core.db.orm.PersistentState` would stop being seen.
"""

from __future__ import annotations

import inspect
import sys

import pytest

from core.providers import memory_provider

pytestmark = pytest.mark.unit


def test_registration_warms_it():
    body = inspect.getsource(memory_provider.register_memory_services)
    assert "_warm_the_orm_off_the_loop()" in body


def test_the_warming_happens_in_a_thread():
    body = inspect.getsource(memory_provider._warm_the_orm_off_the_loop)
    assert "threading.Thread" in body
    assert "daemon=True" in body
    assert "import core.db.orm" in body


def test_the_factory_keeps_its_own_import():
    """Hoisted to module scope it would stop a patch of PersistentState being seen."""
    body = inspect.getsource(memory_provider.register_memory_services)
    assert "from core.db.orm import PersistentState" in body
    module = inspect.getsource(memory_provider)
    head = module[: module.index("def register_memory_services")]
    assert "from core.db.orm import" not in head


def test_a_missing_sqlalchemy_does_not_raise(monkeypatch):
    def refuse(name, *args, **kwargs):
        if name.startswith("core.db.orm"):
            raise ImportError("no sqlalchemy")
        return original(name, *args, **kwargs)

    original = __import__
    monkeypatch.setattr("builtins.__import__", refuse)
    # The warming thread swallows it; the factory's own path answers it.
    memory_provider._warm_the_orm_off_the_loop()


def test_it_leaves_the_module_importable_afterwards():
    memory_provider._warm_the_orm_off_the_loop()
    # Whether SQLAlchemy is present or not, nothing is left half-imported.
    assert "core.db.orm" not in sys.modules or sys.modules["core.db.orm"] is not None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
