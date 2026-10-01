"""Recall from the gateway's records serves what the disk holds, not what a timer last saw.

The index over the gateway's JSON records refreshed on a background thread
every fifteen seconds. A record written since the last pass could not be
recalled, one deleted since could, and quarantined records never left. In the
subject harness that put the time an arm happened to start into what the next
arm recalled.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

import core.memory.gateway_record_index as gri
from core.memory.gateway_record_index import GatewayRecordIndex


def _record(root: Path, name: str, content: str, *, family: str = "episodic") -> Path:
    directory = root / family
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps({"payload": {"content": content, "written_at": time.time(), "metadata": {}}}),
        encoding="utf-8",
    )
    return path


def _found(index: GatewayRecordIndex, query: str) -> set[str]:
    return {entry.memory_id for _score, entry in index.search(query, limit=10)}


def test_a_resync_reads_the_disk_before_it_returns(tmp_path: Path) -> None:
    _record(tmp_path, "kept", "the harbour light at dusk")
    index = GatewayRecordIndex(tmp_path)
    index.resync()
    assert _found(index, "harbour light") == {"kept"}

    gone = _record(tmp_path, "gone", "the harbour light in fog")
    index.resync()
    assert _found(index, "harbour light") == {"kept", "gone"}
    gone.unlink()
    index.resync()
    assert _found(index, "harbour light") == {"kept"}


def test_a_noted_write_and_removal_are_seen_by_the_next_search(tmp_path: Path) -> None:
    index = GatewayRecordIndex(tmp_path)
    index.resync()
    path = _record(tmp_path, "fresh", "a lighthouse keeper's log")
    index.note_change(path)
    assert _found(index, "lighthouse keeper") == {"fresh"}
    path.unlink()
    index.note_change(path)
    assert _found(index, "lighthouse keeper") == set()


def test_a_quarantined_record_is_not_recalled(tmp_path: Path) -> None:
    _record(tmp_path, "episodic_bad", "a record she was told was false", family="_quarantine")
    _record(tmp_path, "good", "a record she was told was false and kept")
    index = GatewayRecordIndex(tmp_path)
    index.resync()
    assert _found(index, "record told false") == {"good"}


def test_a_write_noted_while_a_pass_lists_the_disk_survives_the_pass(tmp_path: Path, monkeypatch) -> None:
    index = GatewayRecordIndex(tmp_path)
    index.resync()
    listed_before = index._list_record_files

    def listing_then_a_write() -> list[tuple[float, Path]]:
        files = listed_before()
        index.note_change(_record(tmp_path, "between", "written while the pass was listing"))
        return files

    monkeypatch.setattr(index, "_list_record_files", listing_then_a_write)
    index.resync()
    assert _found(index, "written while listing") == {"between"}


@pytest.mark.asyncio
async def test_the_gateway_tells_the_index_what_it_wrote(tmp_path: Path, monkeypatch) -> None:
    from core.memory import memory_write_gateway as mwg
    from core.runtime.gateways import MemoryWriteRequest

    async def approve(**_kwargs):
        return {"approved": True, "receipt_id": "gov-test"}

    gateway = mwg.ConcreteMemoryWriteGateway(root=tmp_path, governance_decide=approve)
    monkeypatch.setattr(gri, "_INDEX", None)
    index = gri.get_gateway_record_index(tmp_path)
    index.resync()

    receipt = await gateway.write(
        MemoryWriteRequest(
            content="the tide table for the north quay",
            metadata={"family": "episodic"},
            cause="test",
        )
    )
    assert _found(index, "tide table north quay") == {receipt.record_id}

    await gateway.quarantine(receipt.record_id, "a test")
    await asyncio.sleep(0)
    assert _found(index, "tide table north quay") == set()
