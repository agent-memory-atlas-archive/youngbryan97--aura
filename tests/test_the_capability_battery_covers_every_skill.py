"""U08's battery asks about every shipped skill, and never runs an ask-first one unattended."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def _battery() -> list[dict]:
    return json.loads((ROOT / "config" / "live_capability_battery.json").read_text("utf-8"))["skills"]


def test_every_skill_in_the_inventory_has_one_entry_and_nothing_else_does():
    shipped = {
        entry["name"]
        for entry in json.loads((ROOT / "artifacts" / "shipped_inventory.json").read_text("utf-8"))["entries"]
        if entry.get("kind") == "skill"
    }
    named = [entry["skill"] for entry in _battery()]
    assert len(named) == len(set(named))
    assert set(named) == shipped


def test_every_entry_is_asked_three_ways_under_a_known_policy():
    for entry in _battery():
        assert entry["safety"] in {"run", "ask_first", "describe_only"}, entry["skill"]
        assert len(entry["phrasings"]) == 3 and all(p.strip() for p in entry["phrasings"]), entry["skill"]


def test_an_ask_first_skill_is_never_sent_without_the_owner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_aura_u08", ROOT / "tools" / "live_capability_battery.py")
    battery = importlib.util.module_from_spec(spec)
    sys.modules["_aura_u08"] = battery
    spec.loader.exec_module(battery)

    sent: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["message"])
        return httpx.Response(200, json={"response": "ok", "status": "ok"})

    real_client = httpx.Client
    monkeypatch.setattr(battery.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(answer), **kw))

    ask_first = next(e for e in _battery() if e["safety"] == "ask_first")
    run_one = next(e for e in _battery() if e["safety"] == "run")
    battery.run("http://test", tmp_path, {ask_first["skill"], run_one["skill"]}, False, 5.0)

    assert not set(ask_first["phrasings"]) & set(sent)
    assert set(run_one["phrasings"]) <= set(sent)
    rows = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text("utf-8").splitlines()]
    assert {row["skill"] for row in rows} == {run_one["skill"]}
