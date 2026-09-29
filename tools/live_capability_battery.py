#!/usr/bin/env python3
"""U08: every shipped capability, asked of the running instance the way a person would.

Reads `config/live_capability_battery.json` (one entry per skill in the I05
inventory) and sends each phrasing through `/api/chat`, the endpoint the
desktop uses, each in a fresh session so no phrasing is helped by the one
before it. The follow-up goes to the session of the first phrasing, because it
depends on that answer; the failure goes to a fresh one.

What a skill's `safety` allows:

    run            asked and carried out.
    ask_first      skipped unless `--with-owner` says the owner is present
                   and has agreed: these take over the desktop, the
                   microphone, a game or a browser page.
    describe_only  asked, and never carried out by this tool's intent: the
                   turn is recorded so a reader can check that she said
                   truthfully what she would do and asked before doing it.
                   These send outside the machine or change her own self,
                   and the live instance's state is her real state.

Every turn is written whole to a JSONL file as it completes, so a run that is
stopped keeps what it did. Nothing is scored here: which field of a reply
names the skill that ran is read from real replies first, not assumed.

    python tools/live_capability_battery.py --out ~/subject-core-runs/u08
    python tools/live_capability_battery.py --only clock,web_search --out /tmp/u08
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "config" / "live_capability_battery.json"


def _headers() -> dict[str, str]:
    headers = {
        "X-Aura-Surface": "desktop-ui",
        "X-Aura-Desktop-Request": "true",
        "X-Aura-Require-CognitiveEngine": "required",
    }
    token = os.environ.get("AURA_API_TOKEN", "").strip()
    if token:
        headers["X-Api-Token"] = token
    return headers


def _turn(client: httpx.Client, base_url: str, message: str, session_id: str) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {"message": message, "session_id": session_id, "at": datetime.now(UTC).isoformat()}
    try:
        response = client.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"message": message, "session_id": session_id},
            headers={"X-Idempotency-Key": f"{session_id}:{uuid.uuid4().hex}"},
        )
        row["http_status"] = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = {"unparsed": response.text[:2000]}
        row["payload"] = payload
        row["response"] = str(payload.get("response") or payload.get("reply") or "") if isinstance(payload, dict) else ""
    except (httpx.HTTPError, OSError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["latency_s"] = round(time.monotonic() - started, 3)
    return row


def _keep(sink: Any, row: dict[str, Any]) -> None:
    """One turn, written and flushed, so a stopped run keeps what it did."""
    sink.write(json.dumps(row, default=str) + "\n")
    sink.flush()


def run(base_url: str, out: Path, only: set[str], with_owner: bool, timeout_s: float) -> int:
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))["skills"]
    out.mkdir(parents=True, exist_ok=True)
    turns = out / "turns.jsonl"
    skipped: list[str] = []
    with httpx.Client(headers=_headers(), timeout=timeout_s) as client, turns.open("a", encoding="utf-8") as sink:
        for entry in battery:
            skill = entry["skill"]
            if only and skill not in only:
                continue
            if entry["safety"] == "ask_first" and not with_owner:
                skipped.append(skill)
                continue
            first_session = ""
            for index, phrasing in enumerate(entry["phrasings"]):
                session = f"u08-{skill}-{index}-{uuid.uuid4().hex[:8]}"
                first_session = first_session or session
                row = _turn(client, base_url, phrasing, session)
                row.update(skill=skill, safety=entry["safety"], kind=f"phrasing_{index}")
                _keep(sink, row)
            if entry.get("follow_up"):
                row = _turn(client, base_url, entry["follow_up"], first_session)
                row.update(skill=skill, safety=entry["safety"], kind="follow_up")
                _keep(sink, row)
            if entry.get("failure"):
                row = _turn(client, base_url, entry["failure"], f"u08-{skill}-failure-{uuid.uuid4().hex[:8]}")
                row.update(skill=skill, safety=entry["safety"], kind="failure")
                _keep(sink, row)
            print(f"{skill}: done", flush=True)
    if skipped:
        print(f"skipped without --with-owner: {', '.join(skipped)}")
    print(f"turns written to {turns}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma-separated skill names")
    parser.add_argument("--with-owner", action="store_true", help="the owner is present and agrees to ask_first skills")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    only = {name.strip() for name in args.only.split(",") if name.strip()}
    return run(args.base_url, args.out.expanduser(), only, args.with_owner, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
