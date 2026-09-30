"""Whether an anchor saved by one process forks the same future in another.

Two processes on one seed come up as two organisms (170 to 194 of 438 columns
apart at the first frame), so shards that collect their own anchors sweep
different organisms. If a fork snapshot survives the trip to disk, the shards
can fork from the coordinator's anchors instead. This asks that directly.

    python tools/probe_anchor_travel.py write OUT   # boot, collect anchors, save them and one turn from each
    python tools/probe_anchor_travel.py read OUT    # boot differently, load them, run the same turn, compare

The reading process runs a longer baseline than the writer on purpose, so it is
a different organism when it restores; only what the snapshot carries can make
the two turns agree. Exit 0 when every vector matches to the bit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("AURA_TESTING", "1")

ANCHORS = 4


async def _organism(run_dir: Path, seed: int, rounds: int):
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.isolation import isolate_state
    from tools.run_subject_core_v25 import _seed_every_generator

    await asyncio.to_thread(run_dir.mkdir, parents=True, exist_ok=True)
    isolate_state(run_dir)
    # As the rig brings a shard up under ONE_CLOCK: the global generators from
    # the seed before the organism exists, so what it draws at boot and never
    # changes again (the mesh's weights) is the same in both processes.
    _seed_every_generator(seed)
    runtime = build_runtime(run_dir, seed=seed)
    await start_organism(runtime)
    await quiesce_organism(runtime)
    await calibrate_clock(runtime, CONDITIONS)
    for _ in range(rounds):
        for condition in CONDITIONS:
            await runtime.turn_once(condition)
    return runtime, CONDITIONS


async def _one_turn_from_each(runtime, anchors, condition) -> tuple[np.ndarray, np.ndarray]:
    """Her state as restored, before anything runs, and after one turn from it.

    The first says whether the snapshot carried her; the second whether the
    turn then ran the same. A difference only in the second is the process,
    not the snapshot.
    """
    restored, turned = [], []
    facts: list[list[str]] = []
    for anchor in anchors:
        runtime.restore(anchor.snapshot)
        restored.append(np.asarray(runtime.read(condition.name, "restored", {}).vector(), dtype=np.float64))
        frames = await runtime.turn_once(condition)
        turned.append(np.asarray(frames[-1].vector(), dtype=np.float64))
        world = getattr(getattr(runtime, "state", None), "world", None)
        held = getattr(world, "facts", None) or {}
        entries = list(held.items()) if isinstance(held, dict) else list(held)
        facts.append([repr(entry)[:400] for entry in entries])
    _last_facts.append(facts)
    return np.vstack(restored), np.vstack(turned)


#: What each anchor's turn left in her world facts, for the comparison to name.
_last_facts: list[list[list[str]]] = []


def _compare(what: str, written: np.ndarray, read: np.ndarray) -> bool:
    from core.subject.recording import feature_names

    names = list(feature_names())
    apart = written != read
    print(f"{what}: columns apart per anchor {apart.sum(axis=1).tolist()}; "
          f"largest gap {float(np.max(np.abs(written - read))):.6g}")
    gap = np.abs(written - read).max(axis=0)
    for j in np.argsort(-gap)[:12]:
        if gap[j] > 0:
            print(f"  {names[j] if j < len(names) else j}: {gap[j]:.4g}")
    return not apart.any()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / f"logs-{args.mode}"))
    bank = args.out / "anchors.pickle"
    if args.mode == "write":
        from core.subject.v25_runtime import collect_anchor_bank

        runtime, conditions = await _organism(args.out / "writer", args.seed, rounds=2)
        anchors = list(await collect_anchor_bank(runtime, conditions, rounds=1, history_turns=2, every=1))[:ANCHORS]
        from core.subject.anchor_travel import dumps

        bank.write_bytes(dumps(anchors))
        print(f"saved {len(anchors)} anchors, {bank.stat().st_size} bytes")
        restored, vectors = await _one_turn_from_each(runtime, anchors, conditions[0])
        np.save(args.out / "writer_restored.npy", restored)
        np.save(args.out / "writer_turns.npy", vectors)
        (args.out / "writer_facts.json").write_text(json.dumps(_last_facts[-1], indent=1))
        return 0
    runtime, conditions = await _organism(args.out / "reader", args.seed, rounds=3)
    from dataclasses import replace

    from core.subject.anchor_travel import fill_snapshot, loads

    own = runtime.snapshot()
    holes: list[str] = []
    anchors = [replace(a, snapshot=fill_snapshot(a.snapshot, own, holes=holes)) for a in loads(bank.read_bytes())]
    if holes:
        print(f"{len(holes)} holes, {len(set(h.split(': ')[-1] for h in holes))} kinds:")
        for kind in sorted(set(h.split(": ")[-1] for h in holes)):
            first = next(h for h in holes if h.endswith(": " + kind))
            print(f"  {kind} x{sum(h.endswith(': ' + kind) for h in holes)}, first at {first.rsplit(': ', 1)[0]}")
        return 2
    restored, vectors = await _one_turn_from_each(runtime, anchors, conditions[0])
    np.save(args.out / "reader_restored.npy", restored)
    np.save(args.out / "reader_turns.npy", vectors)
    (args.out / "reader_facts.json").write_text(json.dumps(_last_facts[-1], indent=1))
    print(f"anchors {len(anchors)}")
    same_restored = _compare("as restored", np.load(args.out / "writer_restored.npy"), restored)
    same_turned = _compare("after one turn", np.load(args.out / "writer_turns.npy"), vectors)
    return 0 if same_restored and same_turned else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
