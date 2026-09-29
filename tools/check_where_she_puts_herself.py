#!/usr/bin/env python3
"""Where her record puts her on the questions a page asks, without the cortex.

The placement is a measurement over her own record, so it can be checked
without any of the machinery that answers in words. Point this at a page and it
reads the questions the way the observer does, measures each one, and prints
where she lands and what in her put her there.

General: it knows nothing about any instrument. A page with a run of controls
between two phrases is read as a dimension; a page with labelled answers is
read as a statement with ways of answering it; anything else is skipped and
said so.

    tools/check_where_she_puts_herself.py https://example.org/a/questionnaire
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from html.parser import HTMLParser
from typing import Any


class _Questions(HTMLParser):
    """The radio groups on a page, with the words around each group.

    A cheap stand-in for the live observer's accessibility pass: enough to read
    the shape of a questionnaire from static HTML, so the measurement can be
    checked without a browser.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, Any]] = []
        self._text: list[str] = []
        self._row: list[dict[str, str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        got = {key: (value or "") for key, value in attrs}
        if tag == "tr":
            self._row, self._text = [], []
        elif tag == "input" and got.get("type") == "radio" and self._row is not None:
            self._row.append(
                {
                    "group": got.get("name", ""),
                    "value": got.get("value", ""),
                    "label": "",
                }
            )
            self._text.append("[]")

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row:
            self.rows.append(
                {"options": self._row, "asks": " ".join(" ".join(self._text).split())}
            )
            self._row, self._text = None, []

    def handle_data(self, data: str) -> None:
        if self._row is not None:
            said = " ".join(data.split())
            if said:
                self._text.append(said)


def _sides(asks: str) -> tuple[str, str] | None:
    runs = list(re.finditer(r"(?:\s*\[\])+", asks))
    if len(runs) != 1:
        return None
    left = " ".join(asks[: runs[0].start()].split())
    right = " ".join(asks[runs[0].end() :].split())
    return (left, right) if left and right else None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many")
    args = parser.parse_args(argv)

    request = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(request, timeout=30).read().decode("utf-8", "replace")
    found = _Questions()
    found.feed(html)
    questions = [row for row in found.rows if len(row["options"]) >= 2]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("no question groups on this page")
        return 1

    from core.self.where_i_stand import (
        Lean,
        against_the_rest,
        her_record,
        where_she_stands,
        which_is_most_her,
    )

    record = her_record()
    print(f"her record: {len(record)} things she has valued, chosen or said")
    print(f"questions:  {len(questions)}\n")
    # Every dimension measured first, then placed against the strongest of
    # them: a page is its own unit for how far she goes.
    readings: list[tuple[dict[str, Any], tuple[str, str] | None, Any]] = []
    for row in questions:
        sides = _sides(row["asks"])
        lean = (
            where_she_stands(sides[0], sides[1], record) if sides is not None else None
        )
        readings.append((row, sides, lean))
    scaled = against_the_rest([lean for _r, _s, lean in readings if lean is not None])
    scaled_by_row: dict[int, float] = {}
    place = 0
    for index, (_row, _read, lean) in enumerate(readings):
        if lean is not None:
            scaled_by_row[index] = scaled[place]
            place += 1

    placed = 0
    spread: dict[int, int] = {}
    for where, (row, sides, lean) in enumerate(readings):
        options = row["options"]
        if sides is not None and lean is not None:
            first, second = sides
            index = (
                Lean(
                    toward=scaled_by_row[where],
                    first=lean.first,
                    second=lean.second,
                    measured=lean.measured,
                ).position_in(len(options))
                if lean.measured
                else None
            )
            because = lean.because
            shape = f'"{first}" / "{second}"'
        else:
            statement = " ".join(re.sub(r"(?:\s*\[\])+", " ", row["asks"]).split())
            named = [f"{statement} {one.get('label') or one.get('value')}" for one in options]
            chosen = which_is_most_her(named, record)
            index = chosen.index if chosen.measured else None
            because = chosen.because
            shape = statement[:60]
        if index is None:
            print(f"  --  {shape}: her record could not answer")
            continue
        placed += 1
        spread[index + 1] = spread.get(index + 1, 0) + 1
        print(f"  {index + 1} of {len(options)}  {shape}")
        for said in because[:2]:
            print(f"        {said}")
    print(f"\nplaced {placed} of {len(questions)}")
    print("spread:", ", ".join(f"{place}:{count}" for place, count in sorted(spread.items())))
    return 0 if placed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
