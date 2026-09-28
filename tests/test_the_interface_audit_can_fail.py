"""The interface audit fails a page built to fail it, and passes one built to pass.

`make ui-audit` reports green on every page Aura ships. A gate that has never
been seen to fail cannot tell a clean interface from a check that stopped
checking, and this one has been wrong that way twice (see U09 in the master
todo: one version measured zero of 291 nodes). So each kind of finding gets a
page that must produce it, in the same browser the gate uses.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

playwright = pytest.importorskip("playwright.sync_api")


def _runner():
    spec = importlib.util.spec_from_file_location("audit_ui_pages", ROOT / "tools" / "audit_ui_pages.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("audit_ui_pages", module)
    spec.loader.exec_module(module)
    return module


def _chromium_or_skip() -> None:
    with playwright.sync_playwright() as pw:
        try:
            pw.chromium.launch().close()
        except Exception as exc:  # the browser is an optional download
            pytest.skip(f"no headless Chromium: {exc}")


BAD = """<!doctype html><html><head><meta charset="utf-8"><title>bad</title>
<style>
  body { background: #0d0c0f; color: #f4ede4; margin: 0; }
  .faint { color: #1c1a20; }
  .tiny { width: 12px; height: 12px; padding: 0; }
  input { outline: none; border: 1px solid #555; background: #222; color: #eee; }
  .drift { width: 40px; height: 40px; background: #333; animation: drift 3s linear infinite; }
  @keyframes drift { to { transform: translateX(80px); } }
  .wide { width: 900px; }
</style></head>
<body>
  <p class="faint">This sentence is nearly the colour of the page.</p>
  <button class="tiny" aria-label="close"></button>
  <button></button>
  <input type="text" aria-label="message">
  <div class="drift"></div>
  <div class="wide">wide</div>
  <canvas id="field" width="60" height="60"></canvas>
  <script>
    const field = document.getElementById('field').getContext('2d');
    let x = 0;
    (function spin() { field.clearRect(0, 0, 60, 60); field.fillStyle = '#b1a4ff';
      field.fillRect(x = (x + 1) % 50, 20, 10, 10); requestAnimationFrame(spin); })();
  </script>
</body></html>
"""

GOOD = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>good</title>
<style>
  body { background: #0d0c0f; color: #f4ede4; margin: 0; padding: 16px; }
  button { min-width: 44px; min-height: 44px; color: #f4ede4; background: #2a2833; border: 0; }
  :focus-visible { outline: 2px solid #b1a4ff; outline-offset: 2px; }
  .glow { width: 40px; height: 40px; background: #333; animation: glow 2s ease-in-out infinite; }
  @keyframes glow { 50% { opacity: 0.5; } }
</style></head>
<body>
  <h1>A page with nothing wrong</h1>
  <p>Readable text on its own ground.</p>
  <button>Send</button>
  <div class="glow"></div>
  <canvas id="still" width="60" height="60"></canvas>
  <script>
    const still = document.getElementById('still').getContext('2d');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let x = 0;
    (function spin() { still.clearRect(0, 0, 60, 60); still.fillStyle = '#b1a4ff';
      still.fillRect(x = (x + 1) % 50, 20, 10, 10);
      if (!reduced.matches) requestAnimationFrame(spin); })();
  </script>
</body></html>
"""


@pytest.fixture(scope="module")
def findings(tmp_path_factory) -> dict[str, list[str]]:
    _chromium_or_skip()
    root = tmp_path_factory.mktemp("interface")
    static = root / "static"
    static.mkdir()
    shutil.copy(ROOT / "interface" / "static" / "a11y_audit.js", static / "a11y_audit.js")
    (static / "bad.html").write_text(BAD, encoding="utf-8")
    (static / "good.html").write_text(GOOD, encoding="utf-8")
    problems = _runner().audit(root=root)
    return {
        "bad": [p for p in problems if p.startswith("bad ")],
        "good": [p for p in problems if p.startswith("good ")],
    }


@pytest.mark.parametrize(
    "kind",
    [
        "contrast / faint",
        "unnamed control",
        "small target",
        "no first-level heading",
        "page is ",
        "no visible focus / input",
        "moves under reduced motion / drift",
        "moves under reduced motion / canvas field redrawn",
    ],
)
def test_a_page_built_to_fail_fails(findings, kind):
    assert any(kind in line for line in findings["bad"]), findings["bad"]


def test_a_page_built_to_pass_passes(findings):
    assert findings["good"] == []
