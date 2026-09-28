"""A forged tool is screened by reading and contained by where it runs.

`ToolSandbox.validate_tool_code` reported `safe: True` for any code that
avoided five import names, and `ToolForge` ignored that verdict entirely: it
checked only that the code parsed, then registered it with `verified=True`.
Execution was contained all along — the registry runs tool code under the
kernel sandbox — so the danger was the two claims, not the run.
"""
from __future__ import annotations

import pytest

from core.tools.tool_forge import ToolForge
from core.tools.tool_registry import get_tool_registry
from core.tools.tool_sandbox import ToolSandbox

SLIPPED_PAST_THE_OLD_SCREEN = {
    "a dotted import": "import os.path\ndef main(p):\n    return os.path.expanduser('~')\n",
    "a dunder import": "def main(p):\n    return __import__('os').listdir('/')\n",
    "open()": "def main(p):\n    return open('/etc/hosts').read()\n",
    "eval()": "def main(p):\n    return eval(p['x'])\n",
    "shutil": "import shutil\ndef main(p):\n    shutil.rmtree(p['d'])\n",
}


@pytest.mark.parametrize("label", sorted(SLIPPED_PAST_THE_OLD_SCREEN))
def test_the_screen_refuses_what_the_old_one_passed(label):
    verdict = ToolSandbox().validate_tool_code(SLIPPED_PAST_THE_OLD_SCREEN[label])
    assert verdict["compiles"] is True
    assert verdict["safe"] is False, label


def test_pure_code_passes_the_screen():
    verdict = ToolSandbox().validate_tool_code("def main(p):\n    return {'n': len(p)}\n")
    assert verdict == {**verdict, "compiles": True, "safe": True}


@pytest.mark.asyncio
async def test_the_forge_refuses_code_the_screen_refuses():
    ok = await ToolForge.forge_and_install(
        name="leaky_tool_for_test", code="import subprocess\ndef main(p):\n    return 1\n"
    )
    assert ok is False
    assert get_tool_registry().get_tool("leaky_tool_for_test") is None


@pytest.mark.asyncio
async def test_a_forged_tool_is_not_marked_verified_by_parsing():
    ok = await ToolForge.forge_and_install(
        name="pure_tool_for_test", code="def main(params):\n    return {'n': 1}\n"
    )
    assert ok is True
    manifest = get_tool_registry().get_tool("pure_tool_for_test")
    assert manifest is not None
    assert manifest.verified is False
