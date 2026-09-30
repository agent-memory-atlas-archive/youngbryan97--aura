"""Recording a latent episode refused because its visible answer fell short of the product contract.

Lifted whole out of `latent_cortex_service`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _refuse_an_answer_short_of_the_contract(
    *,
    reason: Any,
) -> None:
    """Record a latent episode refused because its visible answer fell short of the contract.

    Moved out of ``deep_reason`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 1 name(s) from the turn and hands back
    0.
    """
    from .latent_cortex_service import (
        record_degradation,
    )

    record_degradation(
        "latent_cortex.output_quality",
        RuntimeError(reason),
        action=(
            "refused a mechanically complete latent episode whose visible answer did not satisfy the product contract"
        ),
        severity="degraded",
    )


