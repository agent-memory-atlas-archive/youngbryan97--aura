"""The clock the mesh stamps spikes with, which a harness can set.

Lifted whole out of `neural_mesh`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
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


class _SpikeClock:
    """Lifted whole out of NeuralMesh; see neural_mesh.py."""

    def use_clock(self, clock: Any) -> None:
        """Stamp spikes with `clock` from now on, keeping how long ago each one fired.

        The subject-core harness steps the mesh once a frame on a clock of its
        own that advances by a fixed step and rewinds on a restore. The mesh
        read the machine's monotonic clock, so two arms from one snapshot saw
        each spike as older or newer by however long each arm happened to take,
        and spike-timing plasticity moved their weights apart: by about 1e-8 in
        the mesh on seed 7, which the unified field reading it grew to 8e-5 in a
        turn (28 September).
        """
        with self._lock:
            offset = float(clock()) - float(self._clock())
            for column in self.columns:
                fired = column.last_spike_time >= 0.0
                column.last_spike_time[fired] += offset
            self._clock = clock

