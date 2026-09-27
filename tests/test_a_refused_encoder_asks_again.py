"""A refused encoder asks for the lane again once the lane has changed.

LIVE 2026-09-26, boot at 18:57: a standalone probe held the model lane
exclusively when her embedding encoder asked for it. The encoder was refused,
settled on its word-count fallback, and never asked again. The probe finished
before her own model loaded, but semantic routing stayed "unavailable", the
chat warmup failed all three attempts, and chat was held closed until the
process was stopped twenty minutes later.
"""

from __future__ import annotations

import threading

from core.cognition import evidence_relevance
from core.memory import vector_memory_engine as engine_module
from core.memory.vector_memory_engine import EmbeddingEngine
from core.runtime import model_lane_control
from core.runtime.model_lane_control import ModelLaneControlError


class _Model:
    def to(self, _device):
        return self


class _Lease:
    def __init__(self):
        self.released = []

    def set_preemptible(self, _value):
        return True

    def release(self, *, reason=""):
        self.released.append(reason)


def _a_lane(monkeypatch, *, refuse: list[bool]):
    """A lane that refuses while ``refuse[0]`` holds, and a stamp to move by hand."""
    stamp = {"now": 1}
    asked: list[str] = []

    def acquire(**kwargs):
        asked.append(kwargs["owner_id"])
        if refuse[0]:
            raise ModelLaneControlError("sync_in_process_model_admission_refused:exclusive_lane_owned:standalone:1:probe")
        return _Lease()

    monkeypatch.setattr(model_lane_control, "acquire_synchronous_in_process_model_lane", acquire)
    monkeypatch.setattr(engine_module, "_lane_stamp", lambda: stamp["now"])
    monkeypatch.setattr(engine_module.embedding_model, "load_encoder", lambda **_kw: _Model())
    return stamp, asked


def test_a_refusal_stands_while_the_lane_has_not_changed(monkeypatch):
    stamp, asked = _a_lane(monkeypatch, refuse=[True])
    engine = EmbeddingEngine()
    engine._initialize()
    assert engine._model is None and engine._initialized
    assert engine._checkout_model() is None
    assert engine.warm_in_background() is None
    assert len(asked) == 1, "an unchanged lane gives the same answer, so it is not asked twice"


def test_a_lane_that_changed_is_asked_again_and_the_encoder_loads(monkeypatch):
    refuse = [True]
    stamp, asked = _a_lane(monkeypatch, refuse=refuse)
    engine = EmbeddingEngine()
    engine._initialize()
    assert engine._model is None

    refuse[0] = False
    stamp["now"] = 2
    assert engine._checkout_model() is None, "asked again on the loader thread, not by the caller"
    engine._loader.join(5.0)
    assert len(asked) == 2
    model = engine._checkout_model()
    assert isinstance(model, _Model)
    engine._return_model()


def test_a_second_refusal_waits_for_the_next_change(monkeypatch):
    stamp, asked = _a_lane(monkeypatch, refuse=[True])
    engine = EmbeddingEngine()
    engine._initialize()
    stamp["now"] = 2
    loader = engine.warm_in_background()
    assert loader is not None
    loader.join(5.0)
    assert len(asked) == 2 and engine._model is None
    assert engine.warm_in_background() is None
    assert len(asked) == 2


def test_the_prewarm_waits_out_a_load_in_flight(monkeypatch):
    """A prewarm pays for the load; "nothing yet" is not "unavailable"."""
    release = threading.Event()

    class Embedder:
        model = None

        def warm_in_background(self):
            def load():
                release.wait(5.0)
                Embedder.model = _Model()

            loader = threading.Thread(target=load, daemon=True)
            loader.start()
            threading.Timer(0.2, release.set).start()
            return loader

        def _checkout_model(self):
            return Embedder.model

        def _return_model(self):
            pass

    monkeypatch.setattr(evidence_relevance, "_embedder", lambda: Embedder())
    evidence_relevance._wait_for_the_encoder()
    assert evidence_relevance.semantic_routing_available()


def test_a_probe_that_got_nothing_gives_nothing_back(monkeypatch):
    returned: list[int] = []

    class Embedder:
        def _checkout_model(self):
            return None

        def _return_model(self):
            returned.append(1)

    monkeypatch.setattr(evidence_relevance, "_embedder", lambda: Embedder())
    assert not evidence_relevance.semantic_routing_available()
    assert not returned, "returning a model never taken lowers the count of an encode running elsewhere"
