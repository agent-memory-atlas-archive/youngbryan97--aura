"""How text is embedded when no model is loaded.

Lifted whole out of `vector_memory_engine`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_memory_engine import (
        np,
    )


class _EmbedsWithoutAModel:
    """Lifted whole out of EmbeddingEngine; see vector_memory_engine.py."""

    def _init_tfidf_fallback(self) -> None:
        """Simple TF-IDF fallback when sentence-transformers unavailable."""
        from .vector_memory_engine import (
            logger,
        )

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf_fallback = TfidfVectorizer(max_features=512)
            self._tfidf_corpus = []
            logger.info("EmbeddingEngine: TF-IDF fallback initialized")
        except ImportError:
            logger.error(
                "Neither sentence-transformers nor sklearn available. "
                "Memory recall will be degraded."
            )

    def _embed_tfidf(self, text: str) -> np.ndarray:
        """TF-IDF embedding fallback."""
        from .vector_memory_engine import (
            logger,
            np,
        )

        self._tfidf_corpus.append(text)
        try:
            # fit_transform expects a collection of strings
            matrix = self._tfidf_fallback.fit_transform(self._tfidf_corpus)
            vec = matrix[-1].toarray()[0]
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            logger.debug("TF-IDF embedding failed: %s", e)
            return np.zeros(512)

    def _embed_hash(self, text: str) -> np.ndarray:
        """Minimal hash-based embedding (last resort)."""
        from .vector_memory_engine import (
            embedding_model,
            hashlib,
            np,
        )

        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = vec / 255.0
        # Pad/truncate to standard dim
        target = embedding_model.VECTOR_DIM
        if len(vec) < target:
            vec = np.pad(vec, (0, target - len(vec)))
        return vec[:target]

