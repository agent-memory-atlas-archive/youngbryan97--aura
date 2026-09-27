"""Lossless source shards for frozen hidden states with bounded retained arrays."""

from __future__ import annotations

import hashlib
import io
import json
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from core.runtime.atomic_writer import atomic_write_bytes_if_absent
from core.runtime.file_read_gateway import open_stable_readonly_binary

if TYPE_CHECKING:
    import mlx.core as mx

#: (source, program sha) for a program fit; (source, decision, choice) for grammar choices.
StateKey = tuple[str, str] | tuple[str, int, int]


class FrozenStateStore(Mapping):
    """Keep all alternatives on disk; cache only whole, verified source shards."""

    def __init__(self, directory: Path, *, plan_sha256: str, max_resident_bytes: int) -> None:
        if (len(plan_sha256) != 64 or any(c not in "0123456789abcdef" for c in plan_sha256)
                or type(max_resident_bytes) is not int or max_resident_bytes <= 0):
            raise ValueError("invalid frozen state storage contract")
        self.directory = Path(directory)
        self.plan_sha256 = plan_sha256
        self.max_resident_bytes = max_resident_bytes
        self._entries = {}
        self._shards = {}
        self._cache = OrderedDict()
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.loads = 0

    @classmethod
    def open_existing(cls, directory: Path, *, plan_sha256: str,
                      max_resident_bytes: int, sequence_digests: Mapping[StateKey, str]) -> Self:
        """Bind complete immutable shards from a prior capture to fresh fitting."""
        from tools.evaluate_semantic_native_checkpoint import verified_document

        store = cls(directory, plan_sha256=plan_sha256,
                    max_resident_bytes=max_resident_bytes)
        keys = tuple(sorted(sequence_digests))
        sources = sorted({key[0] for key in keys})
        if not keys or any(not isinstance(key, tuple) or len(key) != 3
                           or not isinstance(key[0], str) for key in keys):
            raise ValueError("reused frozen state keys differ")
        manifests = {hashlib.sha256(source.encode()).hexdigest() + ".json"
                     for source in sources}
        if {path.name for path in store.directory.glob("*.json")} != manifests:
            raise ValueError("reused frozen state manifest inventory differs")
        for source in sources:
            name = hashlib.sha256(source.encode()).hexdigest()
            receipt = verified_document(store.directory / f"{name}.json")
            rows = receipt.get("arrays")
            if (receipt.get("schema") != "aura.frozen_state_shard.v1"
                    or receipt.get("source") != source
                    or receipt.get("plan_sha256") != plan_sha256
                    or type(receipt.get("array_bytes")) is not int
                    or not 0 < receipt["array_bytes"] <= max_resident_bytes
                    or not isinstance(rows, list) or not rows):
                raise ValueError("reused frozen source manifest differs")
            path = store.directory / f"{name}.safetensors"
            if not path.is_file() or path.stat().st_size != receipt.get("file_bytes"):
                raise ValueError("reused frozen source file differs")
            store._shards[source] = receipt
            for index, row in enumerate(rows):
                key = tuple(row["key"])
                if (key in store._entries or len(key) != 3 or key[0] != source
                        or row.get("tensor") != str(index)
                        or row.get("sequence_sha256") != sequence_digests.get(key)):
                    raise ValueError("reused frozen source alternatives differ")
                store._entries[key] = (source, str(index))
        if set(store._entries) != set(keys):
            raise ValueError("reused frozen source coverage differs")
        return store

    def write_source(self, source: str, states: Mapping[StateKey, mx.array], *,
                     sequence_digests: Mapping[StateKey, str]) -> dict[str, Any]:
        import mlx.core as mx

        keys = tuple(sorted(states))
        if (not keys or source in self._shards or set(sequence_digests) != set(keys)
                or any(not isinstance(key, tuple) or not key or key[0] != source
                       or any(type(part) not in {str, int} for part in key) for key in keys)
                or any(key in self._entries for key in keys)):
            raise ValueError("frozen source shard identities differ or repeat")
        arrays = {str(index): states[key] for index, key in enumerate(keys)}
        mx.eval(arrays)
        size = sum(value.nbytes for value in arrays.values())
        if size > self.max_resident_bytes:
            raise ValueError("one frozen source exceeds the resident shard bound")
        metadata = [{"key": list(key), "tensor": str(index),
                     "shape": list(arrays[str(index)].shape),
                     "dtype": str(arrays[str(index)].dtype),
                     "sequence_sha256": sequence_digests[key]} for index, key in enumerate(keys)]
        if any(len(row["sequence_sha256"]) != 64
               or any(c not in "0123456789abcdef" for c in row["sequence_sha256"])
               for row in metadata):
            raise ValueError("frozen sequence digest invalid")
        stream = io.BytesIO()
        mx.save_safetensors(stream, arrays)
        payload = stream.getvalue()
        name = hashlib.sha256(source.encode()).hexdigest()
        receipt = {"schema": "aura.frozen_state_shard.v1", "source": source,
                   "plan_sha256": self.plan_sha256, "arrays": metadata,
                   "array_bytes": size, "file_bytes": len(payload),
                   "weights_sha256": hashlib.sha256(payload).hexdigest()}
        encoded = json.dumps(receipt, sort_keys=True, allow_nan=False).encode()
        receipt["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
        path = self.directory / f"{name}.safetensors"
        if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
            raise FileExistsError(path)
        manifest = self.directory / f"{name}.json"
        if not atomic_write_bytes_if_absent(manifest, json.dumps(receipt, sort_keys=True).encode(), mode=0o400):
            raise FileExistsError(manifest)
        self._shards[source] = receipt
        for index, key in enumerate(keys):
            self._entries[key] = (source, str(index))
        return receipt

    def __getitem__(self, key: StateKey) -> mx.array:
        import mlx.core as mx

        source, tensor = self._entries[key]
        if source not in self._cache:
            receipt = self._shards[source]
            while self._cache and self.resident_bytes + receipt["array_bytes"] > self.max_resident_bytes:
                old_source, _ = self._cache.popitem(last=False)
                self.resident_bytes -= self._shards[old_source]["array_bytes"]
            name = hashlib.sha256(source.encode()).hexdigest()
            path = self.directory / f"{name}.safetensors"
            with open_stable_readonly_binary(path, max_bytes=receipt["file_bytes"]) as (handle, identity):
                digest = hashlib.sha256()
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                if identity.size != receipt["file_bytes"] or digest.hexdigest() != receipt["weights_sha256"]:
                    raise ValueError("frozen state shard digest differs")
                handle.seek(0)
                arrays = mx.load(handle, format="safetensors")
                mx.eval(arrays)
            expected = {row["tensor"]: row for row in receipt["arrays"]}
            if (set(arrays) != set(expected)
                    or any(list(value.shape) != expected[name]["shape"]
                           or str(value.dtype) != expected[name]["dtype"] for name, value in arrays.items())
                    or sum(value.nbytes for value in arrays.values()) != receipt["array_bytes"]):
                raise ValueError("frozen state shard geometry differs")
            self._cache[source] = arrays
            self.resident_bytes += receipt["array_bytes"]
            self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
            self.loads += 1
        self._cache.move_to_end(source)
        return self._cache[source][tensor]

    def __iter__(self) -> Iterator[StateKey]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def receipt(self) -> dict[str, Any]:
        return {"schema": "aura.frozen_state_store.v1", "plan_sha256": self.plan_sha256,
                "sources": len(self._shards), "sequences": len(self),
                "max_resident_bytes": self.max_resident_bytes,
                "peak_resident_bytes": self.peak_resident_bytes, "loads": self.loads,
                "lossy_compression": False, "shards": list(self._shards.values())}
