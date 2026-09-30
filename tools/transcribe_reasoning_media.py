"""Make source-hashed, timestamped transcripts for reasoning-source review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Systran/faster-whisper-base")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu threads must be positive")
    from faster_whisper import WhisperModel

    args.output.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device="cpu", compute_type="int8",
                         cpu_threads=args.cpu_threads, num_workers=1)
    for ordinal, source in enumerate(args.sources, start=1):
        source = source.resolve(strict=True)
        digest = _sha256(source)
        destination = args.output / f"{ordinal:02d}-{digest[:16]}.json"
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("source_sha256") != digest:
                raise ValueError(f"transcript source changed: {source}")
            print(json.dumps({"source": str(source), "status": "existing",
                              "transcript": str(destination)}), flush=True)
            continue
        segments, info = model.transcribe(str(source), beam_size=5,
                                          vad_filter=True, language="en")
        body = {
            "schema": "aura.g03.video_transcript.v1",
            "source": str(source), "source_sha256": digest,
            "model": args.model, "language": info.language,
            "duration": info.duration,
            "segments": [{"start": item.start, "end": item.end, "text": item.text.strip()}
                         for item in segments],
        }
        destination.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"source": str(source), "status": "completed",
                          "transcript": str(destination),
                          "segments": len(body["segments"])}), flush=True)


if __name__ == "__main__":
    main()
