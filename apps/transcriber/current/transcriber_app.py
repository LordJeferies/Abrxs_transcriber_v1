#!/usr/bin/env python3
"""
ABRXOS Transcriber App
Punto de entrada principal para transcripción palabra por palabra
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "abrxos.transcript.v1"


def now_iso() -> str:
    """Retorna timestamp ISO actual en UTC"""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 de un archivo"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration(path: Path) -> float | None:
    """Obtiene duración del archivo media con ffprobe"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    """Normaliza texto eliminando espacios múltiples"""
    return " ".join(str(value or "").split())


def word_confidence(word: dict[str, Any]) -> float | None:
    """Extrae confidence de una palabra"""
    value = word.get("confidence")
    if value is None:
        value = word.get("probability")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def transcribe(
    media_path: Path,
    model: str,
    language: str | None,
    project_id: str,
    source_id: str,
) -> dict[str, Any]:
    """
    Transcribe un archivo usando mlx-whisper
    
    Args:
        media_path: Ruta al archivo de audio/vídeo
        model: Ruta o identificador del modelo
        language: Código de idioma (ej: 'es')
        project_id: ID del proyecto
        source_id: ID de la fuente
    
    Returns:
        Diccionario con la transcripción canónica
    """
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "No se pudo importar mlx_whisper. "
            "Comprueba el entorno Python utilizado."
        ) from exc

    options: dict[str, Any] = {
        "path_or_hf_repo": model,
        "word_timestamps": True,
        "verbose": False,
    }

    if language:
        options["language"] = language

    print(f"Transcribiendo: {media_path}")
    print(f"Modelo: {model}")
    print(f"Idioma: {language}")

    result = mlx_whisper.transcribe(str(media_path), **options)

    raw_segments = result.get("segments", [])
    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    cues: list[dict[str, Any]] = []

    next_word_id = 0

    for segment_index, raw_segment in enumerate(raw_segments):
        segment_id = f"SEG_{segment_index + 1:05d}"
        segment_words: list[dict[str, Any]] = []

        for raw_word in raw_segment.get("words", []) or []:
            start = raw_word.get("start")
            end = raw_word.get("end")

            if start is None or end is None:
                continue

            start = float(start)
            end = float(end)

            if end <= start:
                continue

            text = normalize_text(raw_word.get("word") or raw_word.get("text"))

            item = {
                "id": next_word_id,
                "text": text,
                "normalized": text.lower(),
                "start": start,
                "end": end,
                "confidence": word_confidence(raw_word),
                "segmentId": segment_id,
            }

            words.append(item)
            segment_words.append(item)
            next_word_id += 1

        segment_start = raw_segment.get("start")
        segment_end = raw_segment.get("end")

        if segment_start is None and segment_words:
            segment_start = segment_words[0]["start"]

        if segment_end is None and segment_words:
            segment_end = segment_words[-1]["end"]

        if segment_start is None or segment_end is None:
            continue

        segment = {
            "id": segment_id,
            "start": float(segment_start),
            "end": float(segment_end),
            "text": normalize_text(raw_segment.get("text")),
            "wordIds": [word["id"] for word in segment_words],
            "wordStartId": segment_words[0]["id"] if segment_words else None,
            "wordEndId": segment_words[-1]["id"] if segment_words else None,
        }

        segments.append(segment)

        if segment_words:
            cues.append({
                "id": segment_index,
                "segmentIds": [segment_id],
                "wordStartId": segment_words[0]["id"],
                "wordEndId": segment_words[-1]["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "status": "generated",
            })

    duration = probe_duration(media_path)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "transcriptId": f"{project_id}_{source_id}_{media_path.stem}_V1",
        "projectId": project_id,
        "sourceId": source_id,
        "status": "word_aligned" if words else "needs_review",
        "source": {
            "filename": media_path.name,
            "path": str(media_path.resolve()),
            "sha256": sha256_file(media_path),
            "durationSeconds": duration,
            "language": language,
        },
        "engine": {
            "provider": "mlx-whisper",
            "model": model,
            "createdAt": now_iso(),
            "wordTimestampsRequested": True,
        },
        "quality": {
            "wordTimingAvailable": bool(words),
            "sourceAudioVerified": False,
            "needsReview": True,
        },
        "words": words,
        "segments": segments,
        "cues": cues,
    }


def write_text_export(transcript: dict[str, Any], output: Path) -> None:
    """Exporta transcripción a formato TXT legible"""
    lines: list[str] = []

    lines.append(f"TRANSCRIPT_ID: {transcript['transcriptId']}")
    lines.append(f"PROJECT_ID: {transcript['projectId']}")
    lines.append(f"SOURCE_ID: {transcript['sourceId']}")
    lines.append(f"STATUS: {transcript['status']}")
    lines.append("")

    for cue in transcript.get("cues", []):
        lines.append(
            f"[{cue['start']:.3f} - {cue['end']:.3f}] "
            f"CUE_ID: {cue['id']}"
        )
        lines.append(
            f"WORD_START_ID: {cue['wordStartId']} "
            f"WORD_END_ID: {cue['wordEndId']}"
        )
        lines.append(cue.get("text", ""))
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ABRXOS Transcript Builder usando mlx-whisper"
    )

    parser.add_argument("media", type=Path, help="Archivo de audio/vídeo")
    parser.add_argument("--model", required=True, help="Ruta o ID del modelo")
    parser.add_argument("--language", default="es", help="Código de idioma")
    parser.add_argument("--project-id", required=True, help="ID del proyecto")
    parser.add_argument("--source-id", default="DEFAULT", help="ID de la fuente")
    parser.add_argument("--output", type=Path, required=True, help="Ruta de salida JSON")
    parser.add_argument("--text-output", type=Path, help="Ruta de salida TXT opcional")

    args = parser.parse_args()

    if not args.media.exists():
        print(f"ERROR: No existe el archivo: {args.media}", file=sys.stderr)
        return 2

    try:
        transcript = transcribe(
            media_path=args.media,
            model=args.model,
            language=args.language,
            project_id=args.project_id,
            source_id=args.source_id,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.text_output:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        write_text_export(transcript, args.text_output)

    print(f"✓ JSON creado: {args.output}")
    if args.text_output:
        print(f"✓ TXT creado: {args.text_output}")

    print(f"\nPalabras: {len(transcript['words'])}")
    print(f"Segmentos: {len(transcript['segments'])}")
    print(f"Cues: {len(transcript['cues'])}")
    print(f"Estado: {transcript['status']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
