#!/usr/bin/env python3
"""
ABRXOS Transcription Engine
Motor principal de transcripción: coordina adapter, alignment y validación
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "abrxos.transcript.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except Exception:
        return None


def run(
    media_path: Path,
    model: str,
    language: str | None,
    project_id: str,
    source_id: str,
    keep_raw: bool = False,
    raw_output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Ejecuta la transcripción completa.

    Args:
        media_path: Ruta al archivo de audio/vídeo
        model: Ruta o identificador del modelo
        language: Código de idioma
        project_id: ID del proyecto
        source_id: ID de la fuente
        keep_raw: Si guardar el resultado crudo de Whisper
        raw_output_path: Dónde guardar el resultado crudo

    Returns:
        Transcripción canónica como dict
    """
    from mlx_whisper_adapter import transcribe_with_mlx, validate_model
    from alignment_engine import build_words, build_segments, build_cues

    # Validar modelo
    model_check = validate_model(model)
    if not model_check["valid"]:
        raise RuntimeError(f"Modelo no válido: {model_check['error']}")

    print(f"→ Transcribiendo: {media_path.name}")
    print(f"→ Modelo: {model}")
    if language:
        print(f"→ Idioma: {language}")

    # Transcribir
    raw_result = transcribe_with_mlx(
        audio_path=str(media_path),
        model=model,
        language=language,
        word_timestamps=True,
    )

    # Guardar resultado crudo si se pide
    if keep_raw and raw_output_path:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(
            json.dumps(raw_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    raw_segments = raw_result.get("segments", [])

    # Construir estructura canónica
    words = build_words(raw_segments)
    segments = build_segments(raw_segments, words)
    cues = build_cues(segments)

    duration = probe_duration(media_path)
    sha256 = sha256_file(media_path)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "transcriptId": f"{project_id}_{source_id}_{media_path.stem}_V1",
        "projectId": project_id,
        "sourceId": source_id,
        "status": "word_aligned" if words else "needs_review",
        "source": {
            "filename": media_path.name,
            "path": str(media_path.resolve()),
            "sha256": sha256,
            "durationSeconds": duration,
            "language": language,
        },
        "engine": {
            "provider": "mlx-whisper",
            "model": model,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "wordTimestampsRequested": True,
        },
        "cuePolicy": {
            "mode": "segment_based",
            "minDurationSeconds": 0.2,
            "maxDurationSeconds": None,
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
