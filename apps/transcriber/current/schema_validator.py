#!/usr/bin/env python3
"""
ABRXOS Transcript Schema Validator
Valida transcripciones contra el esquema canónico
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    """Imprime error y termina"""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_transcript(data: dict[str, Any]) -> None:
    """
    Valida una transcripción contra el esquema canónico
    
    Args:
        data: Diccionario con la transcripción
    
    Raises:
        SystemExit: Si la validación falla
    """
    # Campos obligatorios
    required = [
        "schemaVersion",
        "transcriptId",
        "projectId",
        "sourceId",
        "status",
        "source",
        "engine",
        "quality",
        "words",
        "segments",
        "cues",
    ]

    for key in required:
        if key not in data:
            fail(f"Falta el campo obligatorio: {key}")

    # Validar que schemaVersion sea correcto
    if data["schemaVersion"] != "abrxos.transcript.v1":
        fail(f"schemaVersion inválido: {data['schemaVersion']}")

    # Validar words
    words = data["words"]
    if not isinstance(words, list):
        fail("'words' debe ser una lista")

    word_ids = set()
    previous_end = -1.0

    for word in words:
        # Campos obligatorios por palabra
        for field in ["id", "text", "normalized", "start", "end", "segmentId"]:
            if field not in word:
                fail(f"Palabra sin campo '{field}': {word}")

        word_id = word["id"]
        if word_id in word_ids:
            fail(f"ID de palabra duplicado: {word_id}")
        word_ids.add(word_id)

        start = word["start"]
        end = word["end"]

        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            fail(f"Timestamps no numéricos en palabra {word_id}")

        if end <= start:
            fail(f"Palabra {word_id} con rango inválido: start={start}, end={end}")

        if start < 0 or end < 0:
            fail(f"Timestamps negativos en palabra {word_id}")

        if start < previous_end:
            fail(f"Las palabras no están ordenadas temporalmente (palabra {word_id})")

        previous_end = end

    # Validar segments
    segments = data["segments"]
    if not isinstance(segments, list):
        fail("'segments' debe ser una lista")

    segment_ids = set()

    for segment in segments:
        for field in ["id", "start", "end", "text", "wordIds"]:
            if field not in segment:
                fail(f"Segmento sin campo '{field}': {segment}")

        segment_id = segment["id"]
        if segment_id in segment_ids:
            fail(f"ID de segmento duplicado: {segment_id}")
        segment_ids.add(segment_id)

        if segment["end"] <= segment["start"]:
            fail(f"Segmento {segment_id} con rango inválido")

        for word_id in segment.get("wordIds", []):
            if word_id not in word_ids:
                fail(
                    f"Segmento {segment_id} referencia palabra inexistente: {word_id}"
                )

    # Validar cues
    cues = data["cues"]
    if not isinstance(cues, list):
        fail("'cues' debe ser una lista")

    cue_ids = set()

    for cue in cues:
        for field in ["id", "segmentIds", "wordStartId", "wordEndId", "start", "end", "text", "status"]:
            if field not in cue:
                fail(f"Cue sin campo '{field}': {cue}")

        cue_id = cue["id"]
        if cue_id in cue_ids:
            fail(f"ID de cue duplicado: {cue_id}")
        cue_ids.add(cue_id)

        if cue["end"] <= cue["start"]:
            fail(f"Cue {cue_id} con rango inválido")

        word_start_id = cue["wordStartId"]
        word_end_id = cue["wordEndId"]

        if word_start_id not in word_ids:
            fail(f"Cue {cue_id} con wordStartId inexistente: {word_start_id}")

        if word_end_id not in word_ids:
            fail(f"Cue {cue_id} con wordEndId inexistente: {word_end_id}")

        for seg_id in cue.get("segmentIds", []):
            if seg_id not in segment_ids:
                fail(f"Cue {cue_id} referencia segmento inexistente: {seg_id}")

    # Validar source
    source = data["source"]
    for field in ["filename", "path", "sha256", "language"]:
        if field not in source:
            fail(f"'source' sin campo '{field}'")

    # Validar engine
    engine = data["engine"]
    for field in ["provider", "model", "createdAt"]:
        if field not in engine:
            fail(f"'engine' sin campo '{field}'")

    # Validar quality
    quality = data["quality"]
    for field in ["wordTimingAvailable", "sourceAudioVerified", "needsReview"]:
        if field not in quality:
            fail(f"'quality' sin campo '{field}'")

    print("✓ Transcripción válida")
    print(f"  Palabras: {len(words)}")
    print(f"  Segmentos: {len(segments)}")
    print(f"  Cues: {len(cues)}")
    print(f"  Estado: {data['status']}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Uso: python schema_validator.py transcript.json")
        return 2

    path = Path(sys.argv[1])

    if not path.exists():
        fail(f"No existe: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"JSON inválido: {exc}")

    validate_transcript(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
