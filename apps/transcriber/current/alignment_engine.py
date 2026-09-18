#!/usr/bin/env python3
"""
ABRXOS Alignment Engine
Normalización y construcción de segmentos y cues a partir del output de Whisper
"""
from __future__ import annotations

import re
from typing import Any


def normalize_text(value: Any) -> str:
    """Normaliza texto eliminando espacios múltiples y strips"""
    return " ".join(str(value or "").split()).strip()


def normalize_word(text: str) -> str:
    """Normaliza una palabra: minúsculas, sin puntuación inicial/final"""
    text = text.lower().strip()
    text = re.sub(r"^[^\w]+", "", text)
    text = re.sub(r"[^\w]+$", "", text)
    return text


def build_words(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Construye la lista canónica de palabras desde los segmentos de Whisper.

    Filtra palabras con timestamps inválidos.
    Asigna IDs secuenciales estables.

    Args:
        raw_segments: Lista de segmentos del output de Whisper

    Returns:
        Lista de palabras canónicas
    """
    words: list[dict[str, Any]] = []
    next_id = 0

    for segment_index, segment in enumerate(raw_segments):
        segment_id = f"SEG_{segment_index + 1:05d}"

        for raw_word in segment.get("words", []) or []:
            start = raw_word.get("start")
            end = raw_word.get("end")

            if start is None or end is None:
                continue

            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                continue

            if end <= start or start < 0:
                continue

            raw_text = raw_word.get("word") or raw_word.get("text") or ""
            text = normalize_text(raw_text)

            if not text:
                continue

            confidence = _extract_confidence(raw_word)
            speaker = raw_word.get("speaker") or None

            words.append({
                "id": next_id,
                "text": text,
                "normalized": normalize_word(text),
                "start": round(start, 6),
                "end": round(end, 6),
                "confidence": confidence,
                "speaker": speaker,
                "segmentId": segment_id,
            })

            next_id += 1

    return words


def build_segments(
    raw_segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Construye la lista canónica de segmentos.

    Args:
        raw_segments: Segmentos crudos de Whisper
        words: Lista de palabras ya construida

    Returns:
        Lista de segmentos canónicos
    """
    segments: list[dict[str, Any]] = []

    for segment_index, raw_segment in enumerate(raw_segments):
        segment_id = f"SEG_{segment_index + 1:05d}"

        # Palabras de este segmento
        segment_words = [w for w in words if w["segmentId"] == segment_id]

        segment_start = raw_segment.get("start")
        segment_end = raw_segment.get("end")

        if segment_start is None and segment_words:
            segment_start = segment_words[0]["start"]
        if segment_end is None and segment_words:
            segment_end = segment_words[-1]["end"]

        if segment_start is None or segment_end is None:
            continue

        try:
            segment_start = float(segment_start)
            segment_end = float(segment_end)
        except (TypeError, ValueError):
            continue

        if segment_end <= segment_start:
            continue

        segment = {
            "id": segment_id,
            "start": round(segment_start, 6),
            "end": round(segment_end, 6),
            "text": normalize_text(raw_segment.get("text")),
            "wordIds": [w["id"] for w in segment_words],
            "wordStartId": segment_words[0]["id"] if segment_words else None,
            "wordEndId": segment_words[-1]["id"] if segment_words else None,
        }

        segments.append(segment)

    return segments


def build_cues(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Construye cues a partir de los segmentos.

    En esta versión: un cue por segmento (cuePolicy: segment_based).

    Args:
        segments: Lista de segmentos canónicos

    Returns:
        Lista de cues canónicos
    """
    cues: list[dict[str, Any]] = []

    for index, segment in enumerate(segments):
        if segment["wordStartId"] is None or segment["wordEndId"] is None:
            continue

        cues.append({
            "id": index,
            "segmentIds": [segment["id"]],
            "wordStartId": segment["wordStartId"],
            "wordEndId": segment["wordEndId"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "status": "generated",
        })

    return cues


def _extract_confidence(word: dict[str, Any]) -> float | None:
    """Extrae confidence de un resultado de Whisper"""
    for key in ("confidence", "probability"):
        value = word.get(key)
        if value is not None:
            try:
                return round(float(value), 4)
            except (TypeError, ValueError):
                pass
    return None
