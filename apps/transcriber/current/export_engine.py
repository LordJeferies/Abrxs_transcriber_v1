#!/usr/bin/env python3
"""
ABRXOS Export Engine
Genera exportaciones para Content Builder
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def export_for_content_builder(
    transcript: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Exporta transcripción para Content Builder
    
    Genera:
    - 04_TRANSCRIPCION.txt (legible para ChatGPT)
    - 04_TRANSCRIPCION.json (copia del transcript)
    - 04_TRANSCRIPCION_REF.json (referencia estructurada)
    
    Args:
        transcript: Diccionario con la transcripción
        output_dir: Directorio de salida
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 04_TRANSCRIPCION.txt
    txt_path = output_dir / "04_TRANSCRIPCION.txt"
    lines: list[str] = []

    lines.append(f"TRANSCRIPT_ID: {transcript['transcriptId']}")
    lines.append(f"PROJECT_ID: {transcript['projectId']}")
    lines.append(f"SOURCE_ID: {transcript['sourceId']}")
    lines.append(f"STATUS: {transcript['status']}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    for cue in transcript.get("cues", []):
        start = cue["start"]
        end = cue["end"]
        
        # Convertir segundos a HH:MM:SS.mmm
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

        lines.append(f"[{format_time(start)} - {format_time(end)}]")
        lines.append(f"CUE_ID: {cue['id']}")
        lines.append(f"WORD_START_ID: {cue['wordStartId']}")
        lines.append(f"WORD_END_ID: {cue['wordEndId']}")
        lines.append("")
        lines.append(cue.get("text", ""))
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Creado: {txt_path}")

    # 04_TRANSCRIPCION.json
    json_path = output_dir / "04_TRANSCRIPCION.json"
    json_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"✓ Creado: {json_path}")

    # 04_TRANSCRIPCION_REF.json
    ref_path = output_dir / "04_TRANSCRIPCION_REF.json"
    ref = {
        "transcriptId": transcript["transcriptId"],
        "projectId": transcript["projectId"],
        "sourceId": transcript["sourceId"],
        "path": "04_TRANSCRIPCION.json",
        "wordTimingAvailable": transcript["quality"]["wordTimingAvailable"],
        "status": transcript["status"],
        "totalWords": len(transcript["words"]),
        "totalSegments": len(transcript["segments"]),
        "totalCues": len(transcript["cues"]),
        "durationSeconds": transcript["source"].get("durationSeconds"),
        "language": transcript["source"].get("language"),
    }

    ref_path.write_text(
        json.dumps(ref, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"✓ Creado: {ref_path}")


def main() -> int:
    if len(sys.argv) != 3:
        print("Uso: python export_engine.py transcript.json output_dir")
        return 2

    transcript_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not transcript_path.exists():
        print(f"ERROR: No existe {transcript_path}", file=sys.stderr)
        return 1

    try:
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: JSON inválido: {exc}", file=sys.stderr)
        return 1

    export_for_content_builder(transcript, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
