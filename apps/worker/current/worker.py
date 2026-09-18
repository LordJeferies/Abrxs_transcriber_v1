#!/usr/bin/env python3

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

from adapter import TranscriptionJob
from protocol import emit, emit_error, emit_log, read_commands


ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / "apps" / "transcriber" / "current" / "transcriber_app.py"

ACTIVE_JOB: TranscriptionJob | None = None


def make_command(command: dict[str, Any]) -> tuple[list[str], Path, Path, Path]:
    source_path = Path(
        str(command["sourcePath"])
    ).expanduser().resolve()

    output_dir = Path(
        str(command["outputDir"]).strip()
    ).expanduser().resolve()

    project_id = str(
        command.get("projectId", "ABRXOS")
    ).strip()

    source_id = str(
        command.get("sourceId", "DEFAULT")
    ).strip()

    model = str(
        command.get(
            "model",
            "mlx-community/whisper-large-v3-turbo",
        )
    ).strip()

    language = str(
        command.get("language", "es")
    ).strip()

    if not ENGINE.exists():
        raise FileNotFoundError(
            f"No existe transcriber_app.py: {ENGINE}"
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"No existe el master: {source_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_output = output_dir / "transcript.json"
    text_output = output_dir / "transcript.txt"

    # Contrato real de transcriber_app.py:
    #
    # transcriber_app.py MEDIA
    #   --model MODEL
    #   --language LANGUAGE
    #   --project-id PROJECT_ID
    #   --source-id SOURCE_ID
    #   --output JSON_FILE
    #   --text-output TXT_FILE
    #
    # No se pasa la palabra "media": media es el argumento posicional.
    cmd = [
        sys.executable,
        "-u",
        str(ENGINE),
        str(source_path),
        "--model",
        model,
        "--language",
        language,
        "--project-id",
        project_id,
        "--source-id",
        source_id,
        "--output",
        str(json_output),
        "--text-output",
        str(text_output),
    ]

    return cmd, source_path, output_dir, json_output


def handle_command(command: dict[str, Any]) -> bool:
    global ACTIVE_JOB

    command_type = str(
        command.get("type", "")
    ).upper()

    job_id = str(
        command.get("jobId") or uuid.uuid4().hex[:12]
    )

    if command_type == "PING":
        emit({
            "type": "PONG",
            "protocol": "abrxos.worker.v1",
            "jobId": None,
        })
        return True

    if command_type == "STATUS":
        running = bool(
            ACTIVE_JOB
            and ACTIVE_JOB.thread
            and ACTIVE_JOB.thread.is_alive()
        )

        emit({
            "type": "STATUS",
            "state": "running" if running else "idle",
            "jobId": ACTIVE_JOB.job_id if running else None,
        })
        return True

    if command_type == "CANCEL":
        if ACTIVE_JOB is None:
            emit({
                "type": "CANCELLED",
                "jobId": job_id,
                "message": "No hay ningún trabajo activo.",
            })
            return True

        ACTIVE_JOB.cancel()

        emit({
            "type": "CANCEL_REQUESTED",
            "jobId": ACTIVE_JOB.job_id,
            "message": "Cancelación solicitada.",
        })
        return True

    if command_type == "SHUTDOWN":
        if ACTIVE_JOB is not None:
            ACTIVE_JOB.cancel()

        emit({
            "type": "BYE",
        })
        return False

    if command_type != "TRANSCRIBE":
        emit_error(
            "UNKNOWN_COMMAND",
            f"Comando no reconocido: {command_type}",
            job_id=job_id,
            recoverable=True,
        )
        return True

    if ACTIVE_JOB is not None:
        if ACTIVE_JOB.thread and ACTIVE_JOB.thread.is_alive():
            emit_error(
                "JOB_ALREADY_RUNNING",
                "Ya existe una transcripción activa.",
                job_id=job_id,
                recoverable=True,
            )
            return True

    required = (
        "sourcePath",
        "projectId",
        "outputDir",
    )

    missing = [
        key for key in required
        if not command.get(key)
    ]

    if missing:
        emit_error(
            "INVALID_COMMAND",
            "Faltan campos: " + ", ".join(missing),
            job_id=job_id,
            recoverable=True,
        )
        return True

    try:
        cmd, source_path, output_dir, json_output = make_command(
            command
        )
    except Exception as exc:
        emit_error(
            "COMMAND_BUILD_FAILED",
            str(exc),
            job_id=job_id,
            recoverable=True,
        )
        return True

    emit({
        "type": "ACCEPTED",
        "jobId": job_id,
        "stage": "QUEUED",
        "message": "Trabajo aceptado.",
    })

    emit({
        "type": "PROGRESS",
        "jobId": job_id,
        "stage": "PREPARE",
        "percent": 2,
        "message": "Master validado; preparando Whisper.",
    })

    emit_log(
        "Ejecutando: " + " ".join(
            repr(part) for part in cmd
        ),
        job_id=job_id,
        stage="WORKER",
    )

    emit_log(
        f"JSON de salida: {json_output}",
        job_id=job_id,
        stage="WORKER",
    )

    ACTIVE_JOB = TranscriptionJob(
        command=cmd,
        job_id=job_id,
        source_path=source_path,
        output_dir=output_dir,
        emit=emit,
    )

    ACTIVE_JOB.start()

    return True


def main() -> int:
    emit({
        "type": "READY",
        "protocol": "abrxos.worker.v1",
        "engine": str(ENGINE),
    })

    for command in read_commands():
        try:
            keep_running = handle_command(command)
        except Exception as exc:
            emit_error(
                "WORKER_EXCEPTION",
                str(exc),
                job_id=command.get("jobId"),
                recoverable=False,
            )
            keep_running = True

        if not keep_running:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
