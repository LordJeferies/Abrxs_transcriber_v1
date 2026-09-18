#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Iterator


_WRITE_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event: dict[str, Any]) -> None:
    """
    stdout es exclusivamente JSONL.
    No escribir texto humano directamente en stdout.
    """
    payload = dict(event)
    payload.setdefault("time", now_iso())

    line = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    with _WRITE_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def emit_log(
    message: str,
    *,
    job_id: str | None = None,
    stage: str = "WORKER",
) -> None:
    emit({
        "type": "LOG",
        "jobId": job_id,
        "stage": stage,
        "message": message,
    })


def emit_error(
    code: str,
    message: str,
    *,
    job_id: str | None = None,
    details: str | None = None,
    recoverable: bool = False,
) -> None:
    event: dict[str, Any] = {
        "type": "ERROR",
        "jobId": job_id,
        "code": code,
        "message": message,
        "recoverable": recoverable,
    }

    if details:
        event["details"] = details

    emit(event)


def read_commands() -> Iterator[dict[str, Any]]:
    """
    Lee exactamente un objeto JSON por línea.
    """
    for raw_line in sys.stdin:
        line = raw_line.strip()

        if not line:
            continue

        try:
            command = json.loads(line)
        except json.JSONDecodeError as exc:
            emit_error(
                "INVALID_JSON",
                "El comando debe ser JSON válido en una sola línea.",
                details=str(exc),
                recoverable=True,
            )
            continue

        if not isinstance(command, dict):
            emit_error(
                "INVALID_COMMAND",
                "El comando debe ser un objeto JSON.",
                recoverable=True,
            )
            continue

        yield command
