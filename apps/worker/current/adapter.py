#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable


Emit = Callable[[dict[str, Any]], None]


PROGRESS_PATTERNS = [
    re.compile(
        r"(?i)\bprogress\b[^0-9]{0,20}"
        r"(\d{1,3}(?:\.\d+)?)\s*%"
    ),
    re.compile(
        r"(?i)\b(\d{1,3}(?:\.\d+)?)\s*%"
    ),
    re.compile(
        r"(?i)\bsegment\s+(\d+)\s*(?:/|of)\s*(\d+)"
    ),
    re.compile(
        r"(?i)\bword\s+(\d+)\s*(?:/|of)\s*(\d+)"
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_progress(text: str) -> float | None:
    for pattern in PROGRESS_PATTERNS:
        match = pattern.search(text)

        if not match:
            continue

        groups = match.groups()

        if len(groups) == 1:
            value = float(groups[0])
            return max(0.0, min(100.0, value))

        if len(groups) >= 2:
            current = float(groups[0])
            total = float(groups[1])

            if total > 0:
                return max(
                    0.0,
                    min(100.0, current / total * 100.0),
                )

    return None


def validate_json(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"No existe: {path}"

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return False, f"JSON inválido: {exc}"

    if not isinstance(data, dict):
        return False, "El JSON raíz no es un objeto."

    return True, "OK"


class TranscriptionJob:
    def __init__(
        self,
        command: list[str],
        job_id: str,
        source_path: Path,
        output_dir: Path,
        emit: Emit,
    ) -> None:
        self.command = command
        self.job_id = job_id
        self.source_path = source_path
        self.output_dir = output_dir
        self.emit = emit

        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.cancel_requested = threading.Event()

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            name=f"transcription-{self.job_id}",
            daemon=True,
        )
        self.thread.start()

    def cancel(self) -> None:
        self.cancel_requested.set()

        process = self.process

        if process is None:
            return

        if process.poll() is not None:
            return

        try:
            os.killpg(
                process.pid,
                signal.SIGTERM,
            )
        except ProcessLookupError:
            pass
        except PermissionError:
            process.terminate()

    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("jobId", self.job_id)
        self.emit(event)

    def _run(self) -> None:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._emit({
            "type": "STARTED",
            "stage": "TRANSCRIBER",
            "message": "Iniciando transcriber_app.py",
        })

        self._emit({
            "type": "PROGRESS",
            "stage": "WHISPER",
            "percent": 3,
            "message": "Cargando modelo Whisper MLX...",
        })

        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            self._emit({
                "type": "ERROR",
                "code": "PROCESS_START_FAILED",
                "message": str(exc),
                "recoverable": True,
            })
            return

        stdout_thread = threading.Thread(
            target=self._read_stream,
            args=(self.process.stdout, "STDOUT"),
            daemon=True,
        )

        stderr_thread = threading.Thread(
            target=self._read_stream,
            args=(self.process.stderr, "STDERR"),
            daemon=True,
        )

        stdout_thread.start()
        stderr_thread.start()

        return_code = self.process.wait()

        stdout_thread.join(timeout=3)
        stderr_thread.join(timeout=3)

        if self.cancel_requested.is_set():
            self._emit({
                "type": "CANCELLED",
                "stage": "TRANSCRIBER",
                "message": "Transcripción cancelada.",
            })
            return

        if return_code != 0:
            self._emit({
                "type": "ERROR",
                "code": "TRANSCRIBER_FAILED",
                "message": (
                    "transcriber_app.py terminó con código "
                    f"{return_code}."
                ),
                "recoverable": True,
            })
            return

        json_path = self.output_dir / "transcript.json"
        txt_path = self.output_dir / "transcript.txt"

        json_ok, json_message = validate_json(json_path)

        if not json_ok:
            self._emit({
                "type": "ERROR",
                "code": "TRANSCRIPT_INVALID",
                "message": json_message,
                "recoverable": False,
            })
            return

        files = {
            "json": str(json_path.resolve()),
            "txt": str(txt_path.resolve()),
        }

        if txt_path.exists():
            files["txt"] = str(txt_path.resolve())

        self._emit({
            "type": "PROGRESS",
            "stage": "VALIDATION",
            "percent": 98,
            "message": "Validando transcript.json...",
        })

        self._emit({
            "type": "DONE",
            "stage": "VALIDATION",
            "percent": 100,
            "message": "Transcripción completada.",
            "sourceSha256": sha256_file(
                self.source_path
            ),
            "files": files,
        })

    def _read_stream(
        self,
        stream,
        stream_name: str,
    ) -> None:
        if stream is None:
            return

        for raw_line in stream:
            text = raw_line.strip()

            if not text:
                continue

            progress = parse_progress(text)

            if progress is not None:
                self._emit({
                    "type": "PROGRESS",
                    "stage": "WHISPER",
                    "percent": progress,
                    "message": text,
                })
            else:
                self._emit({
                    "type": "LOG",
                    "stage": stream_name,
                    "message": text,
                })
