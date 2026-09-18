#!/usr/bin/env python3
"""
ABRXOS MLX Whisper Adapter
Adaptador para Whisper MLX Turbo en macOS Apple Silicon
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def detect_python() -> str:
    """Retorna la ruta del Python en uso"""
    return sys.executable


def detect_mlx_whisper() -> dict[str, Any]:
    """
    Detecta si mlx_whisper está disponible e inspecciona su API real.

    Returns:
        dict con claves: available, path, version, supports_word_timestamps
    """
    result: dict[str, Any] = {
        "available": False,
        "path": None,
        "version": None,
        "supports_word_timestamps": None,
        "error": None,
    }

    try:
        import mlx_whisper  # type: ignore[import]

        result["available"] = True
        result["path"] = getattr(mlx_whisper, "__file__", None)

        # Intentar obtener versión
        try:
            import importlib.metadata
            result["version"] = importlib.metadata.version("mlx-whisper")
        except Exception:
            result["version"] = getattr(mlx_whisper, "__version__", "unknown")

        # Detectar si transcribe acepta word_timestamps
        import inspect
        if hasattr(mlx_whisper, "transcribe"):
            sig = inspect.signature(mlx_whisper.transcribe)
            result["supports_word_timestamps"] = "word_timestamps" in sig.parameters
        else:
            result["supports_word_timestamps"] = False
            result["error"] = "mlx_whisper.transcribe no encontrado"

    except ImportError as exc:
        result["error"] = str(exc)

    return result


def resolve_model(config_model: str | None = None) -> str | None:
    """
    Resuelve la ruta o identificador del modelo.

    Prioridad:
    1. Argumento explícito
    2. Variable de entorno ABRXOS_WHISPER_MODEL
    3. None si no está configurado

    Args:
        config_model: Ruta o identificador del modelo desde configuración

    Returns:
        Ruta o identificador del modelo, o None
    """
    if config_model and config_model.strip() not in (
        "", "/RUTA/LOCAL/AL/MODELO", "RUTA_O_IDENTIFICADOR_DEL_MODELO"
    ):
        return config_model.strip()

    env_model = os.environ.get("ABRXOS_WHISPER_MODEL", "").strip()
    if env_model:
        return env_model

    return None


def validate_model(model: str) -> dict[str, Any]:
    """
    Valida que el modelo exista localmente o sea un identificador conocido.

    No descarga modelos automáticamente.

    Args:
        model: Ruta local o identificador HuggingFace

    Returns:
        dict con claves: valid, is_local, exists, error
    """
    result: dict[str, Any] = {
        "valid": False,
        "is_local": False,
        "exists": False,
        "error": None,
    }

    model_path = Path(model)
    if model_path.is_absolute() or model_path.exists():
        result["is_local"] = True
        if model_path.exists():
            result["valid"] = True
            result["exists"] = True
        else:
            result["error"] = f"Ruta no encontrada: {model}"
    else:
        # Identificador remoto — no descargar automáticamente
        allow_download = os.environ.get("ABRXOS_ALLOW_MODEL_DOWNLOAD", "0").strip()
        if allow_download == "1":
            result["valid"] = True
            result["is_local"] = False
            result["exists"] = True  # asumimos que HF lo resolverá
        else:
            result["error"] = (
                f"El modelo '{model}' parece un identificador remoto. "
                "La descarga automática está desactivada. "
                "Proporciona una ruta local o establece "
                "ABRXOS_ALLOW_MODEL_DOWNLOAD=1 para permitir la descarga."
            )

    return result


def transcribe_with_mlx(
    audio_path: str,
    model: str,
    language: str | None = None,
    word_timestamps: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Ejecuta la transcripción usando mlx_whisper.

    Args:
        audio_path: Ruta al archivo de audio
        model: Ruta o identificador del modelo
        language: Código de idioma (ej: 'es')
        word_timestamps: Si solicitar timestamps por palabra
        verbose: Salida detallada

    Returns:
        Resultado crudo de mlx_whisper.transcribe

    Raises:
        RuntimeError: Si mlx_whisper no está disponible
        RuntimeError: Si el modelo no está configurado
        RuntimeError: Si la API no soporta word_timestamps
    """
    info = detect_mlx_whisper()

    if not info["available"]:
        raise RuntimeError(
            f"mlx_whisper no está disponible: {info['error']}\n"
            f"Python usado: {detect_python()}\n"
            "Instala mlx-whisper con: pip install mlx-whisper"
        )

    if word_timestamps and not info["supports_word_timestamps"]:
        raise RuntimeError(
            "Esta versión de mlx_whisper no soporta word_timestamps. "
            f"Versión instalada: {info['version']}"
        )

    import mlx_whisper  # type: ignore[import]

    options: dict[str, Any] = {
        "path_or_hf_repo": model,
        "verbose": verbose,
    }

    if word_timestamps and info["supports_word_timestamps"]:
        options["word_timestamps"] = True

    if language:
        options["language"] = language

    return mlx_whisper.transcribe(audio_path, **options)


def get_adapter_info() -> dict[str, Any]:
    """Retorna información completa del adaptador y entorno"""
    mlx_info = detect_mlx_whisper()
    return {
        "python": detect_python(),
        "mlx_whisper": mlx_info,
        "platform": sys.platform,
        "architecture": _get_architecture(),
    }


def _get_architecture() -> str:
    import platform
    return platform.machine()


if __name__ == "__main__":
    import json
    info = get_adapter_info()
    print(json.dumps(info, indent=2, default=str))
