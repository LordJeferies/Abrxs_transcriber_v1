# Transcriber App — Start Here

Esta es la aplicación principal del transcriptor.

## Estructura

- `transcriber_app.py` — punto de entrada CLI
- `transcription_engine.py` — lógica principal de transcripción
- `mlx_whisper_adapter.py` — adaptador para Whisper MLX
- `alignment_engine.py` — alineación y normalización
- `schema_validator.py` — validación del JSON canónico
- `export_engine.py` — exportación para Content Builder
- `config.example.json` — plantilla de configuración

## Uso

Configura primero:

```bash
cp config.example.json config.json
# Edita config.json con la ruta real del modelo
```

Ejecuta:

```bash
python3 transcriber_app.py \
  --media "/ruta/al/master.mov" \
  --model "/ruta/al/modelo" \
  --language es \
  --project-id JOC55_AMANDA \
  --source-id DEFAULT \
  --output "/ruta/de/salida/transcript.json"
```

O usa los comandos en `tools/`.
