# vram-manager Integration

## Overview

ha-voice-pipeline does not hold GPU VRAM directly and does not register with vram-manager. It is a consumer of Ollama. Full requirements: `vram-manager/docs/requirements/ha-voice-pipeline.md`.

## Ollama proxy routing

ha-voice-pipeline currently calls Ollama at `OLLAMA_URL` directly (`pipeline/server.py` lifespan, `pipeline/ollama_client.py`). When `VRAM_MANAGER_URL` is set, route Ollama calls through the vram-manager proxy instead:

- Direct: `http://localhost:11434/api/chat`
- Via proxy: `http://<VRAM_MANAGER_URL>/ollama/api/chat`

This gives vram-manager visibility into in-flight requests so it avoids unloading Ollama while ha-voice-pipeline is actively using it.

## No service registration

ha-voice-pipeline does not own VRAM. Do not register on behalf of Ollama.

## Environment variables

- `VRAM_MANAGER_URL` — optional; enables proxy routing when set
