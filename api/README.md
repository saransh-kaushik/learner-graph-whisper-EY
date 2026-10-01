# Whisper Diarization API

## Local dev (no Docker)
```bash
pip install -r api/requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000/ for the Web Studio UI.
Open http://localhost:8000/docs for Swagger UI.

## Docker (GPU)
```bash
# Build from project root
docker build -f api/Dockerfile -t whisper-api .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -e DEVICE=cuda -e COMPUTE_TYPE=int8 \
  whisper-api
```

## Docker Compose (recommended for local GPU testing)
```bash
docker compose -f api/docker-compose.yml up --build
```

## Quick test
```bash
# Submit job
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"file_url": "https://your-url.com/audio.mp4", "num_speakers": 2}'

# Returns: { "job_id": "abc-123", "status": "queued" }

# Poll until done
curl http://localhost:8000/jobs/abc-123
```

## Environment variables
| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `int8` | `int8`, `float16`, `float32` |

## GCP Cloud Run
See `../gcp_cloud_run/deploy.sh`.
