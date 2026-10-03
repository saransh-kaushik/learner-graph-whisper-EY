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

## Batch sessions for one learner
```bash
curl -X POST http://localhost:8000/transcribe/batch \
  -H "Content-Type: application/json" \
  -d '{"file_urls": ["https://…/2026/Sep/21/a.mp4", "https://…/b.mp4"],
       "session_dates": [null, "2026-09-24"],
       "learner_id": "learner_001", "tutor_id": "tutor_001"}'

# One request returns every job's status (transcription + analysis)
curl http://localhost:8000/batches/<batch_id>
```
- Session dates come from `session_dates`, else from the URL (`/2026/Sep/21/` or `2026-09-21`), else today.
  Correct dates matter: progress trends are computed in date order.
- `complete: true` means every recording is transcribed **and** analysed (`kg_status` terminal).
- Poll `/jobs/{id}?include_result=false` for lightweight status; fetch the result once.
- Re-submitting the same URL for the same learner replaces that session's analysis (no duplicates).

## Progress reports
| Method | Endpoint | Notes |
|---|---|---|
| GET | `/kg/learners/{id}/report?n=5&refresh=false` | Cached until the learner has a new session; `refresh=true` regenerates |
| GET | `/kg/learners/{id}/graph?n=5` | Graph data for the tutor view (browser never connects to Neo4j) |
| POST | `/kg/sessions/{id}/reanalyze` | Re-run analysis on a stored session with the current prompts |
| GET | `/kg/ingest/{session_id}` | Status of a `/kg/ingest` or re-analysis request |

Learner-facing report: http://localhost:8000/dashboard.html?learner=<id>

## Migrating existing data (one-time)
Sessions ingested before this version used random ids, free-form grammar names and
didn't store transcripts, so they can't be re-analysed in place. To rebuild a learner:
```bash
python -m knowledge_graph.schema                                   # new indexes
python -m knowledge_graph.maintenance reset-learner --learner-id <id> --yes
# then re-submit the learner's recordings via /transcribe/batch
```
Sessions ingested from now on can be re-analysed without re-transcribing:
`python -m knowledge_graph.ingest.orchestrator --reanalyze-learner <id>`.

## Environment variables
| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `int8` | `int8`, `float16`, `float32` |

## GCP Cloud Run
See `../gcp_cloud_run/deploy.sh`.
