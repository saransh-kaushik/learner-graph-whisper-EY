# API + Docker Implementation Plan
## `whisper-diarization-advanced`

---

## 1. Current State of the Project

The repo is already structured with separation of concerns in mind:

```
whisper-diarization-advanced/
├── core/               ← shared pipeline logic (pipeline.py, preprocess.py, metrics_utils.py)
├── local/              ← CLI runner (main.py) — works today
├── cog/                ← Replicate.com deployment (predict.py + cog.yaml) — works today
├── api/                ← Placeholder (README says TODO, requirements.txt has fastapi commented out)
├── gcp_cloud_run/      ← Placeholder (README says TODO, points to api/server.py)
└── gcp_vm/             ← Placeholder (README says TODO)
```

### What already exists and can be reused
- ✅ `core/pipeline.py` — `WhisperDiarizationPipeline` is fully implemented, accepts `file_url`, `file_path`, `file_string`, and all optional configs
- ✅ `cog/predict.py` — the exact input/output contract we want in the API is already defined here (all params, types, defaults)
- ✅ `api/requirements.txt` — already chains `core/requirements.txt`, just needs `fastapi` uncommented
- ✅ `gcp_cloud_run/requirements.txt` — already chains `api/requirements.txt`

### What needs to be built
- `api/server.py` — FastAPI application
- `api/Dockerfile` — Docker image for the API
- `api/docker-compose.yml` — for local testing with GPU
- `gcp_cloud_run/deploy.sh` — gcloud commands

---

## 2. API Design

### Why FastAPI?
- Already referenced in `api/requirements.txt` (`fastapi==0.104.0`, currently commented out)
- Async-native, automatic OpenAPI/Swagger docs at `/docs`
- Pydantic models for request validation
- Uvicorn ASGI server for production

### Key Design Decision: Synchronous vs Async Jobs

Processing a 30-minute audio takes ~2 minutes on a T4. **A synchronous HTTP request will time out** in most clients/proxies (default 30–60s).

**Recommended: Background job queue pattern**

```
POST /transcribe      → returns { job_id }  immediately
GET  /jobs/{job_id}   → returns status + result when done
```

This means:
- The request returns instantly with a `job_id`
- The client polls `GET /jobs/{job_id}` until `status == "done"`
- No timeout issues, works with any HTTP client

For the queue itself: use Python's `asyncio` + an in-memory queue (simple, no extra deps). Can upgrade to Redis/Celery later for multi-process scaling.

---

## 3. Step-by-Step Implementation

---

### Step 1 — Update `api/requirements.txt`

Uncomment `fastapi` and add `uvicorn`:

```
# api/requirements.txt
-r ../core/requirements.txt
fastapi==0.104.0
uvicorn[standard]==0.24.0
python-multipart==0.0.6
```

**Why `python-multipart`?** Required by FastAPI for form data / file uploads, even if we're primarily using JSON + URL.

---

### Step 2 — Create `api/server.py`

This is the main file. Full structure:

```python
# api/server.py
"""
FastAPI server for Whisper Diarization.
Exposes two endpoints:
  POST /transcribe  — submit a job, returns job_id
  GET  /jobs/{id}  — poll job status and result
"""
import sys, uuid, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from core.pipeline import WhisperDiarizationPipeline

app = FastAPI(title="Whisper Diarization API", version="1.0.0")

# ── Model loaded once at startup ────────────────────────────────────────────
pipeline: WhisperDiarizationPipeline = None

@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = WhisperDiarizationPipeline(device="cuda", compute_type="int8")

# ── In-memory job store ─────────────────────────────────────────────────────
# { job_id: { "status": "queued"|"processing"|"done"|"error", "result": ..., "error": ... } }
jobs: dict = {}

# ── Request / Response models ───────────────────────────────────────────────
class TranscribeRequest(BaseModel):
    file_url: Optional[str] = Field(None, description="Direct URL to audio/video file")
    file_string: Optional[str] = Field(None, description="Base64-encoded audio file")
    num_speakers: Optional[int] = Field(None, ge=1, le=50)
    translate: bool = False
    language: Optional[str] = None
    prompt: Optional[str] = None
    preprocess: int = Field(0, ge=0, le=4)
    highpass_freq: int = 45
    lowpass_freq: int = 8000
    prop_decrease: float = Field(0.3, ge=0.0, le=1.0)
    stationary: bool = True
    target_dBFS: float = -18.0

class JobStatus(BaseModel):
    job_id: str
    status: str           # queued | processing | done | error
    result: Optional[dict] = None
    error: Optional[str] = None

# ── Endpoints ───────────────────────────────────────────────────────────────
@app.post("/transcribe", response_model=JobStatus, status_code=202)
async def transcribe(req: TranscribeRequest):
    if not req.file_url and not req.file_string:
        raise HTTPException(400, "Provide file_url or file_string")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}

    # Run in background thread (pipeline is CPU/GPU blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_job, job_id, req)

    return JobStatus(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return JobStatus(job_id=job_id, **j)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Background worker ───────────────────────────────────────────────────────
def _run_job(job_id: str, req: TranscribeRequest):
    jobs[job_id]["status"] = "processing"
    try:
        result = pipeline.predict(
            file_url=req.file_url,
            file_string=req.file_string,
            num_speakers=req.num_speakers,
            translate=req.translate,
            language=req.language,
            prompt=req.prompt,
            preprocess=req.preprocess,
            highpass_freq=req.highpass_freq,
            lowpass_freq=req.lowpass_freq,
            prop_decrease=req.prop_decrease,
            stationary=req.stationary,
            target_dBFS=req.target_dBFS,
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result.to_dict()
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
```

**Key points:**
- `startup()` loads models once — same pattern as `cog/predict.py`'s `setup()`
- `run_in_executor(None, ...)` runs the blocking pipeline call in a thread pool without blocking the event loop
- `jobs` dict is in-memory — fine for single-instance deployments; replace with Redis for multi-instance
- No authentication yet — add an `API_KEY` header check when deploying publicly

---

### Step 3 — Create `api/Dockerfile`

```dockerfile
# api/Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.12 python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer cache)
COPY core/requirements.txt core/requirements.txt
COPY api/requirements.txt api/requirements.txt
RUN pip3 install --no-cache-dir -r api/requirements.txt

# Copy source
COPY core/ core/
COPY api/ api/
COPY model_cache/ model_cache/

# Expose port
EXPOSE 8000

# Run server (1 worker — GPU memory is shared, not duplicated)
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**Why `--workers 1`?**
The pipeline loads models into GPU VRAM. Multiple workers = multiple model copies = OOM on a single GPU. Keep 1 worker; concurrency is handled by the job queue. For multi-GPU, run one container per GPU with different `CUDA_VISIBLE_DEVICES`.

**Why `nvidia/cuda:12.1.1` base?**
Matches the CUDA version used in the Cog setup. Provides `libcuda.so`, `libcublas`, etc. that PyTorch and faster-whisper need.

---

### Step 4 — Create `api/docker-compose.yml` (for local testing)

```yaml
# api/docker-compose.yml
version: "3.8"

services:
  whisper-api:
    build:
      context: ..          # project root (so COPY core/ works)
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped
```

---

### Step 5 — Build and Run Locally

```bash
# From project root
docker build -f api/Dockerfile -t whisper-api .

# With GPU (requires nvidia-container-toolkit installed on host)
docker run --gpus all -p 8000:8000 whisper-api

# Or with docker-compose
docker-compose -f api/docker-compose.yml up
```

**Test it:**

```bash
# Submit a job
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "file_url": "https://your-url.com/audio.mp4",
    "num_speakers": 2,
    "preprocess": 0,
    "language": "en"
  }'

# Response: { "job_id": "abc-123", "status": "queued" }

# Poll for result
curl http://localhost:8000/jobs/abc-123

# Response when done:
# { "job_id": "abc-123", "status": "done", "result": { "segments": [...], ... } }
```

Swagger UI is auto-generated at: `http://localhost:8000/docs`

---

### Step 6 — GCP Cloud Run Deployment (optional)

> Cloud Run supports GPU instances (T4, L4) as of 2024. The container is identical.

**`gcp_cloud_run/deploy.sh`:**

```bash
#!/bin/bash
set -e

PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
IMAGE="gcr.io/$PROJECT_ID/whisper-api:latest"
SERVICE_NAME="whisper-diarization"

# Build and push from project root
gcloud builds submit \
  --tag "$IMAGE" \
  --project "$PROJECT_ID" \
  ../

# Deploy to Cloud Run with GPU
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 4 \
  --memory 16Gi \
  --timeout 3600 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 3 \
  --no-allow-unauthenticated \
  --project "$PROJECT_ID"
```

**Notes:**
- `--concurrency 1` — one request per instance; job queue handles overflow
- `--min-instances 0` — scales to zero (cold start ~30–60s due to model loading)
- `--min-instances 1` — keeps one warm instance; costs money but avoids cold starts
- `--no-allow-unauthenticated` — require Google identity token; add API key middleware if exposing publicly

---

## 4. Files to Create — Summary

| File | Status | Action |
|------|--------|--------|
| `api/requirements.txt` | Exists (incomplete) | Uncomment fastapi, add uvicorn |
| `api/server.py` | Missing | **Create** |
| `api/Dockerfile` | Missing | **Create** |
| `api/docker-compose.yml` | Missing | **Create** |
| `gcp_cloud_run/Dockerfile` | Missing | Symlink or copy from `api/Dockerfile` |
| `gcp_cloud_run/deploy.sh` | Missing | **Create** |

---

## 5. API Endpoint Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/transcribe` | Submit transcription job → returns `job_id` |
| `GET` | `/jobs/{job_id}` | Poll job status and result |
| `GET` | `/health` | Health check for load balancers |
| `GET` | `/docs` | Auto-generated Swagger UI |

### POST `/transcribe` — Full Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_url` | string | null | Direct URL to audio/video (mp4, mp3, wav, ...) |
| `file_string` | string | null | Base64-encoded audio |
| `num_speakers` | int | null | Auto-detect if null (1–50) |
| `translate` | bool | false | Translate output to English |
| `language` | string | null | Language code (`en`, `pt`, ...) |
| `prompt` | string | null | Hotwords/vocabulary hints |
| `preprocess` | int | 0 | 0=none, 1=sanitize, 2=+filter, 3=+denoise, 4=+normalize |
| `highpass_freq` | int | 45 | High-pass filter cutoff (Hz) |
| `lowpass_freq` | int | 8000 | Low-pass filter cutoff (Hz) |
| `prop_decrease` | float | 0.3 | Noise reduction strength (0.0–1.0) |
| `stationary` | bool | true | Assume stationary noise profile |
| `target_dBFS` | float | -18.0 | Target RMS normalization level |

---

## 6. Future Improvements

- **Webhook support** — `POST /transcribe` accepts a `callback_url`; server POSTs result when done (no polling needed)
- **API key auth** — simple `X-API-Key` header middleware
- **Redis job store** — replace in-memory `jobs` dict for multi-instance/persistence
- **Rate limiting** — via `slowapi` or nginx upstream
- **Multi-GPU routing** — run N containers with `CUDA_VISIBLE_DEVICES=0,1,...` behind a load balancer



## CURL:
```
# 1. Submit
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"file_url": "https://...", "num_speakers": 2}'
# → { "job_id": "abc-123", "status": "queued" }

# 2. Poll
curl http://localhost:8000/jobs/abc-123
# → { "status": "done", "result": { "segments": [...] } }
```
## commands: 
```
# From project root
docker build -f api/Dockerfile -t whisper-api .
docker run --gpus all -p 8000:8000 whisper-api

# Or with compose
docker-compose -f api/docker-compose.yml up --build

```