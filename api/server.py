"""
FastAPI server for Whisper Diarization.

Endpoints:
  POST /transcribe      — submit a job, returns job_id immediately (HTTP 202)
  GET  /jobs/{job_id}   — poll status and result
  GET  /health          — liveness check for load balancers / Cloud Run
  GET  /docs            — auto-generated Swagger UI (FastAPI built-in)
"""
from __future__ import annotations

import asyncio
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.pipeline import WhisperDiarizationPipeline
from api.knowledge_graph_routes import router as kg_router


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Whisper Diarization API",
    description=(
        "Speech-to-text + speaker diarization via faster-whisper and pyannote. "
        "Submit a URL (or base64 audio), get back a job_id, poll for results."
    ),
    version="1.0.0",
)

# ── Pipeline (loaded once at startup, shared across all requests) ────────────
_pipeline: Optional[WhisperDiarizationPipeline] = None


@app.on_event("startup")
async def _startup() -> None:
    global _pipeline
    import os
    device = os.getenv("DEVICE", "cuda")
    compute_type = os.getenv("COMPUTE_TYPE", "int8")
    _pipeline = WhisperDiarizationPipeline(device=device, compute_type=compute_type)


# ── In-memory job store ───────────────────────────────────────────────────────
# Structure: { job_id: {"status": str, "result": dict|None, "error": str|None} }
# For multi-instance deployments, replace with Redis.
_jobs: Dict[str, dict] = {}

# ── Batch store ───────────────────────────────────────────────────────────────
# Structure: { batch_id: {"job_ids": [...], "learner_id": str, "tutor_id": str} }
_batches: Dict[str, dict] = {}

# ── Concurrency limiter ───────────────────────────────────────────────────────
# At most 2 sessions process simultaneously (GPU / LLM cost control).
_session_semaphore = threading.Semaphore(2)


# ── Pydantic models ───────────────────────────────────────────────────────────
class TranscribeRequest(BaseModel):
    file_url: Optional[str] = Field(
        None,
        description="Direct URL to an audio or video file (mp4, mp3, wav, m4a, …)",
        examples=["https://example.com/recording.mp4"],
    )
    file_string: Optional[str] = Field(
        None,
        description="Base64-encoded audio file (alternative to file_url)",
    )
    num_speakers: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Number of speakers. Leave null to auto-detect.",
    )
    translate: bool = Field(False, description="Translate output to English")
    language: Optional[str] = Field(
        None,
        description="Language code, e.g. 'en', 'pt'. Leave null to auto-detect.",
    )
    prompt: Optional[str] = Field(
        None,
        description="Hotwords / vocabulary hints separated by punctuation.",
    )
    preprocess: int = Field(
        0,
        ge=0,
        le=4,
        description=(
            "Audio preprocessing level: "
            "0=none, 1=sanitize, 2=+filter, 3=+denoise, 4=+normalize"
        ),
    )
    highpass_freq: int = Field(45, description="High-pass filter cutoff (Hz)")
    lowpass_freq: int = Field(8000, description="Low-pass filter cutoff (Hz)")
    prop_decrease: float = Field(
        0.3, ge=0.0, le=1.0, description="Noise reduction strength (0.0–1.0)"
    )
    stationary: bool = Field(True, description="Assume stationary noise profile")
    target_dBFS: float = Field(-18.0, description="Target RMS normalization level (dBFS)")
    learner_id: Optional[str] = Field(None, description="Optional Learner ID for Knowledge Graph ingestion")
    tutor_id: Optional[str] = Field(None, description="Optional Tutor ID for Knowledge Graph ingestion")


class BatchTranscribeRequest(BaseModel):
    """Submit multiple audio URLs for one learner in one go."""
    file_urls: List[str] = Field(
        ...,
        description="List of direct audio/video URLs to process.",
        min_length=1,
        max_length=20,
    )
    learner_id: str = Field(..., description="Learner ID — required for batch KG ingestion")
    tutor_id: str = Field(..., description="Tutor ID — required for batch KG ingestion")
    language: Optional[str] = Field(None)
    translate: bool = Field(False)
    preprocess: int = Field(0, ge=0, le=4)
    num_speakers: Optional[int] = Field(None, ge=1, le=50)
    prompt: Optional[str] = Field(None)


class BatchStatus(BaseModel):
    batch_id: str
    learner_id: str
    tutor_id: str
    total: int
    queued: int
    processing: int
    done: int
    error: int
    job_ids: List[str]

class JobStatus(BaseModel):
    job_id: str
    status: str = Field(
        description="One of: queued | processing | done | error"
    )
    result: Optional[dict] = Field(None, description="Full pipeline output when status=done")
    error: Optional[str] = Field(None, description="Error message when status=error")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post(
    "/transcribe",
    response_model=JobStatus,
    status_code=202,
    summary="Submit a transcription job",
    response_description="Job accepted. Use job_id to poll /jobs/{job_id} for results.",
)
async def transcribe(req: TranscribeRequest) -> JobStatus:
    """
    Submit audio for transcription + diarization.

    Returns a `job_id` immediately. Poll `GET /jobs/{job_id}` until
    `status` is `"done"` or `"error"`.
    """
    if not req.file_url and not req.file_string:
        raise HTTPException(status_code=400, detail="Provide file_url or file_string")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    # Run the blocking pipeline call in the default thread-pool executor so
    # the event loop stays responsive to other requests (e.g. /health, polling).
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_job, job_id, req)

    return JobStatus(job_id=job_id, status="queued")


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Poll a job's status and result",
)
async def get_job(job_id: str) -> JobStatus:
    """
    Poll the status of a previously submitted transcription job.

    - `queued` — waiting to start
    - `processing` — running transcription / diarization
    - `done` — finished; `result` contains the full output
    - `error` — failed; `error` contains the exception message
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    j = _jobs[job_id]
    return JobStatus(job_id=job_id, **j)


@app.get("/health", summary="Liveness check")
async def health() -> dict:
    """Returns 200 OK when the server is up. Used by Cloud Run / load balancers."""
    return {"status": "ok"}


@app.post(
    "/transcribe/batch",
    response_model=BatchStatus,
    status_code=202,
    summary="Submit multiple audio URLs for one learner",
)
async def transcribe_batch(req: BatchTranscribeRequest) -> BatchStatus:
    """
    Submit multiple session URLs for a single learner in one go.
    They are queued and processed with max 2 concurrent sessions.
    Poll /batches/{batch_id} for overall status, or /jobs/{job_id} per session.
    """
    batch_id = str(uuid.uuid4())
    job_ids = []

    for url in req.file_urls:
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {"status": "queued", "result": None, "error": None}
        job_ids.append(job_id)

        # Build a single-session request for each URL
        single_req = TranscribeRequest(
            file_url=url,
            num_speakers=req.num_speakers,
            language=req.language,
            translate=req.translate,
            preprocess=req.preprocess,
            prompt=req.prompt,
            learner_id=req.learner_id,
            tutor_id=req.tutor_id,
        )
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_job, job_id, single_req)

    _batches[batch_id] = {
        "job_ids": job_ids,
        "learner_id": req.learner_id,
        "tutor_id": req.tutor_id,
    }

    return _build_batch_status(batch_id)


@app.get(
    "/batches/{batch_id}",
    response_model=BatchStatus,
    summary="Poll overall batch progress",
)
async def get_batch(batch_id: str) -> BatchStatus:
    """Returns aggregated status across all jobs in the batch."""
    if batch_id not in _batches:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    return _build_batch_status(batch_id)


def _build_batch_status(batch_id: str) -> BatchStatus:
    b = _batches[batch_id]
    counts = {"queued": 0, "processing": 0, "done": 0, "error": 0}
    for jid in b["job_ids"]:
        st = _jobs.get(jid, {}).get("status", "queued")
        counts[st] = counts.get(st, 0) + 1
    return BatchStatus(
        batch_id=batch_id,
        learner_id=b["learner_id"],
        tutor_id=b["tutor_id"],
        total=len(b["job_ids"]),
        job_ids=b["job_ids"],
        **counts,
    )


app.include_router(kg_router)


# ── Static Files (Serve Frontend UI) ──────────────────────────────────────────
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


# ── Background worker ────────────────────────────────────────────────────────
def _run_job(job_id: str, req: TranscribeRequest) -> None:
    """Blocking function; runs in a thread-pool executor.
    Acquires a semaphore slot before processing, so at most 2 jobs
    run simultaneously regardless of how many are queued.
    """
    # Block here (in the thread) until a concurrency slot is free.
    # Status stays "queued" while waiting.
    _session_semaphore.acquire()
    try:
        _jobs[job_id]["status"] = "processing"
        try:
            result = _pipeline.predict(
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
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result.to_dict()

            # Knowledge Graph Ingestion Integration
            if req.learner_id and req.tutor_id:
                try:
                    import json
                    import tempfile
                    from datetime import date

                    from knowledge_graph.ingest.orchestrator import ingest_session

                    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
                        json.dump(result.to_dict(), f)
                        temp_path = f.name

                    ingest_session(
                        transcript_path=temp_path,
                        learner_id=req.learner_id,
                        tutor_id=req.tutor_id,
                        session_date=date.today().isoformat(),
                    )
                    print(f"INFO: Successfully ingested job {job_id} to KG for learner {req.learner_id}")
                except Exception:
                    tb_kg = traceback.format_exc()
                    print(f"ERROR: KG ingestion failed for job {job_id}:\n{tb_kg}", file=sys.stderr, flush=True)

        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"ERROR job {job_id}:\n{tb}", file=sys.stderr, flush=True)
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = tb
    finally:
        # Always release — even on error — so queued jobs can proceed
        _session_semaphore.release()

