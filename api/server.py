"""
FastAPI server for Whisper Diarization.

Endpoints:
  POST /transcribe        — submit a job, returns job_id immediately (HTTP 202)
  POST /transcribe/batch  — submit several recordings for one learner
  GET  /jobs/{job_id}     — poll status (and result) of one job
  GET  /batches/{id}      — poll a batch: per-job statuses in one response
  GET  /health            — liveness check for load balancers / Cloud Run
  GET  /docs              — auto-generated Swagger UI (FastAPI built-in)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from core.pipeline import WhisperDiarizationPipeline
from api.knowledge_graph_routes import router as kg_router
from api.session_dates import infer_session_date

logger = logging.getLogger("api.server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Whisper Diarization API",
    description=(
        "Speech-to-text + speaker diarization via faster-whisper and pyannote. "
        "Submit a URL (or base64 audio), get back a job_id, poll for results."
    ),
    version="1.1.0",
)

# ── Pipeline (loaded once at startup, shared across all requests) ────────────
_pipeline: Optional[WhisperDiarizationPipeline] = None


@app.on_event("startup")
async def _startup() -> None:
    global _pipeline
    device = os.getenv("DEVICE", "cuda")
    compute_type = os.getenv("COMPUTE_TYPE", "int8")
    _pipeline = WhisperDiarizationPipeline(device=device, compute_type=compute_type)


# ── In-memory job store ───────────────────────────────────────────────────────
# For multi-instance deployments, replace with Redis.
_jobs: Dict[str, dict] = {}
_batches: Dict[str, dict] = {}
_store_lock = threading.Lock()
_JOB_TTL_SEC = 6 * 3600

# At most 2 sessions process simultaneously (GPU / LLM cost control).
# Jobs wait in the executor queue with status "queued".
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")

_TERMINAL = {"done", "error"}
_KG_TERMINAL = {"done", "error", "skipped"}


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
        None, ge=1, le=50, description="Number of speakers. Leave null to auto-detect.",
    )
    translate: bool = Field(False, description="Translate output to English")
    language: Optional[str] = Field(
        None, description="Language code, e.g. 'en', 'pt'. Leave null to auto-detect.",
    )
    prompt: Optional[str] = Field(
        None, description="Hotwords / vocabulary hints separated by punctuation.",
    )
    preprocess: int = Field(
        0, ge=0, le=4,
        description="Audio preprocessing level: 0=none, 1=sanitize, 2=+filter, 3=+denoise, 4=+normalize",
    )
    highpass_freq: int = Field(45, description="High-pass filter cutoff (Hz)")
    lowpass_freq: int = Field(8000, description="Low-pass filter cutoff (Hz)")
    prop_decrease: float = Field(0.3, ge=0.0, le=1.0, description="Noise reduction strength (0.0–1.0)")
    stationary: bool = Field(True, description="Assume stationary noise profile")
    target_dBFS: float = Field(-18.0, description="Target RMS normalization level (dBFS)")
    learner_id: Optional[str] = Field(None, description="Optional Learner ID for Knowledge Graph ingestion")
    tutor_id: Optional[str] = Field(None, description="Optional Tutor ID for Knowledge Graph ingestion")
    session_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date the session took place (YYYY-MM-DD). Inferred from the URL, else today.",
    )
    speaker_map: Optional[Dict[str, str]] = Field(
        None, description='Optional {"SPEAKER_00": "learner", "SPEAKER_01": "tutor"}',
    )


class BatchTranscribeRequest(BaseModel):
    """Submit multiple audio URLs for one learner in one go."""
    file_urls: List[str] = Field(
        ..., description="List of direct audio/video URLs to process.", min_length=1, max_length=20,
    )
    session_dates: Optional[List[Optional[str]]] = Field(
        None,
        description="Optional session date (YYYY-MM-DD) per URL, same order as file_urls. "
                    "Missing dates are inferred from the URL, else today.",
    )
    learner_id: str = Field(..., description="Learner ID — required for batch KG ingestion")
    tutor_id: str = Field(..., description="Tutor ID — required for batch KG ingestion")
    language: Optional[str] = Field(None)
    translate: bool = Field(False)
    preprocess: int = Field(0, ge=0, le=4)
    num_speakers: Optional[int] = Field(None, ge=1, le=50)
    prompt: Optional[str] = Field(None)

    @model_validator(mode="after")
    def _check_dates(self) -> "BatchTranscribeRequest":
        if self.session_dates is not None:
            if len(self.session_dates) != len(self.file_urls):
                raise ValueError("session_dates must have one entry per file_url")
            for d in self.session_dates:
                if d and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                    raise ValueError(f"Invalid date '{d}', expected YYYY-MM-DD")
        return self


class JobStatus(BaseModel):
    job_id: str
    status: str = Field(description="Transcription: queued | processing | done | error")
    kg_status: str = Field(
        "not_requested",
        description="Knowledge-graph analysis: not_requested | pending | running | done | error | skipped",
    )
    session_id: Optional[str] = None
    session_date: Optional[str] = None
    file_url: Optional[str] = None
    result: Optional[dict] = Field(None, description="Full pipeline output when status=done")
    error: Optional[str] = Field(None, description="Error message when status=error")
    kg_error: Optional[str] = None


class BatchJob(BaseModel):
    job_id: str
    file_url: Optional[str] = None
    session_date: Optional[str] = None
    status: str
    kg_status: str
    error: Optional[str] = None


class BatchStatus(BaseModel):
    batch_id: str
    learner_id: str
    tutor_id: str
    total: int
    queued: int
    processing: int
    done: int
    error: int
    kg_done: int
    kg_error: int
    complete: bool = Field(description="True when every job has finished transcription and analysis")
    job_ids: List[str]
    jobs: List[BatchJob]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _session_id_for(req: TranscribeRequest) -> str:
    """Deterministic id for URL submissions so re-submitting a recording
    replaces its earlier analysis instead of creating a duplicate session."""
    if req.file_url and req.learner_id:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{req.learner_id}|{req.file_url}"))
    return str(uuid.uuid4())


def _new_job(req: TranscribeRequest) -> str:
    job_id = str(uuid.uuid4())
    wants_kg = bool(req.learner_id and req.tutor_id)
    session_date = req.session_date or infer_session_date(req.file_url) or date.today().isoformat()
    with _store_lock:
        _jobs[job_id] = {
            "status": "queued",
            "kg_status": "pending" if wants_kg else "not_requested",
            "session_id": _session_id_for(req) if wants_kg else None,
            "session_date": session_date,
            "file_url": req.file_url,
            "result": None,
            "error": None,
            "kg_error": None,
            "finished_at": None,
        }
    return job_id


def _prune_jobs() -> None:
    """Drop finished jobs (and their batches) older than the TTL."""
    cutoff = time.time() - _JOB_TTL_SEC
    with _store_lock:
        stale = [jid for jid, j in _jobs.items() if j["finished_at"] and j["finished_at"] < cutoff]
        for jid in stale:
            del _jobs[jid]
        for bid in [b for b, v in _batches.items() if not any(j in _jobs for j in v["job_ids"])]:
            del _batches[bid]


def _short_error(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return msg if len(msg) <= 500 else msg[:497] + "…"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post(
    "/transcribe",
    response_model=JobStatus,
    response_model_exclude_none=True,
    status_code=202,
    summary="Submit a transcription job",
)
async def transcribe(req: TranscribeRequest) -> JobStatus:
    """
    Submit audio for transcription + diarization.

    Returns a `job_id` immediately. Poll `GET /jobs/{job_id}` until `status`
    is `"done"` or `"error"` (and `kg_status` is terminal, if learner/tutor ids were given).
    """
    if not req.file_url and not req.file_string:
        raise HTTPException(status_code=400, detail="Provide file_url or file_string")
    _prune_jobs()
    job_id = _new_job(req)
    asyncio.get_running_loop().run_in_executor(_executor, _run_job, job_id, req)
    return _job_status(job_id, include_result=False)


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    response_model_exclude_none=True,
    summary="Poll a job's status (and result)",
)
async def get_job(
    job_id: str,
    include_result: bool = Query(True, description="Set false for lightweight status polling"),
) -> JobStatus:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _job_status(job_id, include_result=include_result)


def _job_status(job_id: str, include_result: bool) -> JobStatus:
    j = _jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=j["status"],
        kg_status=j["kg_status"],
        session_id=j["session_id"],
        session_date=j["session_date"],
        file_url=j["file_url"],
        result=j["result"] if include_result else None,
        error=j["error"],
        kg_error=j["kg_error"],
    )


@app.get("/health", summary="Liveness check")
async def health() -> dict:
    """Returns 200 OK when the server is up. Used by Cloud Run / load balancers."""
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}


@app.post(
    "/transcribe/batch",
    response_model=BatchStatus,
    status_code=202,
    summary="Submit multiple audio URLs for one learner",
)
async def transcribe_batch(req: BatchTranscribeRequest) -> BatchStatus:
    """
    Submit multiple session recordings for a single learner.
    They are processed 2 at a time. Poll `/batches/{batch_id}` — it returns
    every job's status, so one request per poll is enough.
    """
    _prune_jobs()
    loop = asyncio.get_running_loop()
    job_ids = []
    for i, url in enumerate(req.file_urls):
        single_req = TranscribeRequest(
            file_url=url,
            num_speakers=req.num_speakers,
            language=req.language,
            translate=req.translate,
            preprocess=req.preprocess,
            prompt=req.prompt,
            learner_id=req.learner_id,
            tutor_id=req.tutor_id,
            session_date=(req.session_dates[i] if req.session_dates else None) or None,
        )
        job_id = _new_job(single_req)
        job_ids.append(job_id)
        loop.run_in_executor(_executor, _run_job, job_id, single_req)

    batch_id = str(uuid.uuid4())
    with _store_lock:
        _batches[batch_id] = {"job_ids": job_ids, "learner_id": req.learner_id, "tutor_id": req.tutor_id}
    return _build_batch_status(batch_id)


@app.get("/batches/{batch_id}", response_model=BatchStatus, summary="Poll overall batch progress")
async def get_batch(batch_id: str) -> BatchStatus:
    if batch_id not in _batches:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    return _build_batch_status(batch_id)


def _build_batch_status(batch_id: str) -> BatchStatus:
    b = _batches[batch_id]
    counts = {"queued": 0, "processing": 0, "done": 0, "error": 0}
    kg_done = kg_error = 0
    jobs = []
    complete = True
    for jid in b["job_ids"]:
        j = _jobs.get(jid)
        if j is None:
            continue
        counts[j["status"]] = counts.get(j["status"], 0) + 1
        kg_done += j["kg_status"] == "done"
        kg_error += j["kg_status"] == "error"
        if j["status"] not in _TERMINAL or (j["status"] == "done" and j["kg_status"] not in _KG_TERMINAL):
            complete = False
        jobs.append(BatchJob(
            job_id=jid,
            file_url=j["file_url"],
            session_date=j["session_date"],
            status=j["status"],
            kg_status=j["kg_status"],
            error=j["error"] or j["kg_error"],
        ))
    return BatchStatus(
        batch_id=batch_id,
        learner_id=b["learner_id"],
        tutor_id=b["tutor_id"],
        total=len(b["job_ids"]),
        kg_done=kg_done,
        kg_error=kg_error,
        complete=complete,
        job_ids=b["job_ids"],
        jobs=jobs,
        **counts,
    )


app.include_router(kg_router)


# ── Static Files (Serve Frontend UI) ──────────────────────────────────────────
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


# ── Background worker ────────────────────────────────────────────────────────
def _run_job(job_id: str, req: TranscribeRequest) -> None:
    """Blocking; runs on the 2-worker executor (status stays "queued" until picked up)."""
    job = _jobs[job_id]
    job["status"] = "processing"
    try:
        if _pipeline is None:
            raise RuntimeError("Transcription pipeline is not loaded yet")
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
        ).to_dict()
        job["result"] = result
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed for job %s", job_id)
        job["status"] = "error"
        job["error"] = _short_error(exc)
        if job["kg_status"] == "pending":
            job["kg_status"] = "skipped"
        job["finished_at"] = time.time()
        return

    if job["kg_status"] == "pending":
        job["kg_status"] = "running"
        try:
            from knowledge_graph.ingest.orchestrator import ingest_session

            ingest_session(
                transcript_data=result,
                learner_id=req.learner_id,
                tutor_id=req.tutor_id,
                session_date=job["session_date"],
                session_id=job["session_id"],
                speaker_map=req.speaker_map,
                source_url=req.file_url,
            )
            job["kg_status"] = "done"
            logger.info("Ingested job %s to KG for learner %s", job_id, req.learner_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("KG ingestion failed for job %s", job_id)
            job["kg_status"] = "error"
            job["kg_error"] = _short_error(exc)

    job["finished_at"] = time.time()
