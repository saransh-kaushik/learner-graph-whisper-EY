"""
api/knowledge_graph_routes.py
──────────────────────────────
FastAPI routes for the Knowledge Graph module.
"""
from __future__ import annotations

import json
import logging
import tempfile
import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models.kg_models import (
    CreateLearnerRequest,
    CreateTutorRequest,
    IngestRequest,
    IngestResponse,
    LearnerSummary,
    ProgressReport,
    SearchResult,
    SemanticSearchRequest,
    SessionSummary,
)
from knowledge_graph.db import neo4j_client
from knowledge_graph.ingest.orchestrator import ingest_session
from knowledge_graph.reports.generator import generate_report
from knowledge_graph.search import find_similar_errors, find_similar_sessions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["Knowledge Graph"])


# ── Ingestion ─────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_transcript(
    req: IngestRequest, background_tasks: BackgroundTasks
) -> IngestResponse:
    """Ingest a diarized transcript into the knowledge graph."""
    if not req.transcript:
        raise HTTPException(status_code=400, detail="transcript body is required")

    session_id = req.session_id or str(uuid.uuid4())

    def _run_ingest() -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(req.transcript, f)
            temp_path = f.name

        try:
            ingest_session(
                transcript_path=temp_path,
                learner_id=req.learner_id,
                tutor_id=req.tutor_id,
                session_date=req.session_date,
                speaker_map=req.speaker_map,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)

    background_tasks.add_task(_run_ingest)
    return IngestResponse(session_id=session_id, message="Ingestion started in background.")


# ── Learners & Tutors ──────────────────────────────────────────────────────────

@router.post("/learners")
async def create_learner(req: CreateLearnerRequest) -> dict:
    neo4j_client.run_write(
        """
        MERGE (l:Learner {id: $id})
        SET l.name = $name,
            l.native_language = $native_language,
            l.current_cefr_level = $cefr_level,
            l.created_at = coalesce(l.created_at, date())
        """,
        id=req.id,
        name=req.name,
        native_language=req.native_language,
        cefr_level=req.current_cefr_level,
    )
    return {"status": "ok"}


@router.post("/tutors")
async def create_tutor(req: CreateTutorRequest) -> dict:
    neo4j_client.run_write(
        """
        MERGE (t:Tutor {id: $id})
        SET t.name = $name,
            t.specialization = $spec,
            t.created_at = coalesce(t.created_at, date())
        """,
        id=req.id,
        name=req.name,
        spec=req.specialization,
    )
    return {"status": "ok"}


@router.get("/learners", response_model=List[LearnerSummary])
async def list_learners() -> List[LearnerSummary]:
    rows = neo4j_client.run_query(
        """
        MATCH (l:Learner)
        OPTIONAL MATCH (l)-[:ATTENDED]->(s:Session)
        RETURN l.id AS id, l.name AS name,
               l.native_language AS native_language,
               l.current_cefr_level AS current_cefr_level,
               count(s) AS session_count
        """
    )
    return [LearnerSummary(**dict(r)) for r in rows]


@router.get("/learners/{learner_id}", response_model=LearnerSummary)
async def get_learner(learner_id: str) -> LearnerSummary:
    rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $id})
        OPTIONAL MATCH (l)-[:ATTENDED]->(s:Session)
        RETURN l.id AS id, l.name AS name,
               l.native_language AS native_language,
               l.current_cefr_level AS current_cefr_level,
               count(s) AS session_count
        """,
        id=learner_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Learner not found")
    return LearnerSummary(**dict(rows[0]))


@router.get("/learners/{learner_id}/report", response_model=ProgressReport)
async def get_learner_report(learner_id: str, n: int = 5) -> ProgressReport:
    try:
        report = generate_report(learner_id, n_sessions=n)
        return ProgressReport(**report)
    except Exception as exc:
        logger.error("Failed to generate report: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate report")


@router.get("/learners/{learner_id}/sessions", response_model=List[SessionSummary])
async def list_learner_sessions(learner_id: str) -> List[SessionSummary]:
    rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $id})-[:ATTENDED]->(s:Session)
        RETURN s.id AS id, toString(s.date) AS date,
               s.duration_sec AS duration_sec,
               s.talk_time_ratio AS talk_time_ratio,
               s.words_per_turn AS words_per_turn,
               s.speaking_rate_wpm AS speaking_rate_wpm,
               s.code_switch_ratio AS code_switch_ratio,
               s.transcript_summary AS transcript_summary
        ORDER BY s.date DESC
        """,
        id=learner_id,
    )
    return [SessionSummary(**dict(r)) for r in rows]


@router.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    rows = neo4j_client.run_query(
        """
        MATCH (s:Session {id: $id})
        RETURN s.id AS id, toString(s.date) AS date,
               s.duration_sec AS duration_sec,
               s.talk_time_ratio AS talk_time_ratio,
               s.words_per_turn AS words_per_turn,
               s.speaking_rate_wpm AS speaking_rate_wpm,
               s.code_switch_ratio AS code_switch_ratio,
               s.transcript_summary AS transcript_summary
        """,
        id=session_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionSummary(**dict(rows[0]))


# ── Search ─────────────────────────────────────────────────────────────────────

@router.post("/search/errors", response_model=List[SearchResult])
async def search_errors(req: SemanticSearchRequest) -> List[SearchResult]:
    results = find_similar_errors(req.query, top_k=req.top_k)
    return [
        SearchResult(
            id=r["id"],
            score=r["score"],
            metadata={
                "pattern_name": r["pattern_name"],
                "description": r["description"],
            },
        )
        for r in results
    ]


@router.post("/search/sessions", response_model=List[SearchResult])
async def search_sessions(req: SemanticSearchRequest) -> List[SearchResult]:
    results = find_similar_sessions(req.query, top_k=req.top_k)
    return [
        SearchResult(
            id=r["id"],
            score=r["score"],
            metadata={
                "date": str(r["date"]),
                "learner_id": r["learner_id"],
                "transcript_summary": r["transcript_summary"],
            },
        )
        for r in results
    ]
