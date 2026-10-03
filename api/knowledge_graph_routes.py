"""
api/knowledge_graph_routes.py
──────────────────────────────
FastAPI routes for the Knowledge Graph module.

Handlers are plain `def` (not `async def`) on purpose: they make blocking
Neo4j / OpenAI calls, and FastAPI runs sync handlers in a thread pool so the
event loop stays free for job polling and health checks.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from api.models.kg_models import (
    CreateLearnerRequest,
    CreateTutorRequest,
    GraphEdge,
    GraphNode,
    IngestRequest,
    IngestResponse,
    IngestStatus,
    LearnerGraph,
    LearnerSummary,
    ProgressReport,
    SearchResult,
    SemanticSearchRequest,
    SessionSummary,
)
from knowledge_graph.db import neo4j_client
from knowledge_graph.ingest.orchestrator import ingest_session, reanalyze_session
from knowledge_graph.reports.generator import generate_report
from knowledge_graph.search import find_similar_errors, find_similar_sessions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["Knowledge Graph"])

# session_id -> {"status": ..., "error": ...}  (in-memory; single-process deployment)
_ingest_status: Dict[str, dict] = {}
_ingest_lock = threading.Lock()


def _set_ingest_status(session_id: str, status: str, error: str | None = None) -> None:
    with _ingest_lock:
        _ingest_status[session_id] = {"status": status, "error": error}


# ── Ingestion ─────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest_transcript(req: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Ingest a diarized transcript into the knowledge graph (runs in background)."""
    if not req.transcript:
        raise HTTPException(status_code=400, detail="transcript body is required")

    session_id = req.session_id or str(uuid.uuid4())
    _set_ingest_status(session_id, "pending")

    def _run_ingest() -> None:
        _set_ingest_status(session_id, "running")
        try:
            ingest_session(
                transcript_data=req.transcript,
                learner_id=req.learner_id,
                tutor_id=req.tutor_id,
                session_date=req.session_date,
                speaker_map=req.speaker_map,
                session_id=session_id,
            )
            _set_ingest_status(session_id, "done")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion failed for %s", session_id)
            _set_ingest_status(session_id, "error", f"{type(exc).__name__}: {exc}")

    background_tasks.add_task(_run_ingest)
    return IngestResponse(session_id=session_id)


@router.get("/ingest/{session_id}", response_model=IngestStatus)
def get_ingest_status(session_id: str) -> IngestStatus:
    with _ingest_lock:
        st = _ingest_status.get(session_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown ingestion id")
    return IngestStatus(session_id=session_id, **st)


@router.post("/sessions/{session_id}/reanalyze", status_code=202, response_model=IngestResponse)
def reanalyze(session_id: str, background_tasks: BackgroundTasks) -> IngestResponse:
    """Re-run the analysis of a stored session with the current prompts."""
    _set_ingest_status(session_id, "pending")

    def _run() -> None:
        _set_ingest_status(session_id, "running")
        try:
            reanalyze_session(session_id)
            _set_ingest_status(session_id, "done")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Re-analysis failed for %s", session_id)
            _set_ingest_status(session_id, "error", f"{type(exc).__name__}: {exc}")

    background_tasks.add_task(_run)
    return IngestResponse(session_id=session_id, message="Re-analysis started in background.")


# ── Learners & Tutors ─────────────────────────────────────────────────────────

@router.post("/learners")
def create_learner(req: CreateLearnerRequest) -> dict:
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
def create_tutor(req: CreateTutorRequest) -> dict:
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


_LEARNER_SUMMARY_RETURN = """
    RETURN l.id AS id, l.name AS name,
           l.native_language AS native_language,
           l.current_cefr_level AS current_cefr_level,
           count(s) AS session_count,
           toString(max(s.date)) AS last_session_date
"""


@router.get("/learners", response_model=List[LearnerSummary])
def list_learners() -> List[LearnerSummary]:
    rows = neo4j_client.run_query(
        "MATCH (l:Learner) OPTIONAL MATCH (l)-[:ATTENDED]->(s:Session)"
        + _LEARNER_SUMMARY_RETURN
        + " ORDER BY last_session_date DESC"
    )
    return [LearnerSummary(**dict(r)) for r in rows]


@router.get("/learners/{learner_id}", response_model=LearnerSummary)
def get_learner(learner_id: str) -> LearnerSummary:
    rows = neo4j_client.run_query(
        "MATCH (l:Learner {id: $id}) OPTIONAL MATCH (l)-[:ATTENDED]->(s:Session)"
        + _LEARNER_SUMMARY_RETURN,
        id=learner_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Learner not found")
    return LearnerSummary(**dict(rows[0]))


@router.get("/learners/{learner_id}/report", response_model=ProgressReport)
def get_learner_report(
    learner_id: str,
    n: int = Query(5, ge=1, le=20, description="Number of most recent sessions to include"),
    refresh: bool = Query(False, description="Regenerate instead of using the cached report"),
) -> ProgressReport:
    if not neo4j_client.run_query("MATCH (l:Learner {id: $id}) RETURN l.id", id=learner_id):
        raise HTTPException(status_code=404, detail="Learner not found")
    try:
        report = generate_report(learner_id, n_sessions=n, refresh=refresh)
        return ProgressReport(**report)
    except Exception:
        logger.exception("Failed to generate report for %s", learner_id)
        raise HTTPException(status_code=500, detail="Failed to generate report")


_SESSION_RETURN = """
    RETURN s.id AS id, toString(s.date) AS date,
           s.duration_sec AS duration_sec,
           s.talk_time_ratio AS talk_time_ratio,
           s.words_per_turn AS words_per_turn,
           s.average_words_per_sentence AS average_words_per_sentence,
           s.speaking_rate_wpm AS speaking_rate_wpm,
           s.code_switch_ratio AS code_switch_ratio,
           s.estimated_level AS estimated_level,
           coalesce(s.lesson_topics, []) AS lesson_topics,
           s.transcript_summary AS transcript_summary,
           s.source_url AS source_url
"""


@router.get("/learners/{learner_id}/sessions", response_model=List[SessionSummary])
def list_learner_sessions(learner_id: str) -> List[SessionSummary]:
    rows = neo4j_client.run_query(
        "MATCH (l:Learner {id: $id})-[:ATTENDED]->(s:Session)" + _SESSION_RETURN + " ORDER BY s.date DESC",
        id=learner_id,
    )
    return [SessionSummary(**dict(r)) for r in rows]


@router.get("/sessions/{session_id}", response_model=SessionSummary)
def get_session(session_id: str) -> SessionSummary:
    rows = neo4j_client.run_query("MATCH (s:Session {id: $id})" + _SESSION_RETURN, id=session_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionSummary(**dict(rows[0]))


# ── Graph (server-side; the browser never talks to Neo4j directly) ───────────

@router.get("/learners/{learner_id}/graph", response_model=LearnerGraph)
def get_learner_graph(
    learner_id: str,
    n: int = Query(5, ge=1, le=20),
    max_words: int = Query(30, ge=0, le=100),
) -> LearnerGraph:
    learner = neo4j_client.run_query(
        "MATCH (l:Learner {id: $id}) RETURN l.id AS id, l.name AS name", id=learner_id,
    )
    if not learner:
        raise HTTPException(status_code=404, detail="Learner not found")

    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    lid = f"learner:{learner_id}"
    nodes[lid] = GraphNode(id=lid, label=learner[0]["name"] or learner_id, group="learner")

    sessions = neo4j_client.run_query(
        """
        MATCH (:Learner {id: $id})-[:ATTENDED]->(s:Session)
        RETURN s.id AS id, toString(s.date) AS date, s.transcript_summary AS summary
        ORDER BY s.date DESC LIMIT $n
        """,
        id=learner_id, n=n,
    )
    sids = [s["id"] for s in sessions]
    for s in sessions:
        nid = f"session:{s['id']}"
        nodes[nid] = GraphNode(id=nid, label=s["date"] or "session", group="session", title=s["summary"])
        edges.append(GraphEdge(source=lid, target=nid, label="attended"))

    for r in neo4j_client.run_query(
        """
        MATCH (:Learner {id: $id})-[r:MADE_ERROR]->(gp:GrammarPattern)
        WHERE r.session_id IN $sids
        RETURN gp.id AS gid, coalesce(gp.label, gp.pattern_name) AS label,
               r.session_id AS sid, r.error_count AS count
        """,
        id=learner_id, sids=sids,
    ):
        gid = f"grammar:{r['gid']}"
        nodes.setdefault(gid, GraphNode(id=gid, label=r["label"] or r["gid"], group="grammar"))
        edges.append(GraphEdge(source=f"session:{r['sid']}", target=gid, label=f"{r['count']}×"))

    for r in neo4j_client.run_query(
        """
        MATCH (:Learner {id: $id})-[r:USED_WORD]->(w:Word)
        WHERE r.session_id IN $sids
        RETURN w.lemma AS word, w.cefr_level AS level, r.session_id AS sid
        ORDER BY r.date DESC LIMIT $max_words
        """,
        id=learner_id, sids=sids, max_words=max_words,
    ):
        wid = f"word:{r['word']}"
        nodes.setdefault(wid, GraphNode(id=wid, label=r["word"], group="word", title=r["level"]))
        edges.append(GraphEdge(source=f"session:{r['sid']}", target=wid, label="used"))

    for r in neo4j_client.run_query(
        """
        MATCH (:Learner {id: $id})-[r:HAS_GOAL]->(g:Goal)
        RETURN g.id AS gid, g.description AS description, r.status AS status
        LIMIT 10
        """,
        id=learner_id,
    ):
        gid = f"goal:{r['gid']}"
        nodes[gid] = GraphNode(id=gid, label=r["description"], group="goal", title=r["status"])
        edges.append(GraphEdge(source=lid, target=gid, label=r["status"]))

    return LearnerGraph(nodes=list(nodes.values()), edges=edges)


# ── Search ────────────────────────────────────────────────────────────────────

@router.post("/search/errors", response_model=List[SearchResult])
def search_errors(req: SemanticSearchRequest) -> List[SearchResult]:
    results = find_similar_errors(req.query, top_k=req.top_k)
    return [
        SearchResult(
            id=r["id"],
            score=r["score"],
            metadata={"pattern_name": r["pattern_name"], "description": r["description"]},
        )
        for r in results
    ]


@router.post("/search/sessions", response_model=List[SearchResult])
def search_sessions(req: SemanticSearchRequest) -> List[SearchResult]:
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
