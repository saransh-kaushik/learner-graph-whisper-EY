"""
api/models/kg_models.py
────────────────────────
Pydantic request / response models for the Knowledge Graph API routes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    learner_id: str = Field(..., description="ID of the learner node")
    tutor_id: str = Field(..., description="ID of the tutor node")
    session_date: Optional[str] = Field(
        None, description="ISO-8601 date, e.g. '2026-10-01'. Defaults to today."
    )
    session_id: Optional[str] = Field(
        None, description="Custom session UUID. Auto-generated if omitted."
    )
    speaker_map: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Map raw speaker labels to roles. "
            "E.g. {\"SPEAKER_00\": \"learner\", \"SPEAKER_01\": \"tutor\"}"
        ),
    )
    transcript: Optional[dict] = Field(
        None,
        description=(
            "Transcript JSON body (alternative to uploading a file). "
            "Must match the session_stt.json format."
        ),
    )


class CreateLearnerRequest(BaseModel):
    id: str = Field(..., description="Unique learner ID")
    name: str
    native_language: Optional[str] = None
    current_cefr_level: Optional[str] = Field(None, pattern=r"^(A1|A2|B1|B2|C1|C2)$")


class CreateTutorRequest(BaseModel):
    id: str = Field(..., description="Unique tutor ID")
    name: str
    specialization: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)


# ── Response models ────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    session_id: str
    message: str = "Session ingested successfully."


class LearnerSummary(BaseModel):
    id: str
    name: Optional[str] = None
    native_language: Optional[str] = None
    current_cefr_level: Optional[str] = None
    session_count: int = 0


class SessionSummary(BaseModel):
    id: str
    date: Optional[str] = None
    duration_sec: Optional[float] = None
    talk_time_ratio: Optional[float] = None
    words_per_turn: Optional[float] = None
    speaking_rate_wpm: Optional[float] = None
    code_switch_ratio: Optional[float] = None
    transcript_summary: Optional[str] = None


class MetricPoint(BaseModel):
    current: float
    baseline: float
    trend: str      # improving | stable | declining
    history: List[float] = []


class VocabReport(BaseModel):
    new_words: List[str] = []
    mastery_upgrades: List[Dict[str, str]] = []


class GrammarReport(BaseModel):
    fading_errors: List[Dict[str, Any]] = []
    persistent_errors: List[Dict[str, Any]] = []


class GoalStatus(BaseModel):
    description: str
    status: str     # mentioned | practiced | demonstrated


class ProgressReport(BaseModel):
    learner_id: str
    learner_name: Optional[str] = None
    cefr_level: Optional[str] = None
    report_date: str
    sessions_analyzed: int
    metrics: Dict[str, MetricPoint]
    vocabulary: VocabReport
    grammar: GrammarReport
    goals: List[GoalStatus]
    llm_summary: str
    practice_next: str


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any] = {}
