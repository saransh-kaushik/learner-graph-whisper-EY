"""
api/models/kg_models.py
────────────────────────
Pydantic request / response models for the Knowledge Graph API routes.

The ProgressReport models mirror knowledge_graph.reports.generator output;
tests/test_report.py validates real generator output against them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Request models ─────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    learner_id: str = Field(..., description="ID of the learner node")
    tutor_id: str = Field(..., description="ID of the tutor node")
    session_date: Optional[str] = Field(
        None, description="ISO-8601 date, e.g. '2026-10-01'. Defaults to today."
    )
    session_id: Optional[str] = Field(
        None, description="Session id. Re-using an id replaces that session's analysis."
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
        description="Transcript JSON body. Must match the session_stt.json format.",
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
    message: str = "Ingestion started in background."


class IngestStatus(BaseModel):
    session_id: str
    status: str            # pending | running | done | error
    error: Optional[str] = None


class LearnerSummary(BaseModel):
    id: str
    name: Optional[str] = None
    native_language: Optional[str] = None
    current_cefr_level: Optional[str] = None
    session_count: int = 0
    last_session_date: Optional[str] = None


class SessionSummary(BaseModel):
    id: str
    date: Optional[str] = None
    duration_sec: Optional[float] = None
    talk_time_ratio: Optional[float] = None
    words_per_turn: Optional[float] = None
    average_words_per_sentence: Optional[float] = None
    speaking_rate_wpm: Optional[float] = None
    code_switch_ratio: Optional[float] = None
    estimated_level: Optional[str] = None
    lesson_topics: List[str] = []
    transcript_summary: Optional[str] = None
    source_url: Optional[str] = None


# ── Progress report ───────────────────────────────────────────────────────────

class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CefrLevel(_Model):
    value: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None          # tutor | estimated
    confidence: str = "insufficient_data"


class Period(_Model):
    start: Optional[str] = None
    end: Optional[str] = None
    learner_minutes: float = 0
    total_minutes: float = 0
    learner_words: Optional[int] = None


class SessionsAnalyzed(_Model):
    valid: int
    total: int


class DataQualityIssue(_Model):
    session_id: str
    date: Optional[str] = None
    reasons: List[str] = []


class HistoryPoint(_Model):
    session_id: str
    date: Optional[str] = None
    value: Optional[float] = None


class MetricPoint(_Model):
    label: str
    help: str = ""
    unit: str = ""
    better: str = "higher"
    current: Optional[float] = None
    baseline: Optional[float] = None
    recent: Optional[float] = None
    change_pct: Optional[float] = None
    trend: str = "insufficient_data"   # improving | stable | declining | insufficient_data
    history: List[HistoryPoint] = []


class GrammarExample(_Model):
    wrong: str
    correct: Optional[str] = None
    explanation: Optional[str] = None
    date: Optional[str] = None


class SeriesPoint(_Model):
    date: Optional[str] = None
    value: float


class GrammarPatternReport(_Model):
    pattern: str
    label: str
    status: str            # new | resolved | fading | increasing | persistent
    unit: str
    series: List[SeriesPoint] = []
    total_count: int = 0
    sessions_with_error: int = 0
    latest_count: int = 0
    recent_rate: float = 0
    earlier_rate: Optional[float] = None
    examples: List[GrammarExample] = []


class GrammarReport(_Model):
    improved: List[GrammarPatternReport] = []
    needs_work: List[GrammarPatternReport] = []
    new: List[GrammarPatternReport] = []


class WordEntry(_Model):
    word: str
    kind: str = "word"
    level: Optional[str] = None
    level_hint: Optional[str] = None
    meaning: Optional[str] = None
    example: Optional[str] = None
    your_sentence: Optional[str] = None
    tutor_sentence: Optional[str] = None
    correction: Optional[str] = None
    mastery: Optional[str] = None
    date: Optional[str] = None
    times_used: Optional[int] = None


class VocabReport(_Model):
    new_words: List[WordEntry] = []
    now_independent: List[WordEntry] = []
    words_to_fix: List[WordEntry] = []
    words_to_try: List[WordEntry] = []
    totals: Dict[str, int] = {}


class Strength(_Model):
    area: str
    quote: str
    note: str = ""
    date: Optional[str] = None


class GoalStatus(_Model):
    description: str
    status: str            # mentioned | practiced | demonstrated
    date: Optional[str] = None
    first_seen: Optional[str] = None


class LessonTopic(_Model):
    topic: str
    date: Optional[str] = None


class SessionTimelineItem(_Model):
    session_id: str
    date: Optional[str] = None
    is_valid: bool = True
    duration_min: float = 0
    learner_minutes: Optional[float] = None
    summary: str = ""
    topics: List[str] = []


class Win(_Model):
    title: str
    detail: str = ""
    evidence: str = ""


class FocusArea(_Model):
    title: str
    why_it_matters: str = ""
    your_sentence: str = ""
    better_sentence: str = ""
    tip: str = ""


class PracticeTask(_Model):
    title: str
    minutes: int = 5
    situation: str = ""
    steps: List[str] = []


class Narrative(_Model):
    headline: str = ""
    summary: str = ""
    wins: List[Win] = []
    focus_areas: List[FocusArea] = []
    practice_plan: List[PracticeTask] = []
    next_milestone: str = ""
    source: str = "ai"     # ai | fallback | empty


class ProgressReport(_Model):
    learner_id: str
    learner_name: Optional[str] = None
    native_language: Optional[str] = None
    cefr_level: CefrLevel
    report_date: str
    generated_at: Optional[str] = None
    period: Period
    sessions_analyzed: SessionsAnalyzed
    data_quality: List[DataQualityIssue] = []
    metrics: Dict[str, MetricPoint] = {}
    vocabulary: VocabReport
    grammar: GrammarReport
    top_priority_errors: List[GrammarPatternReport] = []
    strengths: List[Strength] = []
    learner_goals: List[GoalStatus] = []
    lesson_topics: List[LessonTopic] = []
    sessions: List[SessionTimelineItem] = []
    narrative: Narrative


# ── Graph & search ────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    label: str
    group: str
    title: Optional[str] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    label: Optional[str] = None


class LearnerGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any] = {}
