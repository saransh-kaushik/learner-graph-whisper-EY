"""
knowledge_graph/ingest/orchestrator.py
───────────────────────────────────────
Top-level ingestion pipeline. Ties together:
  parse → role check → metrics → LLM extraction → embeddings → graph write

Usage (CLI)
-----------
    python -m knowledge_graph.ingest.orchestrator \\
        --transcript /path/to/session_stt.json \\
        --learner-id learner_001 \\
        --tutor-id   tutor_001 \\
        --date       2026-10-01

    # Re-run the analysis of already-ingested sessions (e.g. after a prompt change)
    python -m knowledge_graph.ingest.orchestrator --reanalyze-learner learner_001

Usage (Python)
--------------
    from knowledge_graph.ingest.orchestrator import ingest_session
    session_id = ingest_session("session_stt.json", "learner_001", "tutor_001", "2026-10-01")
"""
from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from knowledge_graph.db import neo4j_client
from knowledge_graph.ingest.graph_writer import write_session
from knowledge_graph.ingest.parser import ParsedTranscript, TranscriptParser
from knowledge_graph.llm.embeddings import embed_batch
from knowledge_graph.llm.extractor import (
    ExtractionError,
    extract_from_transcript,
    identify_learner_label,
)
from knowledge_graph.metrics import compute_session_metrics
from knowledge_graph.taxonomy import GRAMMAR_CATEGORIES

logger = logging.getLogger(__name__)


def ingest_session(
    transcript_path: str | Path | None = None,
    learner_id: str = "",
    tutor_id: str = "",
    session_date: Optional[str] = None,
    speaker_map: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None,
    *,
    transcript_data: Optional[dict | list] = None,
    source_url: Optional[str] = None,
) -> str:
    """
    Full ingestion pipeline for a single session transcript.

    Parameters
    ----------
    transcript_path : path to session_stt.json (or pass transcript_data)
    learner_id      : Learner node id (auto-created if missing)
    tutor_id        : Tutor node id
    session_date    : ISO-8601 date string, defaults to today
    speaker_map     : Optional {"SPEAKER_00": "learner", "SPEAKER_01": "tutor"}
    session_id      : Session id; re-using an id replaces that session's analysis
    transcript_data : Already-loaded pipeline output (avoids temp files)
    source_url      : Recording URL, stored for traceability

    Returns
    -------
    session_id : The id of the created/updated Session node

    Raises
    ------
    ExtractionError if the transcript cannot be analysed (nothing is written).
    """
    if not learner_id or not tutor_id:
        raise ValueError("learner_id and tutor_id are required")
    if session_date is None:
        session_date = date.today().isoformat()
    if session_id is None:
        session_id = str(uuid.uuid4())

    logger.info(
        "Ingesting session %s | learner=%s tutor=%s date=%s",
        session_id, learner_id, tutor_id, session_date,
    )

    # ── Step 1: Parse transcript ───────────────────────────────────────────────
    parser = TranscriptParser(speaker_map=speaker_map)
    if transcript_data is not None:
        transcript = parser.parse_dict(transcript_data, session_path=source_url or "")
    elif transcript_path is not None:
        transcript = parser.parse_file(transcript_path)
    else:
        raise ValueError("Provide transcript_path or transcript_data")

    if not speaker_map:
        _confirm_roles(transcript)

    _analyse_and_write(
        transcript,
        session_id=session_id,
        learner_id=learner_id,
        tutor_id=tutor_id,
        session_date=session_date,
        source_url=source_url,
    )
    return session_id


def _confirm_roles(transcript: ParsedTranscript) -> None:
    """The talk-time heuristic mislabels talkative learners; ask the model
    which speaker is the tutor and swap roles if needed."""
    labels = transcript.speaker_labels
    if len(labels) != 2:
        return
    learner = identify_learner_label(
        [(t.speaker_label, t.text) for t in transcript.turns], labels,
    )
    if learner and learner != transcript.learner_label:
        tutor = next(l for l in labels if l != learner)
        logger.info("Swapping speaker roles: learner=%s tutor=%s", learner, tutor)
        transcript.assign_roles(learner, tutor)


def _analyse_and_write(
    transcript: ParsedTranscript,
    *,
    session_id: str,
    learner_id: str,
    tutor_id: str,
    session_date: str,
    source_url: Optional[str],
) -> None:
    if not transcript.learner_turns:
        raise ExtractionError("No learner speech found in transcript (check speaker roles)")

    # ── Step 2: Session metrics ────────────────────────────────────────────────
    metrics = compute_session_metrics(transcript)
    logger.info(
        "Metrics — talk_ratio=%.2f wpm=%.1f code_switch=%.3f",
        metrics.talk_time_ratio, metrics.speaking_rate_wpm, metrics.code_switch_ratio,
    )

    # ── Step 3: LLM extraction (raises on failure — nothing is written) ───────
    extraction = extract_from_transcript(format_transcript_for_llm(transcript))
    logger.info(
        "Extraction — grammar_errors=%d vocab=%d goals=%d strengths=%d",
        len(extraction.grammar_errors), len(extraction.vocabulary),
        len(extraction.learner_goals), len(extraction.strengths),
    )

    # ── Step 4: Embeddings (session summary, goals, grammar categories) ──────
    goal_texts = [g.description for g in extraction.learner_goals]
    categories = sorted({e.category for e in extraction.grammar_errors})
    texts = [extraction.session_summary or format_transcript_for_llm(transcript)[:2000]]
    texts += goal_texts
    texts += [f"{GRAMMAR_CATEGORIES[c][0]}: {GRAMMAR_CATEGORIES[c][1]}" for c in categories]
    vectors = embed_batch(texts)
    session_embedding = vectors[0]
    goal_embeddings = dict(zip(goal_texts, vectors[1:1 + len(goal_texts)]))
    grammar_embeddings = dict(zip(categories, vectors[1 + len(goal_texts):]))

    # ── Step 5: Write to Neo4j ────────────────────────────────────────────────
    write_session(
        session_id=session_id,
        learner_id=learner_id,
        tutor_id=tutor_id,
        session_date=session_date,
        transcript=transcript,
        metrics=metrics,
        extraction=extraction,
        session_embedding=session_embedding,
        goal_embeddings=goal_embeddings,
        grammar_embeddings=grammar_embeddings,
        source_url=source_url,
    )


def format_transcript_for_llm(transcript: ParsedTranscript) -> str:
    """Format turns as 'LEARNER: …' / 'TUTOR: …' lines, merging consecutive
    turns by the same speaker so sentences split by ASR stay together."""
    lines: List[str] = []
    prev_label = None
    for turn in transcript.turns:
        label = turn.role.upper() if turn.role != "unknown" else (turn.speaker_label or "UNKNOWN")
        if label == prev_label and lines:
            lines[-1] = f"{lines[-1]} {turn.text}"
        else:
            lines.append(f"{label}: {turn.text}")
        prev_label = label
    return "\n".join(lines)


# ── Re-analysis of stored sessions ────────────────────────────────────────────

def reanalyze_session(session_id: str) -> None:
    """Re-run extraction + metrics on a stored session (no re-transcription)."""
    rows = neo4j_client.run_query(
        """
        MATCH (s:Session {id: $id})
        RETURN s.learner_id AS learner_id, s.tutor_id AS tutor_id,
               toString(s.date) AS date, s.source_url AS source_url,
               s.transcript_turns_json AS turns
        """,
        id=session_id,
    )
    if not rows:
        raise ValueError(f"Session {session_id} not found")
    row = rows[0]
    if not row["turns"]:
        raise ValueError(
            f"Session {session_id} was ingested before transcripts were stored; "
            "re-submit its recording instead."
        )
    turns = json.loads(row["turns"])
    roles = {t["speaker"]: t["role"] for t in turns if t.get("role") in ("learner", "tutor")}
    transcript = TranscriptParser(speaker_map=roles or None).parse_dict(
        {"segments": turns}, session_path=row["source_url"] or "",
    )
    if not roles:
        _confirm_roles(transcript)
    _analyse_and_write(
        transcript,
        session_id=session_id,
        learner_id=row["learner_id"],
        tutor_id=row["tutor_id"],
        session_date=row["date"],
        source_url=row["source_url"],
    )


def reanalyze_learner(learner_id: str) -> Dict[str, str]:
    """Re-analyse every stored session of a learner. Returns {session_id: outcome}."""
    rows = neo4j_client.run_query(
        "MATCH (:Learner {id: $id})-[:ATTENDED]->(s:Session) RETURN s.id AS id ORDER BY s.date",
        id=learner_id,
    )
    outcomes: Dict[str, str] = {}
    for r in rows:
        try:
            reanalyze_session(r["id"])
            outcomes[r["id"]] = "ok"
        except Exception as exc:  # noqa: BLE001 — report and continue
            logger.error("Re-analysis failed for %s: %s", r["id"], exc)
            outcomes[r["id"]] = f"error: {exc}"
    return outcomes


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest a session transcript into the Knowledge Graph")
    parser.add_argument("--transcript", help="Path to session_stt.json")
    parser.add_argument("--learner-id", dest="learner_id")
    parser.add_argument("--tutor-id", dest="tutor_id")
    parser.add_argument("--date", default=None, help="ISO-8601 date (default: today)")
    parser.add_argument("--session-id", default=None, dest="session_id")
    parser.add_argument("--reanalyze-session", dest="reanalyze_session_id")
    parser.add_argument("--reanalyze-learner", dest="reanalyze_learner_id")
    args = parser.parse_args()

    if args.reanalyze_session_id:
        reanalyze_session(args.reanalyze_session_id)
        print(f"Re-analysed session {args.reanalyze_session_id}")
        return
    if args.reanalyze_learner_id:
        for sid, outcome in reanalyze_learner(args.reanalyze_learner_id).items():
            print(f"{sid}: {outcome}")
        return

    if not (args.transcript and args.learner_id and args.tutor_id):
        parser.error("--transcript, --learner-id and --tutor-id are required for ingestion")
    sid = ingest_session(
        transcript_path=args.transcript,
        learner_id=args.learner_id,
        tutor_id=args.tutor_id,
        session_date=args.date,
        session_id=args.session_id,
    )
    print(f"Session ingested: {sid}")


if __name__ == "__main__":
    _cli()
