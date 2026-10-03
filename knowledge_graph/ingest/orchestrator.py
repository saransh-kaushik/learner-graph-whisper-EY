"""
knowledge_graph/ingest/orchestrator.py
───────────────────────────────────────
Top-level ingestion pipeline. Ties together:
  parse → metrics → LLM extraction → embeddings → graph write

Usage (CLI)
-----------
    python -m knowledge_graph.ingest.orchestrator \\
        --transcript /path/to/session_stt.json \\
        --learner-id learner_001 \\
        --tutor-id   tutor_001 \\
        --date       2026-10-01

Usage (Python)
--------------
    from knowledge_graph.ingest.orchestrator import ingest_session
    session_id = ingest_session("session_stt.json", "learner_001", "tutor_001", "2026-10-01")
"""
from __future__ import annotations

import argparse
import logging
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, Optional

from knowledge_graph.ingest.graph_writer import write_session
from knowledge_graph.ingest.parser import ParsedTranscript, TranscriptParser
from knowledge_graph.llm.embeddings import embed_text
from knowledge_graph.llm.extractor import extract_from_transcript
from knowledge_graph.metrics import compute_session_metrics

logger = logging.getLogger(__name__)


def ingest_session(
    transcript_path: str | Path,
    learner_id: str,
    tutor_id: str,
    session_date: Optional[str] = None,
    speaker_map: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Full ingestion pipeline for a single session transcript.

    Parameters
    ----------
    transcript_path : path to session_stt.json
    learner_id      : Learner node id (must be pre-created or will be auto-created)
    tutor_id        : Tutor node id
    session_date    : ISO-8601 date string, defaults to today
    speaker_map     : Optional {"SPEAKER_00": "learner", "SPEAKER_01": "tutor"}
    session_id      : Override session UUID (auto-generated if omitted)

    Returns
    -------
    session_id : The UUID string of the created Session node
    """
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
    transcript: ParsedTranscript = parser.parse_file(transcript_path)

    # ── Step 2: Session metrics ────────────────────────────────────────────────
    metrics = compute_session_metrics(transcript)
    logger.info(
        "Metrics — talk_ratio=%.2f wpm=%.1f code_switch=%.3f",
        metrics.talk_time_ratio,
        metrics.speaking_rate_wpm,
        metrics.code_switch_ratio,
    )

    # ── Step 3: Build formatted transcript for LLM ─────────────────────────────
    formatted_transcript = _format_transcript_for_llm(transcript)

    # ── Step 4: OpenAI NLP extraction ─────────────────────────────────────────
    extraction = extract_from_transcript(
        formatted_transcript,
        learner_label=transcript.learner_label or "Learner",
        tutor_label=transcript.tutor_label or "Tutor",
    )
    logger.info(
        "Extraction — grammar_errors=%d vocab=%d goals=%d skills=%d",
        len(extraction.grammar_errors),
        len(extraction.vocabulary),
        len(extraction.goals_discussed),
        len(extraction.skills_covered),
    )

    # ── Step 5: Embed session summary ─────────────────────────────────────────
    embed_text_input = extraction.session_summary or formatted_transcript[:2000]
    session_embedding = embed_text(embed_text_input)

    # ── Step 6: Write to Neo4j ────────────────────────────────────────────────
    write_session(
        session_id=session_id,
        learner_id=learner_id,
        tutor_id=tutor_id,
        session_date=session_date,
        transcript=transcript,
        metrics=metrics,
        extraction=extraction,
        session_embedding=session_embedding,
    )

    return session_id


def _format_transcript_for_llm(transcript: ParsedTranscript) -> str:
    """Format turns into a readable string for the LLM prompt."""
    lines = []
    for turn in transcript.turns:
        role_label = turn.role.upper() if turn.role != "unknown" else turn.speaker_label
        lines.append(f"{role_label}: {turn.text}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest a session transcript into the Knowledge Graph")
    parser.add_argument("--transcript",  required=True, help="Path to session_stt.json")
    parser.add_argument("--learner-id",  required=True, dest="learner_id")
    parser.add_argument("--tutor-id",    required=True, dest="tutor_id")
    parser.add_argument("--date",        default=None,  help="ISO-8601 date (default: today)")
    parser.add_argument("--session-id",  default=None,  dest="session_id")
    args = parser.parse_args()

    sid = ingest_session(
        transcript_path=args.transcript,
        learner_id=args.learner_id,
        tutor_id=args.tutor_id,
        session_date=args.date,
        session_id=args.session_id,
    )
    print(f"✅  Session ingested: {sid}")


if __name__ == "__main__":
    _cli()
