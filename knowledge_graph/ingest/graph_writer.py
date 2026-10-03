"""
knowledge_graph/ingest/graph_writer.py
───────────────────────────────────────
Upserts (MERGE) all nodes and relationships into Neo4j after a session
is parsed and analysed.

All writes are done inside a single transaction for atomicity.
"""
from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any, List

from knowledge_graph.db import neo4j_client
from knowledge_graph.ingest.parser import ParsedTranscript
from knowledge_graph.llm.extractor import ExtractionResult, GrammarError, VocabItem
from knowledge_graph.llm.cefr import lookup_cefr_batch
from knowledge_graph.metrics import SessionMetrics

logger = logging.getLogger(__name__)


# ── Public entry point ─────────────────────────────────────────────────────────

def write_session(
    *,
    session_id: str,
    learner_id: str,
    tutor_id: str,
    session_date: str,          # ISO-8601, e.g. "2026-10-01"
    transcript: ParsedTranscript,
    metrics: SessionMetrics,
    extraction: ExtractionResult,
    session_embedding: List[float],
) -> None:
    """Write the full session graph to Neo4j (all nodes + relationships)."""

    with neo4j_client.session() as s:
        s.execute_write(
            _write_all,
            session_id=session_id,
            learner_id=learner_id,
            tutor_id=tutor_id,
            session_date=session_date,
            transcript=transcript,
            metrics=metrics,
            extraction=extraction,
            session_embedding=session_embedding,
        )
    logger.info("Graph write complete for session %s", session_id)


# ── Transaction function ───────────────────────────────────────────────────────

def _write_all(tx: Any, **kwargs: Any) -> None:
    session_id: str = kwargs["session_id"]
    learner_id: str = kwargs["learner_id"]
    tutor_id: str = kwargs["tutor_id"]
    session_date: str = kwargs["session_date"]
    metrics: SessionMetrics = kwargs["metrics"]
    extraction: ExtractionResult = kwargs["extraction"]
    session_embedding: List[float] = kwargs["session_embedding"]

    # 1. Ensure Learner node exists
    tx.run(
        "MERGE (l:Learner {id: $id}) ON CREATE SET l.created_at = date()",
        id=learner_id,
    )

    # 2. Ensure Tutor node exists
    tx.run(
        "MERGE (t:Tutor {id: $id}) ON CREATE SET t.created_at = date()",
        id=tutor_id,
    )

    # 3. Create Session node
    tx.run(
        """
        MERGE (s:Session {id: $id})
        SET s.learner_id          = $learner_id,
            s.tutor_id            = $tutor_id,
            s.date                = date($date),
            s.duration_sec        = $duration_sec,
            s.talk_time_ratio     = $talk_time_ratio,
            s.words_per_turn      = $words_per_turn,
            s.speaking_rate_wpm   = $speaking_rate_wpm,
            s.code_switch_ratio   = $code_switch_ratio,
            s.transcript_summary  = $summary,
            s.embedding           = $embedding
        """,
        id=session_id,
        learner_id=learner_id,
        tutor_id=tutor_id,
        date=session_date,
        duration_sec=metrics.total_duration_sec,
        talk_time_ratio=metrics.talk_time_ratio,
        words_per_turn=metrics.words_per_turn,
        speaking_rate_wpm=metrics.speaking_rate_wpm,
        code_switch_ratio=metrics.code_switch_ratio,
        summary=extraction.session_summary,
        embedding=session_embedding,
    )

    # 4. Relationships: Learner ATTENDED Session + Tutor CONDUCTED Session
    tx.run(
        """
        MATCH (l:Learner {id: $lid}), (s:Session {id: $sid})
        MERGE (l)-[:ATTENDED {role: 'learner'}]->(s)
        """,
        lid=learner_id, sid=session_id,
    )
    tx.run(
        """
        MATCH (t:Tutor {id: $tid}), (s:Session {id: $sid})
        MERGE (t)-[:CONDUCTED {role: 'tutor'}]->(s)
        """,
        tid=tutor_id, sid=session_id,
    )

    # 5. Grammar errors
    _write_grammar_errors(tx, learner_id, session_id, session_date, extraction.grammar_errors)

    # 6. Vocabulary
    _write_vocabulary(tx, learner_id, session_id, session_date, extraction.vocabulary)

    # 7. Goals
    _write_goals(tx, learner_id, extraction.goals_discussed)

    # 8. Skills
    _write_skills(tx, session_id, extraction.skills_covered)


def _write_grammar_errors(
    tx: Any, learner_id: str, session_id: str, session_date: str,
    errors: list,
) -> None:
    for err in errors:
        gp_id = f"gp_{err.pattern}"
        tx.run(
            """
            MERGE (gp:GrammarPattern {id: $id})
            ON CREATE SET
                gp.pattern_name   = $pattern,
                gp.description    = $description,
                gp.example_wrong  = $wrong,
                gp.example_correct = $correct
            """,
            id=gp_id,
            pattern=err.pattern,
            description=err.description,
            wrong=err.example_wrong,
            correct=err.example_correct,
        )
        tx.run(
            """
            MATCH (l:Learner {id: $lid}), (gp:GrammarPattern {id: $gpid})
            MERGE (l)-[r:MADE_ERROR {session_id: $sid}]->(gp)
            SET r.error_count       = $count,
                r.example_utterance = $wrong,
                r.date              = date($date)
            """,
            lid=learner_id,
            gpid=gp_id,
            sid=session_id,
            count=err.count,
            wrong=err.example_wrong,
            date=session_date,
        )


def _write_vocabulary(
    tx: Any, learner_id: str, session_id: str, session_date: str,
    vocab: list,
) -> None:
    if not vocab:
        return

    # Batch CEFR lookup
    words = [item.word for item in vocab]
    cefr_map = lookup_cefr_batch(words)

    for item in vocab:
        level = cefr_map.get(item.word, "B1")

        # Determine mastery
        if item.spontaneous:
            mastery = "spontaneous"
        elif item.used_by_learner:
            mastery = "prompted"
        else:
            mastery = "introduced"

        # Upsert Word node (Session intro)
        tx.run(
            """
            MERGE (w:Word {lemma: $lemma})
            ON CREATE SET w.cefr_level = $level, w.pos = 'unknown'
            """,
            lemma=item.word,
            level=level,
        )

        # Session introduced word
        tx.run(
            """
            MATCH (s:Session {id: $sid}), (w:Word {lemma: $lemma})
            MERGE (s)-[r:INTRODUCED_WORD]->(w)
            SET r.context_sentence = $ctx
            """,
            sid=session_id,
            lemma=item.word,
            ctx=item.context_sentence,
        )

        # Learner used word
        if item.used_by_learner:
            tx.run(
                """
                MATCH (l:Learner {id: $lid}), (w:Word {lemma: $lemma})
                MERGE (l)-[r:USED_WORD {session_id: $sid}]->(w)
                SET r.correct     = $correct,
                    r.spontaneous = $spontaneous,
                    r.mastery     = $mastery,
                    r.date        = date($date)
                """,
                lid=learner_id,
                lemma=item.word,
                sid=session_id,
                correct=item.correct,
                spontaneous=item.spontaneous,
                mastery=mastery,
                date=session_date,
            )


def _write_goals(tx: Any, learner_id: str, goals: list) -> None:
    for goal in goals:
        goal_id = f"goal_{uuid.uuid5(uuid.NAMESPACE_DNS, goal.description)}"
        tx.run(
            """
            MERGE (g:Goal {id: $id})
            ON CREATE SET g.description = $desc,
                          g.created_at  = date()
            """,
            id=goal_id,
            desc=goal.description,
        )
        tx.run(
            """
            MATCH (l:Learner {id: $lid}), (g:Goal {id: $gid})
            MERGE (l)-[r:HAS_GOAL]->(g)
            SET r.status     = $status,
                r.updated_at = date()
            """,
            lid=learner_id,
            gid=goal_id,
            status=goal.status,
        )


def _write_skills(tx: Any, session_id: str, skills: list) -> None:
    for skill in skills:
        tx.run(
            """
            MERGE (sk:Skill {name: $name})
            ON CREATE SET sk.category = $cat
            """,
            name=skill.name,
            cat=skill.category,
        )
        tx.run(
            """
            MATCH (s:Session {id: $sid}), (sk:Skill {name: $name})
            MERGE (s)-[:COVERED_SKILL]->(sk)
            """,
            sid=session_id,
            name=skill.name,
        )
