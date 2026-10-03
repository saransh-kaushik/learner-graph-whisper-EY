"""
knowledge_graph/ingest/graph_writer.py
───────────────────────────────────────
Upserts (MERGE) all nodes and relationships into Neo4j after a session
is parsed and analysed.

All writes are done inside a single transaction for atomicity. Network calls
(CEFR lookup) happen *before* the transaction so retries never repeat them.
Re-ingesting the same session id replaces that session's previous analysis.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from knowledge_graph.db import neo4j_client
from knowledge_graph.ingest.parser import ParsedTranscript
from knowledge_graph.llm.cefr import lookup_cefr_batch
from knowledge_graph.llm.extractor import ExtractionResult, GrammarError, VocabItem
from knowledge_graph.metrics import SessionMetrics
from knowledge_graph.taxonomy import GRAMMAR_CATEGORIES

logger = logging.getLogger(__name__)

_GOAL_RANK = {"mentioned": 1, "practiced": 2, "demonstrated": 3}
_MASTERY_RANK = {"repeated": 1, "prompted": 2, "spontaneous": 3}


def goal_id_for(description: str) -> str:
    """Stable id for a goal, tolerant of case/punctuation differences."""
    norm = re.sub(r"[^a-z0-9 ]+", "", description.lower())
    norm = " ".join(norm.split())
    return f"goal_{uuid.uuid5(uuid.NAMESPACE_DNS, norm)}"


@dataclass
class _GrammarGroup:
    category: str
    count: int = 0
    description: str = ""
    examples: List[dict] = field(default_factory=list)


def _group_grammar(errors: List[GrammarError]) -> List[_GrammarGroup]:
    """One MADE_ERROR edge per (learner, category, session): merge items that
    share a category instead of letting later items overwrite earlier ones."""
    groups: Dict[str, _GrammarGroup] = {}
    for err in errors:
        g = groups.setdefault(err.category, _GrammarGroup(category=err.category))
        g.count += max(1, err.count)
        if not g.description:
            g.description = err.description
        if len(g.examples) < 4:
            g.examples.append({
                "wrong": err.example_wrong,
                "correct": err.example_correct,
                "explanation": err.explanation,
                "description": err.description,
            })
    return list(groups.values())


def _dedupe_vocab(vocab: List[VocabItem]) -> List[VocabItem]:
    """Keep one item per term, preferring the strongest evidence of use."""
    best: Dict[str, VocabItem] = {}
    for item in vocab:
        cur = best.get(item.term)
        if cur is None or _MASTERY_RANK.get(item.mastery or "", 0) > _MASTERY_RANK.get(cur.mastery or "", 0):
            best[item.term] = item
    return list(best.values())


def _compact_transcript(transcript: ParsedTranscript) -> str:
    """Store turns (without word-level timings) so sessions can be re-analysed
    later without re-transcribing the audio."""
    return json.dumps([
        {"speaker": t.speaker_label, "role": t.role,
         "start": round(t.start, 2), "end": round(t.end, 2), "text": t.text}
        for t in transcript.turns
    ], ensure_ascii=False)


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
    goal_embeddings: Optional[Dict[str, List[float]]] = None,
    grammar_embeddings: Optional[Dict[str, List[float]]] = None,
    source_url: Optional[str] = None,
) -> None:
    """Write the full session graph to Neo4j (all nodes + relationships)."""

    vocab = _dedupe_vocab(extraction.vocabulary)
    cefr_map = lookup_cefr_batch(
        [v.term for v in vocab],
        hints={v.term: v.level for v in vocab},
    ) if vocab else {}

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
            vocab=vocab,
            cefr_map=cefr_map,
            session_embedding=session_embedding,
            goal_embeddings=goal_embeddings or {},
            grammar_embeddings=grammar_embeddings or {},
            source_url=source_url,
        )
    logger.info("Graph write complete for session %s", session_id)


# ── Transaction function ───────────────────────────────────────────────────────

def _write_all(tx: Any, **kwargs: Any) -> None:
    session_id: str = kwargs["session_id"]
    learner_id: str = kwargs["learner_id"]
    tutor_id: str = kwargs["tutor_id"]
    session_date: str = kwargs["session_date"]
    transcript: ParsedTranscript = kwargs["transcript"]
    metrics: SessionMetrics = kwargs["metrics"]
    extraction: ExtractionResult = kwargs["extraction"]

    # 0. Remove any previous analysis of this session (idempotent re-ingest)
    _clear_session(tx, session_id)

    # 1. Ensure Learner / Tutor nodes exist
    tx.run("MERGE (l:Learner {id: $id}) ON CREATE SET l.created_at = date()", id=learner_id)
    tx.run("MERGE (t:Tutor {id: $id}) ON CREATE SET t.created_at = date()", id=tutor_id)

    # 2. Session node
    tx.run(
        """
        MERGE (s:Session {id: $id})
        ON CREATE SET s.ingested_at = datetime()
        SET s.learner_id                 = $learner_id,
            s.tutor_id                   = $tutor_id,
            s.date                       = date($date),
            s.reanalyzed_at              = datetime(),
            s.source_url                 = $source_url,
            s.duration_sec               = $duration_sec,
            s.talk_time_ratio            = $talk_time_ratio,
            s.words_per_turn             = $words_per_turn,
            s.average_words_per_sentence = $awps,
            s.speaking_rate_wpm          = $speaking_rate_wpm,
            s.code_switch_ratio          = $code_switch_ratio,
            s.learner_talk_sec           = $learner_talk_sec,
            s.learner_turn_count         = $learner_turn_count,
            s.total_learner_words        = $total_learner_words,
            s.estimated_level            = $estimated_level,
            s.lesson_topics              = $lesson_topics,
            s.strengths_json             = $strengths_json,
            s.transcript_summary         = $summary,
            s.transcript_turns_json      = $turns_json,
            s.embedding                  = $embedding
        """,
        id=session_id,
        learner_id=learner_id,
        tutor_id=tutor_id,
        date=session_date,
        source_url=kwargs.get("source_url"),
        duration_sec=metrics.total_duration_sec,
        talk_time_ratio=metrics.talk_time_ratio,
        words_per_turn=metrics.words_per_turn,
        awps=metrics.average_words_per_sentence,
        speaking_rate_wpm=metrics.speaking_rate_wpm,
        code_switch_ratio=metrics.code_switch_ratio,
        learner_talk_sec=metrics.learner_talk_sec,
        learner_turn_count=metrics.learner_turn_count,
        total_learner_words=metrics.total_learner_words,
        estimated_level=extraction.estimated_level,
        lesson_topics=extraction.lesson_topics,
        strengths_json=json.dumps(
            [{"area": s.area, "quote": s.quote, "note": s.note} for s in extraction.strengths],
            ensure_ascii=False,
        ),
        summary=extraction.session_summary,
        turns_json=_compact_transcript(transcript),
        embedding=kwargs["session_embedding"],
    )

    # 3. Relationships: Learner ATTENDED Session + Tutor CONDUCTED Session
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

    _write_grammar_errors(tx, learner_id, session_id, session_date,
                          extraction.grammar_errors, kwargs["grammar_embeddings"])
    _write_vocabulary(tx, learner_id, session_id, session_date,
                      kwargs["vocab"], kwargs["cefr_map"])
    _write_goals(tx, learner_id, session_id, session_date,
                 extraction.learner_goals, kwargs["goal_embeddings"])
    _write_skills(tx, session_id, extraction.skills_covered)


def _clear_session(tx: Any, session_id: str) -> None:
    tx.run(
        """
        MATCH (s:Session {id: $sid})-[r:INTRODUCED_WORD|COVERED_SKILL|DISCUSSED_GOAL]->()
        DELETE r
        """,
        sid=session_id,
    )
    tx.run(
        """
        MATCH (:Learner)-[r:MADE_ERROR|USED_WORD]->()
        WHERE r.session_id = $sid
        DELETE r
        """,
        sid=session_id,
    )


def _write_grammar_errors(
    tx: Any, learner_id: str, session_id: str, session_date: str,
    errors: List[GrammarError], embeddings: Dict[str, List[float]],
) -> None:
    for group in _group_grammar(errors):
        gp_id = f"gp_{group.category}"
        label, description = GRAMMAR_CATEGORIES.get(group.category, GRAMMAR_CATEGORIES["other"])
        first = group.examples[0] if group.examples else {}
        tx.run(
            """
            MERGE (gp:GrammarPattern {id: $id})
            ON CREATE SET gp.pattern_name = $pattern
            SET gp.label       = $label,
                gp.description = $description,
                gp.embedding   = coalesce(gp.embedding, $embedding)
            """,
            id=gp_id,
            pattern=group.category,
            label=label,
            description=description,
            embedding=embeddings.get(group.category),
        )
        tx.run(
            """
            MATCH (l:Learner {id: $lid}), (gp:GrammarPattern {id: $gpid})
            MERGE (l)-[r:MADE_ERROR {session_id: $sid}]->(gp)
            SET r.error_count       = $count,
                r.description       = $description,
                r.example_utterance = $wrong,
                r.example_correct   = $correct,
                r.explanation       = $explanation,
                r.examples_json     = $examples_json,
                r.date              = date($date)
            """,
            lid=learner_id,
            gpid=gp_id,
            sid=session_id,
            count=group.count,
            description=group.description,
            wrong=first.get("wrong", ""),
            correct=first.get("correct", ""),
            explanation=first.get("explanation", ""),
            examples_json=json.dumps(group.examples, ensure_ascii=False),
            date=session_date,
        )


def _write_vocabulary(
    tx: Any, learner_id: str, session_id: str, session_date: str,
    vocab: List[VocabItem], cefr_map: Dict[str, str],
) -> None:
    for item in vocab:
        level = cefr_map.get(item.term, item.level or "unknown")

        tx.run(
            """
            MERGE (w:Word {lemma: $lemma})
            ON CREATE SET w.cefr_level = $level
            SET w.kind              = coalesce(w.kind, $kind),
                w.pos               = CASE WHEN w.pos IS NULL OR w.pos = 'unknown' THEN $pos ELSE w.pos END,
                w.meaning           = coalesce(w.meaning, $meaning),
                w.workplace_example = coalesce(w.workplace_example, $example),
                w.cefr_level        = CASE WHEN w.cefr_level IS NULL OR w.cefr_level = 'unknown'
                                           THEN $level ELSE w.cefr_level END
            """,
            lemma=item.term,
            level=level,
            kind=item.kind,
            pos=item.part_of_speech or None,
            meaning=item.meaning or None,
            example=item.workplace_example or None,
        )

        if item.introduced_by_tutor:
            tx.run(
                """
                MATCH (s:Session {id: $sid}), (w:Word {lemma: $lemma})
                MERGE (s)-[r:INTRODUCED_WORD]->(w)
                SET r.context_sentence = $ctx
                """,
                sid=session_id,
                lemma=item.term,
                ctx=item.context_sentence,
            )

        if item.used_by_learner:
            tx.run(
                """
                MATCH (l:Learner {id: $lid}), (w:Word {lemma: $lemma})
                MERGE (l)-[r:USED_WORD {session_id: $sid}]->(w)
                SET r.correct          = $correct,
                    r.correction       = $correction,
                    r.spontaneous      = $spontaneous,
                    r.mastery          = $mastery,
                    r.context_sentence = $ctx,
                    r.date             = date($date)
                """,
                lid=learner_id,
                lemma=item.term,
                sid=session_id,
                correct=item.used_correctly,
                correction=item.correction,
                spontaneous=item.spontaneous,
                mastery=item.mastery,
                ctx=item.context_sentence,
                date=session_date,
            )


def _write_goals(
    tx: Any, learner_id: str, session_id: str, session_date: str,
    goals: list, embeddings: Dict[str, List[float]],
) -> None:
    for goal in goals:
        gid = goal_id_for(goal.description)
        rank = _GOAL_RANK.get(goal.status, 1)
        tx.run(
            """
            MERGE (g:Goal {id: $id})
            ON CREATE SET g.description = $desc,
                          g.created_at  = date($date)
            SET g.category  = 'learner_goal',
                g.embedding = coalesce(g.embedding, $embedding)
            """,
            id=gid,
            desc=goal.description,
            date=session_date,
            embedding=embeddings.get(goal.description),
        )
        # Status only moves forward (mentioned → practiced → demonstrated)
        tx.run(
            """
            MATCH (l:Learner {id: $lid}), (g:Goal {id: $gid})
            MERGE (l)-[r:HAS_GOAL]->(g)
            ON CREATE SET r.first_seen = date($date)
            WITH r, coalesce(r.status_rank,
                     CASE r.status WHEN 'demonstrated' THEN 3 WHEN 'practiced' THEN 2
                                   WHEN 'mentioned' THEN 1 ELSE 0 END) AS old_rank
            SET r.status      = CASE WHEN $rank >= old_rank THEN $status ELSE r.status END,
                r.status_rank = CASE WHEN $rank >= old_rank THEN $rank ELSE old_rank END,
                r.updated_at  = CASE WHEN r.updated_at IS NULL OR date($date) > r.updated_at
                                     THEN date($date) ELSE r.updated_at END
            """,
            lid=learner_id,
            gid=gid,
            status=goal.status,
            rank=rank,
            date=session_date,
        )
        tx.run(
            """
            MATCH (s:Session {id: $sid}), (g:Goal {id: $gid})
            MERGE (s)-[r:DISCUSSED_GOAL]->(g)
            SET r.status = $status
            """,
            sid=session_id,
            gid=gid,
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
