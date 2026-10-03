"""
knowledge_graph/search.py
─────────────────────────
Semantic search using Neo4j vector indexes.

All three search functions use db.index.vector.queryNodes()
which requires Neo4j 5.x with the vector indexes applied (schema.py).
"""
from __future__ import annotations

import logging
from typing import List

from knowledge_graph.db import neo4j_client
from knowledge_graph.llm.embeddings import embed_text

logger = logging.getLogger(__name__)


def find_similar_errors(pattern_text: str, top_k: int = 5) -> List[dict]:
    """
    Find GrammarPattern nodes whose embeddings are closest to *pattern_text*.
    Returns list of {id, pattern_name, description, score}.
    """
    query_vec = embed_text(pattern_text)
    records = neo4j_client.run_query(
        """
        CALL db.index.vector.queryNodes('grammar_embedding', $k, $vec)
        YIELD node AS gp, score
        RETURN gp.id          AS id,
               gp.pattern_name AS pattern_name,
               gp.description  AS description,
               score
        """,
        k=top_k,
        vec=query_vec,
    )
    return [dict(r) for r in records]


def find_similar_sessions(summary_text: str, top_k: int = 5) -> List[dict]:
    """
    Find Session nodes whose embeddings are closest to *summary_text*.
    Returns list of {id, date, learner_id, transcript_summary, score}.
    """
    query_vec = embed_text(summary_text)
    records = neo4j_client.run_query(
        """
        CALL db.index.vector.queryNodes('session_embedding', $k, $vec)
        YIELD node AS s, score
        RETURN s.id                AS id,
               s.date              AS date,
               s.learner_id        AS learner_id,
               s.transcript_summary AS transcript_summary,
               score
        """,
        k=top_k,
        vec=query_vec,
    )
    return [dict(r) for r in records]


def find_similar_goals(goal_text: str, top_k: int = 5) -> List[dict]:
    """
    Find Goal nodes whose embeddings are closest to *goal_text*.
    Returns list of {id, description, score}.
    """
    query_vec = embed_text(goal_text)
    records = neo4j_client.run_query(
        """
        CALL db.index.vector.queryNodes('goal_embedding', $k, $vec)
        YIELD node AS g, score
        RETURN g.id          AS id,
               g.description AS description,
               score
        """,
        k=top_k,
        vec=query_vec,
    )
    return [dict(r) for r in records]
