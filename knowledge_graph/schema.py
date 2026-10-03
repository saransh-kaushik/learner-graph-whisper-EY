"""
knowledge_graph/schema.py
─────────────────────────
Creates Neo4j constraints and vector indexes on a fresh (or existing) database.

Run once:
    python -m knowledge_graph.schema
"""
from __future__ import annotations

import logging

from knowledge_graph.db import neo4j_client

logger = logging.getLogger(__name__)

# ── Uniqueness constraints ─────────────────────────────────────────────────────
CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT learner_id  IF NOT EXISTS FOR (l:Learner)        REQUIRE l.id IS UNIQUE",
    "CREATE CONSTRAINT tutor_id    IF NOT EXISTS FOR (t:Tutor)           REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT session_id  IF NOT EXISTS FOR (s:Session)         REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT goal_id     IF NOT EXISTS FOR (g:Goal)            REQUIRE g.id IS UNIQUE",
    "CREATE CONSTRAINT grammar_id  IF NOT EXISTS FOR (gp:GrammarPattern) REQUIRE gp.id IS UNIQUE",
    "CREATE CONSTRAINT word_lemma  IF NOT EXISTS FOR (w:Word)            REQUIRE w.lemma IS UNIQUE",
    "CREATE CONSTRAINT skill_name  IF NOT EXISTS FOR (sk:Skill)          REQUIRE sk.name IS UNIQUE",
    # Relationship property indexes — per-session lookups and re-ingest cleanup
    "CREATE INDEX made_error_session IF NOT EXISTS FOR ()-[r:MADE_ERROR]-() ON (r.session_id)",
    "CREATE INDEX used_word_session  IF NOT EXISTS FOR ()-[r:USED_WORD]-()  ON (r.session_id)",
    "CREATE INDEX session_date       IF NOT EXISTS FOR (s:Session) ON (s.date)",
]

# ── Vector indexes (Neo4j 5.x) ─────────────────────────────────────────────────
VECTOR_INDEXES: list[str] = [
    """
    CREATE VECTOR INDEX session_embedding IF NOT EXISTS
      FOR (s:Session) ON (s.embedding)
      OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
    """,
    """
    CREATE VECTOR INDEX goal_embedding IF NOT EXISTS
      FOR (g:Goal) ON (g.embedding)
      OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
    """,
    """
    CREATE VECTOR INDEX grammar_embedding IF NOT EXISTS
      FOR (gp:GrammarPattern) ON (gp.embedding)
      OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
    """,
]


def apply_schema() -> None:
    """Apply all constraints and indexes to the connected Neo4j database."""
    logger.info("Applying Neo4j schema constraints …")
    with neo4j_client.session() as s:
        for cypher in CONSTRAINTS:
            try:
                s.run(cypher)
                logger.debug("OK: %s", cypher.strip()[:60])
            except Exception as exc:
                logger.warning("Constraint skipped (%s): %s", type(exc).__name__, exc)

        logger.info("Applying vector indexes …")
        for cypher in VECTOR_INDEXES:
            try:
                s.run(cypher)
                logger.debug("OK: vector index created")
            except Exception as exc:
                logger.warning("Index skipped (%s): %s", type(exc).__name__, exc)

    logger.info("Schema setup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_schema()
