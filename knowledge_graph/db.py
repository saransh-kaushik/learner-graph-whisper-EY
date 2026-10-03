"""
knowledge_graph/db.py
─────────────────────
Singleton Neo4j driver with helper methods.

Usage
-----
    from knowledge_graph.db import neo4j_client

    records = neo4j_client.run_query("MATCH (n:Learner) RETURN n LIMIT 5")
    neo4j_client.run_write("CREATE (:Learner {id: $id, name: $name})", id="l1", name="Alice")
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, List

from neo4j import GraphDatabase, Record
from neo4j import Session as Neo4jSession

from knowledge_graph.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Thread-safe singleton wrapper around the Neo4j driver."""

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j driver connected to %s", settings.neo4j_uri)

    # ── Context manager ────────────────────────────────────────────────────────
    @contextmanager
    def session(self) -> Generator[Neo4jSession, None, None]:
        with self._driver.session(database=settings.neo4j_database) as s:
            yield s

    # ── Read ──────────────────────────────────────────────────────────────────
    def run_query(self, cypher: str, **params: Any) -> List[Record]:
        """Execute a read-only Cypher query and return all records."""
        with self.session() as s:
            result = s.run(cypher, **params)
            return list(result)

    # ── Write ─────────────────────────────────────────────────────────────────
    def run_write(self, cypher: str, **params: Any) -> List[Record]:
        """Execute a write Cypher statement inside a transaction."""

        def _tx(tx: Any) -> List[Record]:
            result = tx.run(cypher, **params)
            return list(result)

        with self.session() as s:
            return s.execute_write(_tx)

    def run_write_batch(self, cypher: str, rows: List[dict]) -> None:
        """Execute a write statement for each row in *rows*."""

        def _tx(tx: Any) -> None:
            for row in rows:
                tx.run(cypher, **row)

        with self.session() as s:
            s.execute_write(_tx)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def verify_connectivity(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.error("Neo4j connectivity check failed: %s", exc)
            return False

    def close(self) -> None:
        self._driver.close()
        logger.info("Neo4j driver closed.")


# Singleton — import this everywhere
neo4j_client = Neo4jClient()
