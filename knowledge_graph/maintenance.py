"""
knowledge_graph/maintenance.py
──────────────────────────────
Data clean-up helpers.

Sessions ingested before session ids were derived from the recording URL got
random ids, so re-submitting the same recordings would create duplicates.
Use these commands to remove old sessions before re-ingesting.

    python -m knowledge_graph.maintenance list-sessions   --learner-id 1
    python -m knowledge_graph.maintenance delete-session  --session-id <id>
    python -m knowledge_graph.maintenance reset-learner   --learner-id 1 --yes
"""
from __future__ import annotations

import argparse
import logging

from knowledge_graph.db import neo4j_client

logger = logging.getLogger(__name__)


def delete_session(session_id: str) -> None:
    """Delete one session and every relationship that belongs to it."""
    neo4j_client.run_write(
        """
        MATCH (:Learner)-[r:MADE_ERROR|USED_WORD]->()
        WHERE r.session_id = $sid
        DELETE r
        """,
        sid=session_id,
    )
    neo4j_client.run_write("MATCH (s:Session {id: $sid}) DETACH DELETE s", sid=session_id)


def reset_learner(learner_id: str) -> int:
    """Delete all sessions, errors, word usage and goal links of a learner.
    The Learner node itself (name, level) is kept. Returns sessions deleted."""
    rows = neo4j_client.run_query(
        "MATCH (:Learner {id: $id})-[:ATTENDED]->(s:Session) RETURN s.id AS id", id=learner_id,
    )
    for r in rows:
        delete_session(r["id"])
    neo4j_client.run_write(
        "MATCH (:Learner {id: $id})-[r:MADE_ERROR|USED_WORD|HAS_GOAL]->() DELETE r", id=learner_id,
    )
    return len(rows)


def _cli() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Knowledge graph maintenance")
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list-sessions")
    ls.add_argument("--learner-id", required=True)
    ds = sub.add_parser("delete-session")
    ds.add_argument("--session-id", required=True)
    rl = sub.add_parser("reset-learner")
    rl.add_argument("--learner-id", required=True)
    rl.add_argument("--yes", action="store_true", help="Confirm deletion")
    args = p.parse_args()

    if args.cmd == "list-sessions":
        rows = neo4j_client.run_query(
            """
            MATCH (:Learner {id: $id})-[:ATTENDED]->(s:Session)
            RETURN s.id AS id, toString(s.date) AS date, s.source_url AS url,
                   s.transcript_turns_json IS NOT NULL AS reanalyzable
            ORDER BY s.date
            """,
            id=args.learner_id,
        )
        for r in rows:
            print(f"{r['date']}  {r['id']}  reanalyzable={r['reanalyzable']}  {r['url'] or ''}")
    elif args.cmd == "delete-session":
        delete_session(args.session_id)
        print(f"Deleted session {args.session_id}")
    elif args.cmd == "reset-learner":
        if not args.yes:
            p.error("reset-learner deletes data; pass --yes to confirm")
        print(f"Deleted {reset_learner(args.learner_id)} sessions for learner {args.learner_id}")


if __name__ == "__main__":
    _cli()
