"""Print one stored Session node (without its embedding or stored transcript).

    python scripts/peek_session.py [session_id]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_graph.db import neo4j_client  # noqa: E402

if len(sys.argv) > 1:
    rows = neo4j_client.run_query("MATCH (s:Session {id: $id}) RETURN s", id=sys.argv[1])
else:
    rows = neo4j_client.run_query("MATCH (s:Session) RETURN s ORDER BY s.date DESC LIMIT 1")
for r in rows:
    props = dict(r["s"])
    props.pop("embedding", None)
    props.pop("transcript_turns_json", None)
    for k, v in sorted(props.items()):
        print(f"{k:28} {v}")
