from knowledge_graph.db import neo4j_client
import json

rows = neo4j_client.run_query("MATCH (s:Session) RETURN s limit 1")
print(rows)
