"""Print the aggregated (pre-LLM) report data for a learner (Neo4j only).

    python scripts/print_aggregate.py <learner_id> [n_sessions]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_graph.reports.aggregator import aggregate_learner_progress  # noqa: E402

learner_id = sys.argv[1] if len(sys.argv) > 1 else "1"
n = int(sys.argv[2]) if len(sys.argv) > 2 else None
data = aggregate_learner_progress(learner_id, n)
print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
