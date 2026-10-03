"""Print a learner's full progress report (calls Neo4j + OpenAI).

    python scripts/print_report.py <learner_id> [n_sessions] [--refresh]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_graph.reports.generator import generate_report  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
learner_id = args[0] if args else "1"
n = int(args[1]) if len(args) > 1 else None
report = generate_report(learner_id, n, refresh="--refresh" in sys.argv)
print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
