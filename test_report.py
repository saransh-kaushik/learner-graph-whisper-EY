from knowledge_graph.reports.generator import generate_report
import json

report = generate_report("1")
print(json.dumps(report, indent=2))
