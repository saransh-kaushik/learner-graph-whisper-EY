from knowledge_graph.reports.aggregator import aggregate_learner_progress
import json
from datetime import date, datetime

def default_serializer(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    # also handle Neo4j spatial or duration if any, but we know it's date
    from neo4j.time import Date, DateTime, Time
    if isinstance(obj, (Date, DateTime, Time)):
        return obj.iso_format()
    raise TypeError(f"Type {type(obj)} not serializable")

data = aggregate_learner_progress("1", 4)
print(json.dumps(data, indent=2, default=default_serializer))
