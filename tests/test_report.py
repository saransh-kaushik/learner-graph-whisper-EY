import pytest
from unittest.mock import patch
from knowledge_graph.reports.generator import generate_report

@patch("knowledge_graph.reports.generator.aggregate_learner_progress")
@patch("knowledge_graph.reports.generator._get_client")
def test_generate_report(mock_get_client, mock_aggregator):
    # Mock Neo4j aggregated data
    mock_aggregator.return_value = {
        "learner_id": "l1",
        "learner_name": "Test Learner",
        "cefr_level": "B2",
        "sessions_analyzed": 5,
        "metrics": {
            "talk_time_ratio": {"current": 0.5, "baseline": 0.4, "trend": "improving"}
        },
        "vocabulary": {"new_words": [], "mastery_upgrades": []},
        "grammar": {"fading_errors": [], "persistent_errors": []},
        "goals": [],
        "sessions": []
    }
    
    # Mock OpenAI response
    mock_openai = mock_get_client.return_value
    mock_openai.chat.completions.create.return_value.choices[0].message.content = '{"llm_summary": "Great progress.", "practice_next": "Keep going."}'
    
    report = generate_report("l1", 5)
    
    assert report["learner_id"] == "l1"
    assert report["llm_summary"] == "Great progress."
    assert report["practice_next"] == "Keep going."
    assert report["metrics"]["talk_time_ratio"]["trend"] == "improving"
