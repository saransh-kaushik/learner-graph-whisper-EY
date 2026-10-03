import pytest
from unittest.mock import patch, MagicMock
from knowledge_graph.ingest.graph_writer import write_session
from knowledge_graph.ingest.parser import ParsedTranscript
from knowledge_graph.metrics import SessionMetrics
from knowledge_graph.llm.extractor import ExtractionResult, GrammarError, VocabItem, GoalItem

@patch("knowledge_graph.ingest.graph_writer.neo4j_client")
def test_write_session(mock_neo4j):
    # Mock data
    transcript = ParsedTranscript("test.json", [], "l1", "t1")
    metrics = SessionMetrics(0.5, 10.0, 50.0, 0.1, 10.0, 10.0, 20.0, 5, 50)
    extraction = ExtractionResult(
        grammar_errors=[GrammarError("tense", "Wrong tense", "I go yesterday", "I went yesterday", 1)],
        vocabulary=[VocabItem("test", "This is a test", False, True, True, True)],
        goals_discussed=[GoalItem("Fluency", "practiced")],
        session_summary="Good session."
    )
    embedding = [0.1] * 1536
    
    mock_tx = MagicMock()
    mock_neo4j.session.return_value.__enter__.return_value.write_transaction.side_effect = lambda f, **k: f(mock_tx, **k)

    write_session(
        session_id="s1",
        learner_id="l1",
        tutor_id="t1",
        session_date="2026-10-01",
        transcript=transcript,
        metrics=metrics,
        extraction=extraction,
        session_embedding=embedding
    )
    
    # Verify that transaction was run multiple times for nodes/edges
    assert mock_tx.run.call_count > 5
