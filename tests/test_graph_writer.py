from unittest.mock import MagicMock, patch

from knowledge_graph.ingest.graph_writer import goal_id_for, write_session
from knowledge_graph.ingest.parser import ParsedTranscript
from knowledge_graph.llm.extractor import ExtractionResult, GoalItem, GrammarError, VocabItem
from knowledge_graph.metrics import SessionMetrics


def _run_write(extraction: ExtractionResult) -> MagicMock:
    transcript = ParsedTranscript("test.json", [], "l1", "t1")
    metrics = SessionMetrics(0.5, 10.0, 8.0, 120.0, 0.1, 100.0, 100.0, 200.0, 10, 100)
    mock_tx = MagicMock()
    with patch("knowledge_graph.ingest.graph_writer.neo4j_client") as mock_neo4j, \
         patch("knowledge_graph.ingest.graph_writer.lookup_cefr_batch",
               side_effect=lambda words, hints=None: {w: hints.get(w, "B2") for w in words}):
        mock_neo4j.session.return_value.__enter__.return_value.execute_write.side_effect = (
            lambda f, **k: f(mock_tx, **k)
        )
        write_session(
            session_id="s1", learner_id="l1", tutor_id="t1", session_date="2026-10-01",
            transcript=transcript, metrics=metrics, extraction=extraction,
            session_embedding=[0.1] * 1536,
        )
    return mock_tx


def _calls_matching(tx: MagicMock, fragment: str) -> list:
    return [c for c in tx.run.call_args_list if fragment in c.args[0]]


def test_write_session_groups_grammar_and_stores_matching_corrections():
    extraction = ExtractionResult(
        grammar_errors=[
            GrammarError("verb_tense", "past simple", "I go yesterday", "I went yesterday", "Use past.", 2),
            GrammarError("verb_tense", "present perfect", "I have went", "I have gone", "Use participle.", 1),
        ],
        session_summary="Good session.",
    )
    tx = _run_write(extraction)

    # Previous analysis of the same session is cleared first
    assert _calls_matching(tx, "DELETE r")
    edges = _calls_matching(tx, "MERGE (l)-[r:MADE_ERROR")
    assert len(edges) == 1                       # one edge per category per session
    kwargs = edges[0].kwargs
    assert kwargs["count"] == 3
    assert kwargs["wrong"] == "I go yesterday"
    assert kwargs["correct"] == "I went yesterday"   # paired with its own sentence
    assert "I have gone" in kwargs["examples_json"]


def test_only_tutor_taught_words_get_introduced_edge():
    extraction = ExtractionResult(vocabulary=[
        VocabItem("negotiate", source="learner_spontaneous", introduced_by_tutor=False),
        VocabItem("follow up on", kind="phrasal_verb", source="learner_after_prompt", introduced_by_tutor=True),
        VocabItem("stakeholder", source="tutor_only", introduced_by_tutor=True),
    ])
    tx = _run_write(extraction)

    introduced = {c.kwargs["lemma"] for c in _calls_matching(tx, "MERGE (s)-[r:INTRODUCED_WORD")}
    used = {c.kwargs["lemma"]: c.kwargs["mastery"] for c in _calls_matching(tx, "USED_WORD {session_id")}
    assert introduced == {"follow up on", "stakeholder"}
    assert used == {"negotiate": "spontaneous", "follow up on": "prompted"}


def test_goal_status_write_never_regresses_and_ids_are_normalised():
    extraction = ExtractionResult(learner_goals=[GoalItem("Speak confidently in client calls", "practiced")])
    tx = _run_write(extraction)
    cypher = _calls_matching(tx, "HAS_GOAL")[0].args[0]
    assert "$rank >= old_rank" in cypher
    assert goal_id_for("Speak confidently in client calls.") == goal_id_for("speak  confidently in CLIENT calls")
