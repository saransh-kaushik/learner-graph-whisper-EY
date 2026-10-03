from knowledge_graph.ingest.orchestrator import format_transcript_for_llm
from knowledge_graph.ingest.parser import TranscriptParser


def test_transcript_parser_basic():
    data = {
        "segments": [
            {"speaker": "SPEAKER_00", "text": "Hello, how are you?", "start": 0.0, "end": 2.0},
            {"speaker": "SPEAKER_01", "text": "I am fine, thank you.", "start": 2.5, "end": 4.5},
            {"speaker": "SPEAKER_00", "text": "That is good to hear.", "start": 5.0, "end": 7.0}
        ]
    }

    parser = TranscriptParser()
    parsed = parser.parse_dict(data, session_path="dummy.json")

    assert len(parsed.turns) == 3
    # Learner should be SPEAKER_01 because they spoke less (2.0s vs 4.0s)
    assert parsed.learner_label == "SPEAKER_01"
    assert parsed.tutor_label == "SPEAKER_00"

    assert parsed.turns[0].role == "tutor"
    assert parsed.turns[1].role == "learner"
    assert parsed.turns[2].role == "tutor"


def test_transcript_parser_with_map():
    data = {
        "segments": [
            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "text": "Hi", "start": 1.0, "end": 2.0}
        ]
    }

    parser = TranscriptParser(speaker_map={"SPEAKER_00": "learner", "SPEAKER_01": "tutor"})
    parsed = parser.parse_dict(data, session_path="dummy.json")

    assert parsed.learner_label == "SPEAKER_00"
    assert parsed.turns[0].role == "learner"


def test_blank_segments_are_skipped_and_roles_can_be_swapped():
    data = [
        {"speaker": "A", "text": "Tell me about your weekend.", "start": 0, "end": 2},
        {"speaker": "B", "text": "  ", "start": 2, "end": 3},
        {"speaker": "B", "text": "I went to my village and met my family there.", "start": 3, "end": 9},
    ]
    parsed = TranscriptParser().parse_dict(data)
    assert len(parsed.turns) == 2
    assert parsed.learner_label == "A"          # heuristic: less talk time

    parsed.assign_roles("B", "A")               # e.g. after LLM role check
    assert parsed.learner_label == "B"
    assert [t.role for t in parsed.turns] == ["tutor", "learner"]


def test_llm_format_uses_role_labels_and_merges_consecutive_turns():
    data = [
        {"speaker": "A", "text": "So tell me,", "start": 0, "end": 1},
        {"speaker": "A", "text": "what did you do yesterday?", "start": 1, "end": 2},
        {"speaker": "B", "text": "I go to office.", "start": 2, "end": 3},
    ]
    parsed = TranscriptParser(speaker_map={"A": "tutor", "B": "learner"}).parse_dict(data)
    assert format_transcript_for_llm(parsed) == (
        "TUTOR: So tell me, what did you do yesterday?\nLEARNER: I go to office."
    )
