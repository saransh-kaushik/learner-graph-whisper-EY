import pytest
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
    
    # Check roles
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
