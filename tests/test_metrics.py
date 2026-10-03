import pytest
from knowledge_graph.ingest.parser import ParsedTranscript, Turn
from knowledge_graph.metrics import compute_session_metrics

def test_metrics_computation():
    turns = [
        Turn("SPK_0", "tutor", 0.0, 10.0, "Welcome to the class. Let's start."),
        Turn("SPK_1", "learner", 10.5, 20.5, "Thank you. Main Hindi bolta hu."),
        Turn("SPK_0", "tutor", 21.0, 30.0, "Great.")
    ]
    
    transcript = ParsedTranscript(
        session_path="test.json",
        turns=turns,
        learner_label="SPK_1",
        tutor_label="SPK_0"
    )
    
    metrics = compute_session_metrics(transcript)
    
    # tutor spoke 10 + 9 = 19 sec. learner spoke 10 sec. total = 29 sec.
    assert metrics.learner_talk_sec == 10.0
    assert metrics.tutor_talk_sec == 19.0
    assert abs(metrics.talk_time_ratio - (10.0 / 29.0)) < 0.01
    
    # Learner words: 6
    # WPM: 6 words / (10 / 60 min) = 36 WPM
    assert metrics.words_per_turn == 6.0
    assert abs(metrics.speaking_rate_wpm - 36.0) < 0.1
    
    # Code switch: "Main", "Hindi", "bolta", "hu" -> 4 Hindi words? "hu" is in list. "main" is in list.
    assert metrics.code_switch_ratio > 0.0
