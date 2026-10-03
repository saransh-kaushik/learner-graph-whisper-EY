from knowledge_graph.ingest.parser import ParsedTranscript, Turn
from knowledge_graph.metrics import compute_session_metrics, hindi_token_flags


def _transcript(learner_text: str) -> ParsedTranscript:
    turns = [
        Turn("SPK_0", "tutor", 0.0, 10.0, "Welcome to the class. Let's start."),
        Turn("SPK_1", "learner", 10.5, 20.5, learner_text),
        Turn("SPK_0", "tutor", 21.0, 30.0, "Great."),
    ]
    return ParsedTranscript("test.json", turns, learner_label="SPK_1", tutor_label="SPK_0")


def test_metrics_computation():
    metrics = compute_session_metrics(_transcript("Thank you. Main Hindi bolta hu."))

    # tutor spoke 10 + 9 = 19 sec. learner spoke 10 sec. total = 29 sec.
    assert metrics.learner_talk_sec == 10.0
    assert metrics.tutor_talk_sec == 19.0
    assert abs(metrics.talk_time_ratio - (10.0 / 29.0)) < 0.01

    # 6 learner words in 10 seconds → 36 WPM
    assert metrics.words_per_turn == 6.0
    assert abs(metrics.speaking_rate_wpm - 36.0) < 0.1
    assert metrics.total_learner_words == 6

    # "bolta" and "hu" are Hindi
    assert metrics.code_switch_ratio > 0.0


def test_plain_english_is_not_counted_as_hindi():
    text = "I think the main problem is the deadline for the project. Do you agree?"
    metrics = compute_session_metrics(_transcript(text))
    assert metrics.code_switch_ratio == 0.0
    # sentence lengths 11 and 3 English words → mean 7.0
    assert metrics.average_words_per_sentence == 7.0


def test_ambiguous_words_count_as_hindi_next_to_hindi():
    tokens = "main bolta hoon par the main point is clear".split()
    flags = hindi_token_flags(tokens)
    assert flags[:4] == [True, True, True, True]  # main bolta hoon par
    assert flags[4:] == [False] * 5               # the main point is clear


def test_ambiguous_word_adjacent_to_hindi():
    flags = hindi_token_flags("kaam par hai".split())
    assert flags == [False, True, True]          # "kaam" unknown, "par" next to "hai"
