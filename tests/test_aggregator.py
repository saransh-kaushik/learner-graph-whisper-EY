from knowledge_graph.reports.aggregator import (
    _cefr_level,
    build_grammar,
    build_vocabulary,
    classify_error_series,
    metric_trend,
)


def _sessions(n: int, words: int = 200) -> list:
    return [
        {"id": f"s{i}", "date": f"2026-09-{10 + i:02d}", "is_valid": True, "total_learner_words": words}
        for i in range(n)
    ]


# ── metric trends ─────────────────────────────────────────────────────────────

def test_metric_trend_needs_three_sessions():
    assert metric_trend("words_per_turn", [5, 8])["trend"] == "insufficient_data"


def test_metric_trend_direction_respects_better():
    assert metric_trend("words_per_turn", [5, 5, 8, 9])["trend"] == "improving"
    assert metric_trend("code_switch_ratio", [0.30, 0.28, 0.10, 0.08])["trend"] == "improving"
    assert metric_trend("code_switch_ratio", [0.010, 0.012, 0.014])["trend"] == "stable"  # tiny absolute change


def test_speaking_rate_uses_target_band():
    # Moving from too slow (70) into the 110–150 band is improving
    assert metric_trend("speaking_rate_wpm", [70, 72, 115, 120])["trend"] == "improving"
    # Moving from 130 to 175 (too fast) is declining even though it is "higher"
    assert metric_trend("speaking_rate_wpm", [130, 130, 170, 180])["trend"] == "declining"


# ── grammar ───────────────────────────────────────────────────────────────────

def test_classify_error_series():
    assert classify_error_series([3, 2, 0, 0]) == "resolved"
    assert classify_error_series([4, 4, 1, 1]) == "fading"
    assert classify_error_series([2, 2, 2, 2]) == "persistent"
    assert classify_error_series([1, 1, 3, 3]) == "increasing"
    assert classify_error_series([0, 0, 2]) == "new"
    assert classify_error_series([2]) == "new"


def test_build_grammar_zero_fills_and_pairs_examples():
    sessions = _sessions(4)
    rows = [
        # articles: present early, gone in the last two sessions → resolved
        {"pattern": "article_usage", "count": 3, "session_id": "s0", "wrong": "I am engineer",
         "correct": None, "gp_wrong": "She is doctor", "gp_correct": "She is a doctor", "examples_json": None},
        {"pattern": "articles", "count": 2, "session_id": "s1",
         "examples_json": '[{"wrong": "Give me pen", "correct": "Give me a pen", "explanation": "Use a/an."}]'},
        # tense: every session → persistent, priority
        *[{"pattern": "verb_tense", "count": 2, "session_id": s["id"],
           "examples_json": f'[{{"wrong": "I go {s["id"]}", "correct": "I went {s["id"]}"}}]'} for s in sessions],
    ]
    grammar, priority = build_grammar(rows, sessions)

    improved = {p["pattern"]: p for p in grammar["improved"]}
    assert improved["articles"]["status"] == "resolved"
    assert [pt["value"] for pt in improved["articles"]["series"]] == [1.5, 1.0, 0, 0]  # per 100 words, zero-filled
    # Legacy row: gp correction does not belong to this sentence → not paired
    legacy = next(e for e in improved["articles"]["examples"] if e["wrong"] == "I am engineer")
    assert legacy["correct"] is None

    assert [p["pattern"] for p in priority] == ["verb_tense"]
    assert priority[0]["examples"][0]["wrong"] == "I go s3"      # most recent first
    assert priority[0]["examples"][0]["correct"] == "I went s3"


# ── vocabulary ────────────────────────────────────────────────────────────────

def test_build_vocabulary_categories():
    sessions = _sessions(3)[1:]          # window = s1, s2 (s0 is before the window)
    used = [
        # used before the window → not "new"
        {"word": "deadline", "level": "B1", "kind": "word", "mastery": "spontaneous", "correct": True,
         "session_id": "s0", "date": "2026-09-10"},
        {"word": "deadline", "level": "B1", "kind": "word", "mastery": "spontaneous", "correct": True,
         "session_id": "s1", "date": "2026-09-11"},
        # genuinely new, used correctly
        {"word": "negotiate", "level": "B2", "kind": "word", "meaning": "discuss to agree",
         "mastery": "spontaneous", "correct": True, "context": "We negotiate the price",
         "session_id": "s2", "date": "2026-09-12"},
        # only repeated after tutor → not learned
        {"word": "leverage", "level": "C1", "kind": "word", "mastery": "repeated", "correct": True,
         "session_id": "s2", "date": "2026-09-12"},
        # misused
        {"word": "revert", "level": "B2", "kind": "word", "mastery": "spontaneous", "correct": False,
         "correction": "reply", "context": "Please revert me", "session_id": "s2", "date": "2026-09-12"},
        # basic word → ignored
        {"word": "good", "level": "A1", "kind": "word", "mastery": "spontaneous", "correct": True,
         "session_id": "s2", "date": "2026-09-12"},
        # taught in s1, used on own in s2 → now independent
        {"word": "follow up on", "level": "B2", "kind": "phrasal_verb", "mastery": "spontaneous",
         "correct": True, "context": "I will follow up on it", "session_id": "s2", "date": "2026-09-12"},
    ]
    intro = [
        {"word": "follow up on", "level": "B2", "kind": "phrasal_verb", "session_id": "s1", "date": "2026-09-11"},
        {"word": "stakeholder", "level": "B2", "kind": "word", "session_id": "s2", "date": "2026-09-12",
         "context": "Inform every stakeholder"},
    ]
    v = build_vocabulary(used, intro, sessions)

    assert [w["word"] for w in v["new_words"]] == ["negotiate"]
    assert v["new_words"][0]["level_hint"] == "Professional"
    assert [w["word"] for w in v["now_independent"]] == ["follow up on"]
    assert [w["word"] for w in v["words_to_fix"]] == ["revert"]
    assert v["words_to_fix"][0]["correction"] == "reply"
    assert [w["word"] for w in v["words_to_try"]] == ["stakeholder"]


def test_cefr_level_prefers_tutor_then_session_estimates():
    assert _cefr_level("B2", [])["source"] == "tutor"
    est = _cefr_level(None, [{"estimated_level": "B1"}, {"estimated_level": "B2"}, {"estimated_level": "B2"}])
    assert est["value"] == "B2" and est["source"] == "estimated" and est["label"] == "Upper-intermediate"
    assert _cefr_level(None, [])["confidence"] == "insufficient_data"
