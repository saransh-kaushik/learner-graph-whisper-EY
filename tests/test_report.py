import json
from unittest.mock import patch

from api.models.kg_models import ProgressReport
from api.session_dates import infer_session_date
from knowledge_graph.reports import generator
from knowledge_graph.reports.aggregator import build_grammar, build_vocabulary, compute_metric_trends


def _aggregated() -> dict:
    sessions = [
        {"id": f"s{i}", "date": f"2026-09-{10 + i:02d}", "is_valid": True, "total_learner_words": 200,
         "talk_time_ratio": 0.3 + i * 0.05, "words_per_turn": 6 + i, "speaking_rate_wpm": 100 + i * 5,
         "code_switch_ratio": 0.2 - i * 0.03, "average_words_per_sentence": 7 + i}
        for i in range(4)
    ]
    grammar, priority = build_grammar(
        [{"pattern": "verb_tense", "count": 2, "session_id": s["id"],
          "examples_json": json.dumps([{"wrong": "I go yesterday", "correct": "I went yesterday"}])}
         for s in sessions],
        sessions,
    )
    return {
        "learner_id": "l1",
        "learner_name": "Kiran Paunikar",
        "native_language": "Hindi",
        "cefr_level": {"value": "B1", "label": "Intermediate", "description": "…", "source": "estimated",
                       "confidence": "medium"},
        "period": {"start": "2026-09-10", "end": "2026-09-13", "learner_minutes": 40.0,
                   "total_minutes": 120.0, "learner_words": 800},
        "sessions_analyzed": {"valid": 4, "total": 5},
        "data_quality": [{"session_id": "s9", "date": "2026-09-09", "reasons": ["no_learner_speech"]}],
        "metrics": compute_metric_trends(sessions),
        "vocabulary": build_vocabulary(
            [{"word": "negotiate", "level": "B2", "kind": "word", "mastery": "spontaneous",
              "correct": True, "session_id": "s3", "date": "2026-09-13"}], [], sessions),
        "grammar": grammar,
        "top_priority_errors": priority,
        "strengths": [{"area": "fluency", "quote": "I led the call", "note": "", "date": "2026-09-13"}],
        "learner_goals": [{"description": "Lead client calls", "status": "practiced", "date": "2026-09-13",
                           "first_seen": "2026-09-10"}],
        "lesson_topics": [{"topic": "Project updates", "date": "2026-09-13"}],
        "sessions": [{"session_id": "s3", "date": "2026-09-13", "is_valid": True, "duration_min": 30,
                      "learner_minutes": 10, "summary": "You practised updates.", "topics": []}],
    }


def _llm_response(content: dict):
    def create(**kwargs):
        resp = type("R", (), {})()
        msg = type("M", (), {"content": json.dumps(content), "refusal": None})()
        resp.choices = [type("C", (), {"message": msg, "finish_reason": "stop"})()]
        return resp
    return create


NARRATIVE = {
    "headline": "Longer answers, same tense slip",
    "summary": "You are speaking more.",
    "wins": [{"title": "More English", "detail": "Less Hindi mixed in.", "evidence": "20% → 11%"}],
    "focus_areas": [{"title": "Past tense", "why_it_matters": "Clear updates.",
                     "your_sentence": "I invented this sentence", "better_sentence": "x", "tip": "Use -ed."}],
    "practice_plan": [{"title": "Stand-up", "minutes": 5, "situation": "Daily stand-up", "steps": ["Say it"]}],
    "next_milestone": "Use past tense in two updates.",
}


def setup_function():
    generator._cache.clear()


@patch("knowledge_graph.reports.generator.aggregate_learner_progress")
@patch("knowledge_graph.reports.generator._get_client")
def test_report_matches_api_model_and_replaces_invented_quotes(mock_client, mock_agg):
    mock_agg.return_value = _aggregated()
    mock_client.return_value.chat.completions.create.side_effect = _llm_response(NARRATIVE)

    report = generator.generate_report("l1", 5)
    ProgressReport(**report)                       # the API response model accepts it

    fa = report["narrative"]["focus_areas"][0]
    assert fa["your_sentence"] == "I go yesterday"   # invented quote replaced with the real one
    assert fa["better_sentence"] == "I went yesterday"
    assert report["narrative"]["source"] == "ai"

    payload = generator.build_prompt_payload(mock_agg.return_value)
    assert "talk_time_ratio" not in json.dumps(payload)   # only plain labels reach the LLM
    assert payload["learner_first_name"] == "Kiran"


@patch("knowledge_graph.reports.generator.aggregate_learner_progress")
@patch("knowledge_graph.reports.generator._get_client")
def test_report_is_cached_until_new_session(mock_client, mock_agg):
    mock_agg.return_value = _aggregated()
    mock_client.return_value.chat.completions.create.side_effect = _llm_response(NARRATIVE)

    generator.generate_report("l1", 5)
    generator.generate_report("l1", 5)
    assert mock_client.return_value.chat.completions.create.call_count == 1

    generator.generate_report("l1", 5, refresh=True)
    assert mock_client.return_value.chat.completions.create.call_count == 2


@patch("knowledge_graph.reports.generator.aggregate_learner_progress")
@patch("knowledge_graph.reports.generator._get_client")
def test_fallback_narrative_when_llm_fails(mock_client, mock_agg):
    mock_agg.return_value = _aggregated()
    mock_client.return_value.chat.completions.create.side_effect = RuntimeError("down")

    report = generator.generate_report("l1", 5)
    ProgressReport(**report)
    n = report["narrative"]
    assert n["source"] == "fallback"
    assert n["focus_areas"][0]["your_sentence"] == "I go yesterday"
    assert len(n["practice_plan"]) == 3
    assert "1 skipped" in n["summary"]


@patch("knowledge_graph.reports.generator.aggregate_learner_progress")
def test_empty_report_validates(mock_agg):
    data = _aggregated()
    data.update(sessions_analyzed={"valid": 0, "total": 1}, metrics={}, top_priority_errors=[],
                grammar={"improved": [], "needs_work": [], "new": []})
    mock_agg.return_value = data
    report = generator.generate_report("l1", 5)
    ProgressReport(**report)
    assert report["narrative"]["source"] == "empty"


def test_session_date_inference_from_url():
    url = "https://x.cloudfront.net/Session-Recordings/2026/Sep/21/abc/rec.mp4"
    assert infer_session_date(url) == "2026-09-21"
    assert infer_session_date("https://x/rec_2026-08-05.mp3") == "2026-08-05"
    assert infer_session_date("https://x/rec.mp3") is None
    assert infer_session_date("https://x/2026/Feb/31/rec.mp3") is None
