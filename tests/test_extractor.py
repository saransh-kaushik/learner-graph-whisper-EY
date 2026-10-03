from unittest.mock import patch

import pytest

from knowledge_graph.llm.extractor import (
    ExtractionError,
    _json_schema,
    _parse_extraction,
    extract_from_transcript,
)
from knowledge_graph.taxonomy import normalize_grammar_category


def test_parse_extraction_normalises_and_validates():
    data = {
        "grammar_errors": [
            {"category": "verb_tense", "description": "past", "example_wrong": "I go yesterday",
             "example_correct": "I went yesterday", "explanation": "Use past.", "count": 2},
            {"category": "made_up", "description": "x", "example_wrong": "foo",
             "example_correct": "bar", "explanation": "", "count": 0},
            {"category": "articles", "description": "no quote", "example_wrong": "",
             "example_correct": "", "explanation": "", "count": 1},
        ],
        "vocabulary": [
            {"term": " Follow Up On ", "kind": "phrasal_verb", "part_of_speech": "verb", "level": "b2",
             "meaning": "check later", "workplace_example": "I'll follow up on this.",
             "context_sentence": "I will follow up on mail", "source": "learner_spontaneous",
             "introduced_by_tutor": False, "used_correctly": True, "correction": ""},
            {"term": "follow up on", "kind": "phrase", "part_of_speech": "", "level": "B2",
             "meaning": "", "workplace_example": "", "context_sentence": "",
             "source": "tutor_only", "introduced_by_tutor": True, "used_correctly": True, "correction": ""},
        ],
        "strengths": [{"area": "fluency", "quote": "I handled the client call alone", "note": "Long answer"}],
        "learner_goals": [{"description": "Lead client calls", "status": "bogus"}],
        "lesson_topics": ["Project updates", " "],
        "skills_covered": [],
        "estimated_level": "b1",
        "session_summary": "You practised project updates.",
    }
    r = _parse_extraction(data)

    assert [e.category for e in r.grammar_errors] == ["verb_tense", "other"]
    assert r.grammar_errors[1].count == 1                 # clamped to ≥1
    assert len(r.vocabulary) == 1                          # duplicate term dropped
    v = r.vocabulary[0]
    assert v.term == "follow up on" and v.level == "B2" and v.mastery == "spontaneous"
    assert r.learner_goals[0].status == "mentioned"
    assert r.lesson_topics == ["Project updates"]
    assert r.estimated_level == "B1"


def test_schema_is_strict_compatible():
    def check(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        if node.get("type") == "array":
            check(node["items"])
    check(_json_schema())


def test_extraction_raises_instead_of_returning_empty():
    with patch("knowledge_graph.llm.extractor._chat_json", side_effect=ExtractionError("boom")):
        with pytest.raises(ExtractionError):
            extract_from_transcript("LEARNER: hello")


def test_legacy_pattern_names_map_to_fixed_categories():
    assert normalize_grammar_category("incorrect_past_tense_usage") == "verb_tense"
    assert normalize_grammar_category("article_usage") == "articles"
    assert normalize_grammar_category("Subject-Verb Agreement") == "subject_verb_agreement"
    assert normalize_grammar_category("something odd") == "other"
