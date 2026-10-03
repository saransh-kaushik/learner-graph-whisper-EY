"""
knowledge_graph/llm/extractor.py
─────────────────────────────────
OpenAI-powered NLP extraction from diarized transcripts.

Extracts (via structured JSON output):
  • grammar_errors     — list of pattern + examples
  • vocabulary         — words used/introduced with context
  • goals_discussed    — goals and their status
  • skills_covered     — skills practiced in the session
  • session_summary    — 2-3 sentence plain-text summary
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from openai import OpenAI

from knowledge_graph.config import settings

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class GrammarError:
    pattern: str                # short name, e.g. "article_usage"
    description: str
    example_wrong: str
    example_correct: str
    count: int = 1


@dataclass
class VocabItem:
    word: str
    context_sentence: str
    introduced_by_tutor: bool = False
    used_by_learner: bool = False
    correct: bool = True
    spontaneous: bool = False


@dataclass
class GoalItem:
    description: str
    status: Literal["mentioned", "practiced", "demonstrated"] = "mentioned"


@dataclass
class SkillItem:
    name: str
    category: str   # speaking | vocabulary | grammar | fluency | listening


@dataclass
class ExtractionResult:
    grammar_errors: List[GrammarError] = field(default_factory=list)
    vocabulary: List[VocabItem] = field(default_factory=list)
    goals_discussed: List[GoalItem] = field(default_factory=list)
    skills_covered: List[SkillItem] = field(default_factory=list)
    session_summary: str = ""


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert English language tutor analyst.
You will receive a diarized transcript of an English tutoring session
(learner vs tutor turns labelled). Analyse it carefully and return a
JSON object with exactly these keys:

{
  "grammar_errors": [
    {
      "pattern": "<short_snake_case_name>",
      "description": "<what the error pattern is>",
      "example_wrong": "<learner's actual wrong utterance>",
      "example_correct": "<corrected version>",
      "count": <integer occurrences>
    }
  ],
  "vocabulary": [
    {
      "word": "<base lemma form>",
      "context_sentence": "<sentence where word appeared>",
      "introduced_by_tutor": <true|false>,
      "used_by_learner": <true|false>,
      "correct": <true|false>,
      "spontaneous": <true|false — true if learner used without tutor prompting>
    }
  ],
  "goals_discussed": [
    {
      "description": "<goal description>",
      "status": "<mentioned|practiced|demonstrated>"
    }
  ],
  "skills_covered": [
    {
      "name": "<skill name, e.g. 'present perfect tense'>",
      "category": "<speaking|vocabulary|grammar|fluency|listening>"
    }
  ],
  "session_summary": "<2-3 sentence plain-English summary of the session>"
}

Rules:
- Only include vocabulary words that are B1-C2 level or were explicitly
  focused on by the tutor.
- "demonstrated" status means the learner used the goal skill correctly
  and spontaneously without prompting.
- Return ONLY valid JSON, no prose before or after.
"""

_USER_TEMPLATE = """SESSION TRANSCRIPT
==================
{transcript}

LEARNER LABEL: {learner_label}
TUTOR LABEL:   {tutor_label}
"""


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_from_transcript(
    transcript_text: str,
    learner_label: str = "Learner",
    tutor_label: str = "Tutor",
) -> ExtractionResult:
    """
    Call GPT-4o to extract grammar errors, vocabulary, goals and skills
    from a formatted transcript string.

    Parameters
    ----------
    transcript_text:
        Pre-formatted string with speaker labels, e.g.
        "LEARNER: I go to market yesterday.\nTUTOR: You should say 'went'."
    """
    prompt = _USER_TEMPLATE.format(
        transcript=transcript_text,
        learner_label=learner_label,
        tutor_label=tutor_label,
    )

    logger.info("Calling OpenAI extraction (model=%s)…", settings.openai_model)
    response = _get_client().chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    raw_json = response.choices[0].message.content or "{}"
    logger.debug("Raw extraction JSON: %s", raw_json[:300])

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse OpenAI JSON response: %s", exc)
        return ExtractionResult()

    return _parse_extraction(data)


def _parse_extraction(data: dict) -> ExtractionResult:
    result = ExtractionResult()

    for item in data.get("grammar_errors", []):
        try:
            result.grammar_errors.append(GrammarError(
                pattern=item.get("pattern", "unknown"),
                description=item.get("description", ""),
                example_wrong=item.get("example_wrong", ""),
                example_correct=item.get("example_correct", ""),
                count=int(item.get("count", 1)),
            ))
        except Exception as exc:
            logger.warning("Skipping grammar_error item: %s", exc)

    for item in data.get("vocabulary", []):
        try:
            result.vocabulary.append(VocabItem(
                word=item.get("word", "").lower().strip(),
                context_sentence=item.get("context_sentence", ""),
                introduced_by_tutor=bool(item.get("introduced_by_tutor", False)),
                used_by_learner=bool(item.get("used_by_learner", False)),
                correct=bool(item.get("correct", True)),
                spontaneous=bool(item.get("spontaneous", False)),
            ))
        except Exception as exc:
            logger.warning("Skipping vocabulary item: %s", exc)

    for item in data.get("goals_discussed", []):
        try:
            status = item.get("status", "mentioned")
            if status not in ("mentioned", "practiced", "demonstrated"):
                status = "mentioned"
            result.goals_discussed.append(GoalItem(
                description=item.get("description", ""),
                status=status,  # type: ignore[arg-type]
            ))
        except Exception as exc:
            logger.warning("Skipping goal item: %s", exc)

    for item in data.get("skills_covered", []):
        try:
            result.skills_covered.append(SkillItem(
                name=item.get("name", ""),
                category=item.get("category", "speaking"),
            ))
        except Exception as exc:
            logger.warning("Skipping skill item: %s", exc)

    result.session_summary = data.get("session_summary", "")
    return result
