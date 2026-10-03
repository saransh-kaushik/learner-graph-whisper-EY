"""
knowledge_graph/reports/generator.py
─────────────────────────────────────
Generates a structured progress report for a learner by:
  1. Aggregating data from Neo4j (aggregator.py)
  2. Calling GPT-4o to produce a natural-language summary
  3. Returning the full report as a typed dict
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Dict, Optional

from openai import OpenAI

from knowledge_graph.config import settings
from knowledge_graph.reports.aggregator import aggregate_learner_progress

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client

_SYSTEM_PROMPT = """You are an expert English language tutor coach.
You will receive structured data about a learner's recent session metrics,
vocabulary mastery, grammar errors, and goals. 

Write:
1. "llm_summary": a 3-4 sentence, warm, encouraging progress summary.
   Rules for summary:
   - Must be grounded ONLY in the computed metrics and flags.
   - Do not speculate about causes (e.g. do not guess about technical issues unless explicitly in data_quality flags).
   - If data_quality flags exist, say so plainly (e.g., "X of Y sessions had no usable learner speech data, so trends are not shown").
   - Do not call something "improving" or "declining" unless the trend explicitly says so (i.e. not "insufficient_data").
   - Address the learner directly, warm and simple (beginner-friendly, no analytics jargon).
   - Lead with something genuinely achieved.

2. "practice_next": a single specific, actionable practice recommendation.
   Rules for practice_next:
   - Target the #1 priority error from the top_priority_errors list.
   - Reuse the learner's own real sentences from the examples as material (not a generic "tell a story" task).
   - Give one concrete 5-minute exercise based on their examples.

Return ONLY valid JSON with exactly these two keys.
"""

def generate_report(
    learner_id: str,
    n_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a full progress report for *learner_id*.
    """
    data = aggregate_learner_progress(learner_id, n_sessions)

    if data["sessions_analyzed"]["valid"] == 0:
        return {
            **data,
            "report_date": date.today().isoformat(),
            "llm_summary": "No valid session data found for this learner yet.",
            "practice_next": "Complete your first session with speaking exercises to see recommendations.",
        }

    llm_summary, practice_next = _generate_llm_summary(data)

    return {
        "learner_id": data["learner_id"],
        "learner_name": data["learner_name"],
        "cefr_level": data["cefr_level"],
        "report_date": date.today().isoformat(),
        "sessions_analyzed": data["sessions_analyzed"],
        "data_quality": data["data_quality"],
        "metrics": data["metrics"],
        "vocabulary": data["vocabulary"],
        "grammar": data["grammar"],
        "top_priority_errors": data["top_priority_errors"],
        "learner_goals": data["learner_goals"],
        "lesson_topics": data["lesson_topics"],
        "session_summaries": data["session_summaries"],
        "llm_summary": llm_summary,
        "practice_next": practice_next,
    }

def _generate_llm_summary(data: Dict[str, Any]) -> tuple[str, str]:
    prompt_data = {
        "learner_name": data["learner_name"],
        "sessions_analyzed": data["sessions_analyzed"],
        "data_quality": data["data_quality"],
        "metrics": data["metrics"],
        "new_advanced_words": data["vocabulary"].get("learner_new_advanced_words", [])[:5],
        "top_priority_errors": data["top_priority_errors"],
        "learner_goals": data["learner_goals"],
    }

    try:
        response = _get_client().chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt_data, indent=2)},
            ],
            temperature=0.7,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        return (
            result.get("llm_summary", "Progress data collected."),
            result.get("practice_next", "Continue practicing regularly."),
        )
    except Exception as exc:
        logger.error("LLM summary generation failed: %s", exc)
        return "Progress data collected successfully.", "Continue practicing regularly."
