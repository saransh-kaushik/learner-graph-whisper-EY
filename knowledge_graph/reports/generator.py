"""
knowledge_graph/reports/generator.py
─────────────────────────────────────
Generates a learner-facing progress report by:
  1. Aggregating data from Neo4j (aggregator.py) — all numbers computed in code
  2. Asking the LLM to turn those facts into a short, structured narrative
  3. Caching the result until the learner has a new session
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, Tuple

from openai import BadRequestError, OpenAI

from knowledge_graph.config import settings
from knowledge_graph.reports.aggregator import aggregate_learner_progress

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None
_cache: Dict[Tuple, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 200


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key, max_retries=3, timeout=90)
    return _client


_SYSTEM_PROMPT = """You write the narrative part of an English-speaking progress report for EnglishYaari.

THE READER
A working professional in India (engineer, manager, analyst, sales, etc.) who takes
one-to-one spoken English sessions to perform better at work: meetings, client calls,
interviews, presentations. They are busy and will spend about two minutes on this report.
They want to know, quickly: what got better, what still needs work, and exactly what to
practise next. Write for that person.

TONE
- Professional, warm and direct — like a good coach, not a school teacher.
- Second person ("you"). Plain English. Short sentences.
- No analytics jargon: never say "talk-time ratio", "code-switch ratio", "WPM",
  "CEFR", "baseline", "metric", "trend" or "data". Use the plain labels given in the data.
- No exaggeration, no exclamation marks, no emojis, no filler praise ("Great job!").

ACCURACY RULES (critical)
- Use ONLY facts in the JSON you receive. Never invent numbers, words, quotes or events.
- Call something improved only if the data marks it improved (metric trend "improving",
  grammar status "resolved"/"fading", or words in now_independent / new_words).
- Call something a problem only if the data marks it (metric trend "declining",
  grammar in needs_work / new / top_priority_errors, words_to_fix).
- If trends are "insufficient_data" (fewer than 3 sessions), do not talk about change
  over time; describe what you saw in these sessions instead.
- If data_quality lists sessions without usable speech, mention it once, plainly,
  in the summary (e.g. "1 of 5 recordings had no clear speech from you, so it was skipped").
- Quotes in "evidence", "your_sentence" and "better_sentence" must be copied exactly
  from the examples, strengths or vocabulary sentences provided.

WHAT TO WRITE
- headline: one line (max 12 words) capturing the single most important takeaway.
- summary: 2–3 sentences. Lead with a genuine achievement, then the main thing to work on.
- wins: 1–3 items. Each: a short title, one sentence of detail, and evidence (a quote or a
  concrete fact from the data). Prefer improvements over time; if there is no history,
  use strengths and well-used new words.
- focus_areas: 1–2 items from top_priority_errors (or words_to_fix). Each: title in plain
  words, why_it_matters at work (one sentence), the learner's own sentence, the corrected
  sentence, and a one-line tip they can remember.
- practice_plan: exactly 3 tasks for the next 7 days, 5–10 minutes each, each tied to a
  real work situation (stand-up update, client call, email follow-up, interview answer,
  presentation opening). At least one task must reuse the learner's own sentences from
  focus_areas; at least one should use a word from new_words or words_to_try. Steps must
  be concrete enough to do without a tutor.
- next_milestone: one sentence describing a realistic, observable goal for the next
  2–3 sessions.
"""


def _report_schema() -> dict:
    s = {"type": "string"}

    def obj(props: dict) -> dict:
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    return obj({
        "headline": s,
        "summary": s,
        "wins": {"type": "array", "items": obj({"title": s, "detail": s, "evidence": s})},
        "focus_areas": {"type": "array", "items": obj({
            "title": s, "why_it_matters": s, "your_sentence": s, "better_sentence": s, "tip": s,
        })},
        "practice_plan": {"type": "array", "items": obj({
            "title": s, "minutes": {"type": "integer"}, "situation": s,
            "steps": {"type": "array", "items": s},
        })},
        "next_milestone": s,
    })


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    learner_id: str,
    n_sessions: Optional[int] = None,
    *,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Generate (or return cached) progress report for *learner_id*."""
    data = aggregate_learner_progress(learner_id, n_sessions)

    latest = next((s["session_id"] for s in data.get("sessions", [])), None)
    key = (learner_id, n_sessions, latest, data["sessions_analyzed"]["valid"], data["cefr_level"].get("value"))
    if not refresh:
        with _cache_lock:
            cached = _cache.get(key)
        if cached:
            return cached

    if data["sessions_analyzed"]["valid"] == 0:
        narrative = _empty_narrative(data)
    else:
        narrative = _generate_narrative(data)

    report = {
        **data,
        "report_date": date.today().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "narrative": narrative,
    }
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = report
    return report


# ── Narrative ─────────────────────────────────────────────────────────────────

def _first_name(name: str) -> str:
    return (name or "").split()[0] if name else ""


def _slim_pattern(p: dict) -> dict:
    return {
        "label": p["label"],
        "status": p["status"],
        "recent_rate": p["recent_rate"],
        "earlier_rate": p["earlier_rate"],
        "rate_unit": p["unit"],
        "examples": p["examples"][:2],
    }


def _slim_word(w: dict) -> dict:
    return {k: w.get(k) for k in ("word", "level_hint", "meaning", "your_sentence", "correction", "tutor_sentence") if w.get(k)}


def build_prompt_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, pre-digested facts for the LLM (no raw histories)."""
    metrics = {
        m["label"]: {
            "current": m["current"], "unit": m["unit"], "trend": m["trend"],
            "change_pct": m["change_pct"], "what_it_means": m["help"],
        }
        for m in data.get("metrics", {}).values() if m.get("current") is not None
    }
    vocab = data.get("vocabulary", {})
    grammar = data.get("grammar", {})
    return {
        "learner_first_name": _first_name(data.get("learner_name", "")),
        "level": data.get("cefr_level"),
        "period": data.get("period"),
        "sessions_analyzed": data.get("sessions_analyzed"),
        "data_quality": data.get("data_quality"),
        "metrics": metrics,
        "grammar_improved": [_slim_pattern(p) for p in grammar.get("improved", [])[:3]],
        "top_priority_errors": [_slim_pattern(p) for p in data.get("top_priority_errors", [])],
        "strengths": data.get("strengths", [])[:4],
        "new_words": [_slim_word(w) for w in vocab.get("new_words", [])[:8]],
        "now_independent": [_slim_word(w) for w in vocab.get("now_independent", [])[:5]],
        "words_to_fix": [_slim_word(w) for w in vocab.get("words_to_fix", [])[:3]],
        "words_to_try": [_slim_word(w) for w in vocab.get("words_to_try", [])[:5]],
        "learner_goals": data.get("learner_goals", [])[:5],
        "lesson_topics": [t["topic"] for t in data.get("lesson_topics", [])[:6]],
        "recent_session_summaries": [
            {"date": s["date"], "summary": s["summary"]}
            for s in data.get("sessions", [])[:3] if s.get("summary")
        ],
    }


def _generate_narrative(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = build_prompt_payload(data)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=1, ensure_ascii=False)},
    ]
    try:
        client = _get_client()
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "progress_report", "strict": True, "schema": _report_schema(),
                }},
                messages=messages,
                temperature=0.3,
            )
        except BadRequestError:
            response = client.chat.completions.create(
                model=settings.openai_model,
                response_format={"type": "json_object"},
                messages=messages + [{"role": "system", "content":
                    "Return ONLY JSON matching this schema:\n" + json.dumps(_report_schema())}],
                temperature=0.3,
            )
        narrative = json.loads(response.choices[0].message.content or "{}")
        narrative = _sanitize(narrative, data)
        narrative["source"] = "ai"
        return narrative
    except Exception as exc:
        logger.error("LLM report generation failed, using fallback: %s", exc)
        return _fallback_narrative(data)


def _known_quotes(data: Dict[str, Any]) -> set:
    quotes = set()
    for p in data.get("top_priority_errors", []) + sum(data.get("grammar", {}).values(), []):
        for ex in p.get("examples", []):
            for k in ("wrong", "correct"):
                if ex.get(k):
                    quotes.add(ex[k].strip().lower())
    for w in data.get("vocabulary", {}).get("words_to_fix", []):
        for k in ("your_sentence", "correction"):
            if w.get(k):
                quotes.add(w[k].strip().lower())
    return quotes


def _sanitize(narrative: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce limits and make sure focus-area sentences are the learner's real ones."""
    narrative.setdefault("headline", "")
    narrative.setdefault("summary", "")
    narrative["wins"] = list(narrative.get("wins") or [])[:3]
    narrative["practice_plan"] = list(narrative.get("practice_plan") or [])[:3]
    narrative.setdefault("next_milestone", "")

    quotes = _known_quotes(data)
    priority = data.get("top_priority_errors", [])
    focus = []
    for i, fa in enumerate(list(narrative.get("focus_areas") or [])[:2]):
        if (fa.get("your_sentence") or "").strip().lower() not in quotes:
            # Replace an unverifiable quote with the real example for that priority error
            ex = next(iter(priority[i]["examples"]), None) if i < len(priority) else None
            if not ex:
                continue
            fa["your_sentence"] = ex["wrong"]
            fa["better_sentence"] = ex.get("correct") or fa.get("better_sentence", "")
        focus.append(fa)
    narrative["focus_areas"] = focus
    return narrative


def _fallback_narrative(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic narrative used when the LLM is unavailable."""
    sa = data["sessions_analyzed"]
    improved = data.get("grammar", {}).get("improved", [])
    priority = data.get("top_priority_errors", [])
    new_words = data.get("vocabulary", {}).get("new_words", [])

    wins = []
    for p in improved[:2]:
        wins.append({
            "title": f"Fewer mistakes with {p['label'].lower()}",
            "detail": "This came up less often in your recent sessions than before.",
            "evidence": f"{p['earlier_rate']} → {p['recent_rate']} {p['unit']}",
        })
    if new_words:
        words = ", ".join(w["word"] for w in new_words[:3])
        wins.append({"title": "New words in use", "detail": f"You used {words} correctly.", "evidence": words})
    for s in data.get("strengths", [])[: max(0, 2 - len(wins))]:
        wins.append({"title": f"Strong {s['area']}", "detail": s.get("note", ""), "evidence": s["quote"]})

    focus = []
    for p in priority[:2]:
        ex = p["examples"][0] if p["examples"] else {}
        focus.append({
            "title": p["label"],
            "why_it_matters": "Getting this right makes you sound clearer and more confident at work.",
            "your_sentence": ex.get("wrong", ""),
            "better_sentence": ex.get("correct") or "",
            "tip": ex.get("explanation") or "",
        })

    plan = []
    if focus:
        plan.append({
            "title": f"Fix it in your own words: {focus[0]['title'].lower()}",
            "minutes": 5,
            "situation": "Daily stand-up update",
            "steps": [
                f"Say the corrected sentence aloud three times: \"{focus[0]['better_sentence']}\"",
                "Record a 1-minute update about yesterday's work using the same pattern.",
                "Listen back and note any slips.",
            ],
        })
    if new_words:
        plan.append({
            "title": "Use your new words",
            "minutes": 5,
            "situation": "Email or chat with a colleague",
            "steps": [f"Write two work sentences using \"{w['word']}\"." for w in new_words[:2]],
        })
    plan.append({
        "title": "Speak for two minutes without stopping",
        "minutes": 10,
        "situation": "Explaining your role to a new client",
        "steps": [
            "Set a 2-minute timer and describe what you do and why it matters.",
            "Avoid switching to Hindi; if you get stuck, rephrase in simpler English.",
            "Repeat once, aiming for fewer pauses.",
        ],
    })

    return {
        "headline": "Here's where your spoken English stands this period",
        "summary": (
            f"This report covers {sa['valid']} session{'s' if sa['valid'] != 1 else ''}"
            + (f" ({sa['total'] - sa['valid']} skipped: no clear speech from you)." if sa["total"] > sa["valid"] else ".")
            + (f" Your main focus next is {priority[0]['label'].lower()}." if priority else "")
        ),
        "wins": wins[:3],
        "focus_areas": focus,
        "practice_plan": plan[:3],
        "next_milestone": (
            f"Use {priority[0]['label'].lower()} correctly in your next two sessions."
            if priority else "Keep speaking in longer, complete answers in your next sessions."
        ),
        "source": "fallback",
    }


def _empty_narrative(data: Dict[str, Any]) -> Dict[str, Any]:
    total = data["sessions_analyzed"]["total"]
    return {
        "headline": "Your report will appear after your first analysed session",
        "summary": (
            f"We found {total} recording{'s' if total != 1 else ''}, but none had clear speech from you to analyse."
            if total else "No sessions have been analysed yet."
        ),
        "wins": [],
        "focus_areas": [],
        "practice_plan": [],
        "next_milestone": "Complete a session where you speak for at least a few minutes.",
        "source": "empty",
    }
