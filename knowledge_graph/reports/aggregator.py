"""
knowledge_graph/reports/aggregator.py
───────────────────────────────────────
Queries Neo4j to aggregate a learner's progress across their last N sessions.

Everything numeric in the report (trends, counts, what improved) is computed
here, deterministically. The LLM in generator.py only turns it into prose.
"""
from __future__ import annotations

import json
import logging
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence

from knowledge_graph.config import settings
from knowledge_graph.db import neo4j_client
from knowledge_graph.taxonomy import (
    CEFR_LEVELS,
    CEFR_RANK,
    METRIC_INFO,
    WORD_LEVEL_HINTS,
    describe_level,
    grammar_label,
    normalize_grammar_category,
)

logger = logging.getLogger(__name__)

MIN_SESSIONS_FOR_TREND = 3
_REL_THRESHOLD = 0.10          # ±10% relative change counts as a real change
_ABS_THRESHOLD = {"%": 0.02, "words": 0.5, "wpm": 5.0}
_BASIC_LEVELS = ("A1", "A2")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _loads(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _d(row: dict) -> str:
    """Comparable ISO date of a row ('' when missing)."""
    return row.get("date") or ""


def _is_basic_word(kind: Optional[str], level: Optional[str]) -> bool:
    return (kind in (None, "word")) and level in _BASIC_LEVELS


# ── Entry point ───────────────────────────────────────────────────────────────

def aggregate_learner_progress(
    learner_id: str,
    n_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    n = max(1, n_sessions or settings.rolling_baseline_sessions)

    learner_rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $id})
        RETURN l.name AS name, l.current_cefr_level AS level, l.native_language AS native_language
        """,
        id=learner_id,
    )
    learner = dict(learner_rows[0]) if learner_rows else {}
    learner_name = learner.get("name") or learner_id

    session_rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[:ATTENDED]->(s:Session)
        RETURN s.id AS id, s.date AS date,
               s.talk_time_ratio            AS talk_time_ratio,
               s.words_per_turn             AS words_per_turn,
               s.average_words_per_sentence AS average_words_per_sentence,
               s.speaking_rate_wpm          AS speaking_rate_wpm,
               s.code_switch_ratio          AS code_switch_ratio,
               s.duration_sec               AS duration_sec,
               s.learner_talk_sec           AS learner_talk_sec,
               s.learner_turn_count         AS learner_turn_count,
               s.total_learner_words        AS total_learner_words,
               s.estimated_level            AS estimated_level,
               s.lesson_topics              AS lesson_topics,
               s.strengths_json             AS strengths_json,
               s.transcript_summary         AS summary
        ORDER BY s.date DESC, s.ingested_at DESC
        LIMIT $n
        """,
        lid=learner_id,
        n=n,
    )

    sessions = _prepare_sessions(session_rows)
    valid = [s for s in sessions if s["is_valid"]]
    data_quality = [
        {"session_id": s["id"], "date": s["date"], "reasons": s["issues"]}
        for s in sessions if not s["is_valid"]
    ]

    base = {
        "learner_id": learner_id,
        "learner_name": learner_name,
        "native_language": learner.get("native_language"),
        "cefr_level": _cefr_level(learner.get("level"), valid),
        "period": _period(sessions, valid),
        "sessions_analyzed": {"valid": len(valid), "total": len(sessions)},
        "data_quality": data_quality,
        "sessions": [_session_public(s) for s in reversed(sessions)],  # newest first
    }

    if not valid:
        return {
            **base,
            "metrics": {},
            "vocabulary": _empty_vocabulary(),
            "grammar": {"improved": [], "needs_work": [], "new": []},
            "top_priority_errors": [],
            "strengths": [],
            "learner_goals": _aggregate_goals(learner_id),
            "lesson_topics": [],
        }

    valid_ids = [s["id"] for s in valid]

    grammar_rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[r:MADE_ERROR]->(gp:GrammarPattern)
        WHERE r.session_id IN $sids
        RETURN gp.pattern_name AS pattern,
               gp.example_wrong AS gp_wrong, gp.example_correct AS gp_correct,
               r.error_count AS count, r.session_id AS session_id, r.date AS date,
               r.example_utterance AS wrong, r.example_correct AS correct,
               r.explanation AS explanation, r.examples_json AS examples_json
        """,
        lid=learner_id,
        sids=valid_ids,
    )
    used_rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[r:USED_WORD]->(w:Word)
        RETURN w.lemma AS word, w.cefr_level AS level, w.kind AS kind,
               w.meaning AS meaning, w.workplace_example AS example,
               r.mastery AS mastery, r.correct AS correct, r.correction AS correction,
               r.context_sentence AS context, r.session_id AS session_id, r.date AS date
        ORDER BY r.date ASC
        """,
        lid=learner_id,
    )
    intro_rows = neo4j_client.run_query(
        """
        MATCH (:Learner {id: $lid})-[:ATTENDED]->(s:Session)-[r:INTRODUCED_WORD]->(w:Word)
        RETURN w.lemma AS word, w.cefr_level AS level, w.kind AS kind,
               w.meaning AS meaning, w.workplace_example AS example,
               s.id AS session_id, s.date AS date, r.context_sentence AS context
        ORDER BY s.date ASC
        """,
        lid=learner_id,
    )

    grammar, top_priority = build_grammar(grammar_rows, valid)

    return {
        **base,
        "metrics": compute_metric_trends(sessions),
        "vocabulary": build_vocabulary(used_rows, intro_rows, valid),
        "grammar": grammar,
        "top_priority_errors": top_priority,
        "strengths": _strengths(valid),
        "learner_goals": _aggregate_goals(learner_id),
        "lesson_topics": _lesson_topics(valid),
    }


# ── Sessions ──────────────────────────────────────────────────────────────────

def _prepare_sessions(rows: Iterable[Any]) -> List[dict]:
    sessions = []
    for r in rows:
        s = dict(r)
        s["date"] = _iso(s.get("date"))
        issues = []
        no_ratio = not s.get("talk_time_ratio")
        no_rate = not s.get("speaking_rate_wpm")
        if no_ratio and no_rate:
            issues.append("no_learner_speech")
        s["issues"] = issues
        s["is_valid"] = not issues
        if s.get("learner_talk_sec") is None and s.get("talk_time_ratio") and s.get("duration_sec"):
            # legacy sessions: approximate learner speaking time
            s["learner_talk_sec"] = s["talk_time_ratio"] * s["duration_sec"]
        sessions.append(s)
    return list(reversed(sessions))  # chronological


def _session_public(s: dict) -> dict:
    return {
        "session_id": s["id"],
        "date": s["date"],
        "is_valid": s["is_valid"],
        "duration_min": round((s.get("duration_sec") or 0) / 60, 1),
        "learner_minutes": round((s.get("learner_talk_sec") or 0) / 60, 1) if s["is_valid"] else None,
        "summary": s.get("summary") or "",
        "topics": list(s.get("lesson_topics") or []),
    }


def _period(sessions: List[dict], valid: List[dict]) -> dict:
    return {
        "start": sessions[0]["date"] if sessions else None,
        "end": sessions[-1]["date"] if sessions else None,
        "learner_minutes": round(sum((s.get("learner_talk_sec") or 0) for s in valid) / 60, 1),
        "total_minutes": round(sum((s.get("duration_sec") or 0) for s in sessions) / 60, 1),
        "learner_words": sum((s.get("total_learner_words") or 0) for s in valid) or None,
    }


def _cefr_level(set_level: Optional[str], valid: List[dict]) -> dict:
    if set_level in CEFR_LEVELS:
        return {**describe_level(set_level), "source": "tutor", "confidence": "high"}
    estimates = [s["estimated_level"] for s in valid if s.get("estimated_level") in CEFR_LEVELS]
    if not estimates:
        return {**describe_level(None), "source": None, "confidence": "insufficient_data"}
    ranks = sorted(CEFR_RANK[e] for e in estimates)
    level = CEFR_LEVELS[int(median(ranks[-3:])) - 1]   # recent-weighted: last 3 sessions
    confidence = "high" if len(estimates) >= 4 else ("medium" if len(estimates) >= 2 else "low")
    return {**describe_level(level), "source": "estimated", "confidence": confidence}


# ── Metrics ───────────────────────────────────────────────────────────────────

def metric_trend(name: str, values: Sequence[float]) -> dict:
    """Trend of one metric over chronological valid values: recent half vs earlier half."""
    info = METRIC_INFO[name]
    out = {
        "label": info["label"],
        "help": info["help"],
        "unit": info["unit"],
        "better": info["better"],
        "current": round(values[-1], 4) if values else None,
        "baseline": None,
        "recent": None,
        "change_pct": None,
        "trend": "insufficient_data",
    }
    if len(values) < MIN_SESSIONS_FOR_TREND:
        return out

    split = len(values) // 2
    earlier, recent = mean(values[:split]), mean(values[split:])
    out["baseline"], out["recent"] = round(earlier, 4), round(recent, 4)
    delta = recent - earlier
    if earlier:
        out["change_pct"] = round(delta / earlier * 100, 1)

    if info["better"] == "band":
        lo, hi = info["band"]

        def dist(v: float) -> float:
            return 0.0 if lo <= v <= hi else min(abs(v - lo), abs(v - hi))

        de, dr = dist(earlier), dist(recent)
        threshold = _ABS_THRESHOLD.get(info["unit"], 0)
        if dr < de - threshold:
            out["trend"] = "improving"
        elif dr > de + threshold:
            out["trend"] = "declining"
        else:
            out["trend"] = "stable"
        return out

    rel = abs(delta / earlier) if earlier else (1.0 if delta else 0.0)
    if rel < _REL_THRESHOLD or abs(delta) < _ABS_THRESHOLD.get(info["unit"], 0):
        out["trend"] = "stable"
    elif (delta > 0) == (info["better"] == "higher"):
        out["trend"] = "improving"
    else:
        out["trend"] = "declining"
    return out


def compute_metric_trends(sessions: List[dict]) -> Dict[str, dict]:
    result = {}
    for name in METRIC_INFO:
        history, values = [], []
        for s in sessions:
            val = s.get(name) if s["is_valid"] else None
            if val is not None:
                values.append(val)
            history.append({
                "session_id": s["id"],
                "date": s["date"],
                "value": round(val, 4) if val is not None else None,
            })
        result[name] = {**metric_trend(name, values), "history": history}
    return result


# ── Grammar ───────────────────────────────────────────────────────────────────

def classify_error_series(series: Sequence[float]) -> str:
    """
    Classify one error category's per-session rates (chronological, zero-filled).
      new         — only appears in the most recent session(s)
      resolved    — appeared earlier, absent recently
      fading      — recent rate ≤ 70% of earlier rate
      increasing  — recent rate ≥ 130% of earlier rate
      persistent  — otherwise
    """
    n = len(series)
    if n <= 1:
        return "new"
    k = 2 if n >= 4 else 1
    earlier, recent = series[:-k], series[-k:]
    if sum(earlier) == 0:
        return "new"
    if sum(recent) == 0:
        return "resolved"
    e, r = mean(earlier), mean(recent)
    if r <= 0.7 * e:
        return "fading"
    if r >= 1.3 * e:
        return "increasing"
    return "persistent"


def _session_word_count(s: dict) -> Optional[float]:
    if s.get("total_learner_words"):
        return float(s["total_learner_words"])
    if s.get("words_per_turn") and s.get("learner_turn_count"):
        return float(s["words_per_turn"] * s["learner_turn_count"])
    return None


def _row_examples(r: dict) -> List[dict]:
    examples = _loads(r.get("examples_json"), None)
    if examples:
        return [
            {"wrong": e.get("wrong", ""), "correct": e.get("correct") or None,
             "explanation": e.get("explanation") or None}
            for e in examples if e.get("wrong")
        ]
    wrong = r.get("wrong")
    if not wrong:
        return []
    correct = r.get("correct")
    if not correct and r.get("gp_wrong") == wrong:
        # legacy rows: the pattern's correction only matches its own example
        correct = r.get("gp_correct")
    return [{"wrong": wrong, "correct": correct or None, "explanation": r.get("explanation") or None}]


def build_grammar(rows: Iterable[Any], valid_sessions: List[dict]) -> tuple[dict, list]:
    order = [s["id"] for s in valid_sessions]
    dates = {s["id"]: s["date"] for s in valid_sessions}
    words = {s["id"]: _session_word_count(s) for s in valid_sessions}
    normalized = all(words.values())

    counts: Dict[str, Dict[str, int]] = {}
    examples: Dict[str, List[dict]] = {}
    for row in rows:
        r = dict(row)
        sid = r["session_id"]
        if sid not in dates:
            continue
        cat = normalize_grammar_category(r.get("pattern"))
        counts.setdefault(cat, {})
        counts[cat][sid] = counts[cat].get(sid, 0) + int(r.get("count") or 0)
        for ex in _row_examples(r):
            examples.setdefault(cat, []).append({**ex, "date": dates[sid], "_order": order.index(sid)})

    patterns = []
    for cat, per_session in counts.items():
        series_counts = [per_session.get(sid, 0) for sid in order]
        series = [
            round(c / words[sid] * 100, 2) if normalized else c
            for c, sid in zip(series_counts, order)
        ]
        status = classify_error_series(series)
        k = 2 if len(series) >= 4 else 1
        seen, picked = set(), []
        for ex in sorted(examples.get(cat, []), key=lambda e: -e["_order"]):
            key = ex["wrong"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append({k_: v for k_, v in ex.items() if k_ != "_order"})
            if len(picked) == 3:
                break
        patterns.append({
            "pattern": cat,
            "label": grammar_label(cat),
            "status": status,
            "unit": "per 100 words" if normalized else "per session",
            "series": [{"date": dates[sid], "value": v} for sid, v in zip(order, series)],
            "total_count": sum(series_counts),
            "sessions_with_error": sum(1 for c in series_counts if c),
            "latest_count": series_counts[-1],
            "recent_rate": round(mean(series[-k:]), 2),
            "earlier_rate": round(mean(series[:-k]), 2) if len(series) > k else None,
            "examples": picked,
        })

    def by_rate(p: dict) -> tuple:
        return (-p["recent_rate"], -p["total_count"])

    improved = sorted(
        (p for p in patterns if p["status"] in ("resolved", "fading")),
        key=lambda p: -((p["earlier_rate"] or 0) - p["recent_rate"]),
    )
    needs_work = sorted((p for p in patterns if p["status"] in ("persistent", "increasing")), key=by_rate)
    new = sorted((p for p in patterns if p["status"] == "new"), key=by_rate)

    top_priority = sorted(
        (p for p in patterns if p["status"] in ("persistent", "increasing", "new") and p["latest_count"] > 0),
        key=by_rate,
    )[:3]
    return {"improved": improved, "needs_work": needs_work, "new": new}, top_priority


# ── Vocabulary ────────────────────────────────────────────────────────────────

def _empty_vocabulary() -> dict:
    return {
        "new_words": [], "now_independent": [], "words_to_fix": [], "words_to_try": [],
        "totals": {"new_words": 0, "now_independent": 0, "words_to_fix": 0, "words_to_try": 0},
    }


def _word_entry(r: dict, **extra: Any) -> dict:
    level = r.get("level") if r.get("level") in CEFR_LEVELS else None
    return {
        "word": r["word"],
        "kind": r.get("kind") or "word",
        "level": level,
        "level_hint": WORD_LEVEL_HINTS.get(level) if level else None,
        "meaning": r.get("meaning"),
        "example": r.get("example"),
        **extra,
    }


def build_vocabulary(used_rows: Iterable[Any], intro_rows: Iterable[Any], valid_sessions: List[dict]) -> dict:
    window = {s["id"] for s in valid_sessions}
    window_start = valid_sessions[0]["date"]

    usages: Dict[str, List[dict]] = {}
    for row in used_rows:
        r = dict(row)
        r["date"] = _iso(r.get("date"))
        usages.setdefault(r["word"], []).append(r)

    intros: Dict[str, List[dict]] = {}
    for row in intro_rows:
        r = dict(row)
        r["date"] = _iso(r.get("date"))
        intros.setdefault(r["word"], []).append(r)

    new_words, now_independent, words_to_fix = [], [], []

    for word, uses in usages.items():
        uses.sort(key=_d)
        in_window = [u for u in uses if u["session_id"] in window]
        if not in_window:
            continue
        sample = in_window[-1]
        if _is_basic_word(sample.get("kind"), sample.get("level")):
            continue

        good = [u for u in in_window if u.get("correct") is not False and u.get("mastery") in ("spontaneous", "prompted")]

        # Taught/prompted earlier → now used unprompted
        upgraded = None
        for u in in_window:
            if u.get("correct") is False or u.get("mastery") != "spontaneous":
                continue
            earlier_prompted = any(
                _d(p) < _d(u) and p.get("mastery") in ("prompted", "repeated") for p in uses
            )
            earlier_taught = any(_d(i) < _d(u) for i in intros.get(word, []))
            if earlier_prompted or earlier_taught:
                upgraded = u
                break

        if upgraded:
            now_independent.append(_word_entry(
                sample, your_sentence=upgraded.get("context"), date=upgraded["date"],
                times_used=len(in_window),
            ))
        elif good and uses[0]["session_id"] in window and _d(uses[0]) >= (window_start or ""):
            best = next((u for u in reversed(good) if u.get("mastery") == "spontaneous"), good[-1])
            new_words.append(_word_entry(
                sample, your_sentence=best.get("context"), date=good[0]["date"],
                mastery=best.get("mastery"), times_used=len(in_window),
            ))

        last = in_window[-1]
        later_correct = any(u.get("correct") is not False and _d(u) > _d(last) for u in uses)
        misuse = next((u for u in reversed(in_window) if u.get("correct") is False), None)
        if misuse and not later_correct and (last.get("correct") is False):
            words_to_fix.append(_word_entry(
                sample, your_sentence=misuse.get("context"), correction=misuse.get("correction"),
                date=misuse["date"],
            ))

    words_to_try = []
    for word, items in intros.items():
        recent = [i for i in items if i["session_id"] in window]
        if not recent:
            continue
        if any(u.get("correct") is not False for u in usages.get(word, [])):
            continue  # learner already uses it
        latest = recent[-1]
        if _is_basic_word(latest.get("kind"), latest.get("level")):
            continue
        words_to_try.append(_word_entry(latest, tutor_sentence=latest.get("context"), date=latest["date"]))

    # Highest level first (then newest) for new words; newest first elsewhere
    new_words.sort(key=lambda e: (CEFR_RANK.get(e["level"] or "", 0), e.get("date") or ""), reverse=True)
    for lst in (now_independent, words_to_fix, words_to_try):
        lst.sort(key=lambda e: e.get("date") or "", reverse=True)

    return {
        "new_words": new_words[:12],
        "now_independent": now_independent[:8],
        "words_to_fix": words_to_fix[:6],
        "words_to_try": words_to_try[:8],
        "totals": {
            "new_words": len(new_words),
            "now_independent": len(now_independent),
            "words_to_fix": len(words_to_fix),
            "words_to_try": len(words_to_try),
        },
    }


# ── Strengths, topics, goals ──────────────────────────────────────────────────

def _strengths(valid: List[dict]) -> List[dict]:
    out = []
    for s in reversed(valid):
        for item in _loads(s.get("strengths_json"), []):
            if item.get("quote"):
                out.append({**item, "date": s["date"]})
    return out[:6]


def _lesson_topics(valid: List[dict]) -> List[dict]:
    seen, out = set(), []
    for s in reversed(valid):
        for topic in s.get("lesson_topics") or []:
            key = topic.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append({"topic": topic.strip(), "date": s["date"]})
    return out[:8]


def _aggregate_goals(learner_id: str) -> List[dict]:
    rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[r:HAS_GOAL]->(g:Goal)
        WHERE g.category IS NULL OR g.category = 'learner_goal'
        RETURN g.id AS id, g.description AS description, r.status AS status,
               r.updated_at AS updated_at, r.first_seen AS first_seen
        ORDER BY r.updated_at DESC
        LIMIT 8
        """,
        lid=learner_id,
    )
    return [
        {
            "description": r["description"],
            "status": r["status"] or "mentioned",
            "date": _iso(r["updated_at"]),
            "first_seen": _iso(r["first_seen"]),
        }
        for r in rows if r["description"]
    ]
