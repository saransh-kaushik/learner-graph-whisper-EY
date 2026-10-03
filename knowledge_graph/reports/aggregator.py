"""
knowledge_graph/reports/aggregator.py
───────────────────────────────────────
Queries Neo4j to aggregate a learner's progress data across their last N sessions.

Returns a structured dict consumed by the report generator.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from knowledge_graph.config import settings
from knowledge_graph.db import neo4j_client

logger = logging.getLogger(__name__)

_TREND_UP_THRESHOLD = 1.10
_TREND_DOWN_THRESHOLD = 0.90

ERROR_MAP = {
    "incorrect_past_tense_usage": "past_tense",
    "incorrect_verb_tense": "verb_form",
    "article_and_plural_errors": "article",
    "missing_subject_or_object": "missing_subject",
    "incorrect_preposition_usage": "preposition",
    "incorrect_word_order": "word_order",
    "incorrect_verb_form": "verb_form",
    "pluralization": "plural",
    "incorrect_past_tense": "past_tense",
    "unnecessary_preposition_with_home": "preposition",
    "incorrect_verb_collocation": "collocation",
    "incorrect_auxiliary_usage": "auxiliary",
    "incorrect_modal_usage": "modal",
    "incorrect_past_participle": "verb_form",
    "present_perfect_vs_simple_past": "past_tense",
    "noun_usage": "vocabulary_choice",
    "word_order": "word_order",
    "preposition_usage": "preposition",
    "verb_form": "verb_form",
    "run_on_sentence": "run_on",
    "sentence_fragment": "fragment",
    "missing_subject": "missing_subject",
    "article_usage": "article",
}

def aggregate_learner_progress(
    learner_id: str,
    n_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    n = n_sessions or settings.rolling_baseline_sessions

    learner_rows = neo4j_client.run_query(
        "MATCH (l:Learner {id: $id}) RETURN l.name AS name, l.current_cefr_level AS level",
        id=learner_id,
    )
    learner_name = learner_rows[0]["name"] if learner_rows and learner_rows[0]["name"] else learner_id
    cefr_level   = learner_rows[0]["level"] if learner_rows and learner_rows[0]["level"] else None

    session_rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[:ATTENDED]->(s:Session)
        RETURN s.id AS id, s.date AS date,
               s.talk_time_ratio   AS ttr,
               s.words_per_turn    AS wpt,
               s.average_words_per_sentence AS awps,
               s.speaking_rate_wpm AS wpm,
               s.code_switch_ratio AS csr,
               s.transcript_summary AS summary,
               s.duration_sec AS duration
        ORDER BY s.date DESC
        LIMIT $n
        """,
        lid=learner_id,
        n=n,
    )
    
    sessions = []
    data_quality = []
    valid_sessions_count = 0
    total_sessions_count = 0

    for r in session_rows:
        s = dict(r)
        total_sessions_count += 1
        
        # fix date formatting
        if hasattr(s["date"], "iso_format"):
            s["date"] = s["date"].iso_format()
        elif hasattr(s["date"], "isoformat"):
            s["date"] = s["date"].isoformat()
        else:
            s["date"] = str(s["date"])
            
        issues = []
        if (s.get("ttr") == 0.0 or s.get("ttr") is None) and (s.get("wpm") == 0.0 or s.get("wpm") is None):
            issues.append("no_learner_speech")
        # other validity rules could go here
        
        if issues:
            data_quality.append({
                "session_id": s["id"],
                "date": s["date"],
                "reasons": issues
            })
            s["ttr"] = None
            s["wpt"] = None
            s["awps"] = None
            s["wpm"] = None
            s["csr"] = None
            s["is_valid"] = False
        else:
            valid_sessions_count += 1
            s["is_valid"] = True
            
        sessions.append(s)

    # Reverse to chronological order for trends
    sessions = list(reversed(sessions))

    if not cefr_level:
        est_cefr = _estimate_cefr(learner_id, [s for s in sessions if s["is_valid"]])
        if est_cefr["value"]:
            cefr_level_dict = est_cefr
        else:
            cefr_level_dict = {"value": None, "confidence": est_cefr["confidence"]}
    else:
        cefr_level_dict = {"value": cefr_level, "confidence": "high"}

    if not sessions:
        return {
            "learner_id": learner_id,
            "learner_name": learner_name,
            "cefr_level": cefr_level_dict,
            "sessions_analyzed": {"valid": 0, "total": 0},
            "data_quality": data_quality,
            "metrics": {},
            "vocabulary": {"learner_new_advanced_words": [], "tutor_taught_words": [], "mastery_upgrades": []},
            "grammar": {"fading_errors": [], "persistent_errors": []},
            "top_priority_errors": [],
            "goals": [],
            "learner_goals": [],
            "lesson_topics": [],
            "session_summaries": [],
            "sessions": [],
        }

    metrics = _compute_metric_trends(sessions)
    vocab = _aggregate_vocabulary(learner_id, sessions)
    grammar, top_priority_errors = _aggregate_grammar(learner_id, sessions)
    learner_goals, lesson_topics = _aggregate_goals(learner_id)

    session_summaries = []
    for s in sessions:
        if s["is_valid"] and s.get("summary"):
            session_summaries.append({
                "date": s["date"],
                "summary": s["summary"]
            })

    return {
        "learner_id": learner_id,
        "learner_name": learner_name,
        "cefr_level": cefr_level_dict,
        "sessions_analyzed": {
            "valid": valid_sessions_count,
            "total": total_sessions_count
        },
        "data_quality": data_quality,
        "metrics": metrics,
        "vocabulary": vocab,
        "grammar": grammar,
        "top_priority_errors": top_priority_errors,
        "goals": learner_goals + lesson_topics, # legacy key
        "learner_goals": learner_goals,
        "lesson_topics": lesson_topics,
        "session_summaries": session_summaries,
        "sessions": sessions,
    }


def _estimate_cefr(learner_id: str, valid_sessions: List[dict]) -> dict:
    if len(valid_sessions) < 2:
        return {"value": None, "confidence": "insufficient_data"}
    session_ids = [s["id"] for s in valid_sessions]
    rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[r:USED_WORD]->(w:Word)
        WHERE r.session_id IN $sids AND w.cefr_level IS NOT NULL
        RETURN w.cefr_level AS cefr
        """,
        lid=learner_id, sids=session_ids
    )
    levels = [r["cefr"] for r in rows if r["cefr"] in ("A1", "A2", "B1", "B2", "C1", "C2")]
    if not levels:
        return {"value": None, "confidence": "insufficient_data"}
    
    level_map = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    rev_map = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}
    avg = sum(level_map[l] for l in levels) / len(levels)
    est_level = rev_map[round(avg)]
    conf = "high" if len(levels) > 50 else ("medium" if len(levels) > 20 else "low")
    return {"value": est_level, "confidence": conf}


def _compute_metric_trends(sessions: List[dict]) -> Dict[str, dict]:
    metric_keys = {
        "talk_time_ratio": "ttr",
        "average_words_per_sentence": "awps",
        "words_per_turn": "wpt",
        "speaking_rate_wpm": "wpm",
        "code_switch_ratio": "csr",
    }
    result = {}
    for name, key in metric_keys.items():
        history = []
        valid_values = []
        for s in sessions:
            val = s.get(key)
            if s["is_valid"] and val is not None:
                valid_values.append(val)
                val_to_record = round(val, 4)
            else:
                val_to_record = None
                
            history.append({
                "session_id": s["id"],
                "date": s["date"],
                "value": val_to_record
            })
            
        if not valid_values:
            result[name] = {
                "current": None,
                "baseline": None,
                "trend": "insufficient_data",
                "history": history
            }
            continue
            
        current = valid_values[-1] # most recent valid
        
        if len(valid_values) < 3:
            trend = "insufficient_data"
            baseline = sum(valid_values) / len(valid_values)
        else:
            earlier_values = valid_values[:-1]
            baseline = sum(earlier_values) / len(earlier_values)
            if name == "code_switch_ratio":
                if current < baseline * _TREND_DOWN_THRESHOLD:
                    trend = "improving"
                elif current > baseline * _TREND_UP_THRESHOLD:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                if current > baseline * _TREND_UP_THRESHOLD:
                    trend = "improving"
                elif current < baseline * _TREND_DOWN_THRESHOLD:
                    trend = "declining"
                else:
                    trend = "stable"

        result[name] = {
            "current": round(current, 4),
            "baseline": round(baseline, 4),
            "trend": trend,
            "history": history,
        }
    return result


def _aggregate_vocabulary(learner_id: str, sessions: List[dict]) -> dict:
    session_ids = [s["id"] for s in sessions if s["is_valid"]]
    if not session_ids:
        return {"learner_new_advanced_words": [], "tutor_taught_words": [], "mastery_upgrades": []}

    rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[r:USED_WORD]->(w:Word)
        WHERE r.session_id IN $sids
        RETURN w.lemma AS word, w.cefr_level AS cefr,
               r.mastery AS mastery, r.session_id AS session_id,
               r.date AS date
        ORDER BY r.date ASC
        """,
        lid=learner_id,
        sids=session_ids,
    )
    
    tutor_rows = neo4j_client.run_query(
        """
        MATCH (s:Session)-[r:INTRODUCED_WORD]->(w:Word)
        WHERE s.id IN $sids
        RETURN w.lemma AS word, s.date AS date, r.context_sentence AS context
        ORDER BY s.date ASC
        """,
        sids=session_ids,
    )
    
    TRIVIAL_WORDS = {"abide", "subscription", "pdf", "session", "network"}
    HINDI_WORDS = {"prashad", "seva", "parikrama"}
    
    learner_new_advanced_words = []
    seen_learner_words = set()
    mastery_upgrades = []

    for r in rows:
        word = r["word"].lower()
        if word in TRIVIAL_WORDS or word in HINDI_WORDS:
            continue
        cefr = r["cefr"]
        if cefr in ("A1", "A2", "unknown"):
            continue
            
        mastery = r["mastery"]
        if word not in seen_learner_words:
            learner_new_advanced_words.append({
                "word": word,
                "date": r["date"].iso_format() if hasattr(r["date"], "iso_format") else str(r["date"])
            })
            seen_learner_words.add(word)

    tutor_taught_words = []
    seen_tutor_words = set()
    for r in tutor_rows:
        word = r["word"].lower()
        if word in TRIVIAL_WORDS or word in HINDI_WORDS:
            continue
        if word not in seen_tutor_words:
            tutor_taught_words.append({
                "word": word,
                "date": r["date"].iso_format() if hasattr(r["date"], "iso_format") else str(r["date"]),
                "context_sentence": r.get("context")
            })
            seen_tutor_words.add(word)

    return {
        "learner_new_advanced_words": learner_new_advanced_words[-10:],
        "tutor_taught_words": tutor_taught_words[-10:],
        "mastery_upgrades": mastery_upgrades
    }


def _aggregate_grammar(learner_id: str, sessions: List[dict]) -> tuple[dict, list]:
    session_ids = [s["id"] for s in sessions if s["is_valid"]]
    if not session_ids:
        return {"fading_errors": [], "persistent_errors": []}, []

    rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[r:MADE_ERROR]->(gp:GrammarPattern)
        WHERE r.session_id IN $sids
        RETURN gp.pattern_name AS pattern, r.error_count AS count, r.date AS date,
               r.example_utterance AS wrong, gp.example_correct AS correct,
               r.session_id AS session_id
        ORDER BY r.date ASC
        """,
        lid=learner_id,
        sids=session_ids,
    )

    session_words = {s["id"]: s.get("wpt", 0)*10 for s in sessions if s["is_valid"]} 
    
    by_pattern = {}
    for r in rows:
        raw_pat = r["pattern"]
        pat = ERROR_MAP.get(raw_pat, raw_pat)
        
        sid = r["session_id"]
        words = session_words.get(sid) or 100
        norm_count = (r["count"] / words) * 100 if words > 0 else 0
        
        if pat not in by_pattern:
            by_pattern[pat] = {"counts": [], "examples": [], "total_count": 0}
            
        by_pattern[pat]["counts"].append(norm_count)
        by_pattern[pat]["total_count"] += r["count"]
        if r["wrong"] and len(by_pattern[pat]["examples"]) < 2:
            by_pattern[pat]["examples"].append({
                "wrong": r["wrong"],
                "correct": r["correct"]
            })

    fading, persistent, new_errs = [], [], []
    priority_list = []
    
    for pat, data in by_pattern.items():
        counts = data["counts"]
        total = data["total_count"]
        
        priority_list.append({
            "pattern": pat,
            "total_count": total,
            "examples": data["examples"],
            "latest_norm_freq": counts[-1] if counts else 0
        })
        
        if len(counts) == 1:
            new_errs.append({"pattern": pat, "trend": counts})
        elif len(counts) >= 2:
            if counts[-1] < counts[0] * 0.8:
                fading.append({"pattern": pat, "trend": counts})
            else:
                persistent.append({"pattern": pat, "trend": counts})
                
    priority_list.sort(key=lambda x: x["latest_norm_freq"], reverse=True)
    top_priority_errors = priority_list[:3]

    return {"fading_errors": fading, "persistent_errors": persistent, "new_errors": new_errs}, top_priority_errors


def _aggregate_goals(learner_id: str) -> tuple[list, list]:
    rows = neo4j_client.run_query(
        """
        MATCH (l:Learner {id: $lid})-[r:HAS_GOAL]->(g:Goal)
        RETURN g.description AS description, r.status AS status, r.updated_at AS updated_at
        ORDER BY r.updated_at DESC
        """,
        lid=learner_id,
    )
    
    learner_goals = []
    lesson_topics = []
    
    for r in rows:
        desc = r["description"].lower()
        if "improve" in desc or "quality" in desc or "stable" in desc or "network" in desc:
            continue
            
        date_str = r["updated_at"].iso_format() if hasattr(r["updated_at"], "iso_format") else str(r["updated_at"])
        
        entry = {
            "description": r["description"],
            "status": r["status"],
            "date": date_str
        }
        
        if "use of" in desc or "differentiate" in desc or "forming" in desc or "pronounce" in desc or "prepare" in desc or "answer common" in desc or "introduce" in desc:
            lesson_topics.append(entry)
        else:
            learner_goals.append(entry)
            
    def dedup(items):
        seen = set()
        res = []
        for x in items:
            if x["description"] not in seen:
                res.append(x)
                seen.add(x["description"])
        return res

    return dedup(learner_goals), dedup(lesson_topics)
