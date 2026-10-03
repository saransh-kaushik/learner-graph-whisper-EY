"""
knowledge_graph/metrics.py
──────────────────────────
Session-level metrics computed from a ParsedTranscript.

Builds on core/metrics_utils.py talk-time data and adds:
  • words_per_turn
  • speaking_rate_wpm
  • code_switch_ratio  (Hindi/Hinglish detection)
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import List

from knowledge_graph.ingest.parser import ParsedTranscript, Turn

logger = logging.getLogger(__name__)

# ── Hindi word lists (romanised, curated) ─────────────────────────────────────
# Unambiguous: these are not English words, so they always count as Hindi.
_HINDI_WORDS: frozenset[str] = frozenset({
    "aur", "hai", "hain", "tha", "thi", "mein", "ka", "ki", "ke", "ko",
    "yeh", "ye", "woh", "wo", "kya", "kaise", "kyun", "kyunki", "lekin",
    "toh", "bhi", "nahi", "nahin", "haan", "naa", "bahut", "accha", "acha",
    "achha", "theek", "thik", "sahi", "matlab", "samajh", "samjha", "samjhe",
    "kar", "karo", "karna", "karta", "karti", "karte", "kiya", "phir", "abhi",
    "pehle", "baad", "sirf", "ek", "teen", "paanch", "kab", "yahan", "wahan",
    "itna", "kitna", "hoon", "hun", "hu", "raha", "rahi", "rahe", "gaya",
    "gayi", "gaye", "wala", "wali", "wale", "bolta", "bolti", "bolte", "bol",
    "kuch", "sab", "sabhi", "apna", "apni", "apne", "mera", "meri", "mere",
    "tera", "teri", "tumhara", "aap", "aapka", "aapki", "hum", "humko",
    "mujhe", "tujhe", "unko", "usko", "isko", "yaar", "arre", "thoda",
    "zyada", "jaldi", "chalo", "dekho", "suno", "pata", "chahiye", "sakta",
    "sakti", "sakte", "hota", "hoti", "hote", "jaise", "waise", "kaun",
    "kahan", "kitne", "unka", "uska", "iska", "humara", "hamara", "se",
})

# Ambiguous: valid Hindi *and* common English ("the", "main", "do", "par" …).
# Counted as Hindi only when next to an unambiguous Hindi word, so
# "the main point" stays English while "main bolta hoon" is Hindi.
_AMBIGUOUS_WORDS: frozenset[str] = frozenset({
    "the", "main", "do", "par", "pe", "na", "tab", "jab", "ab", "bas", "char",
})

# Devanagari Unicode block
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_TOKEN_CLEAN_RE = re.compile(r"[^\wऀ-ॿ]")


def _clean_token(tok: str) -> str:
    return _TOKEN_CLEAN_RE.sub("", tok).lower()


def hindi_token_flags(tokens: List[str]) -> List[bool]:
    """Return one flag per token: True if the token is Hindi/Hinglish."""
    cleaned = [_clean_token(t) for t in tokens]
    certain = [
        bool(c) and (bool(_DEVANAGARI_RE.search(c)) or c in _HINDI_WORDS)
        for c in cleaned
    ]
    flags = list(certain)
    for i, c in enumerate(cleaned):
        if c in _AMBIGUOUS_WORDS:
            prev_hindi = i > 0 and certain[i - 1]
            next_hindi = i + 1 < len(cleaned) and certain[i + 1]
            flags[i] = prev_hindi or next_hindi
    return flags


@dataclass
class SessionMetrics:
    talk_time_ratio: float      # learner_talk_sec / total_talk_sec
    words_per_turn: float       # mean learner words per turn
    average_words_per_sentence: float # mean english words per sentence
    speaking_rate_wpm: float    # learner words per minute
    code_switch_ratio: float    # Hindi/Hinglish tokens / total learner tokens
    learner_talk_sec: float
    tutor_talk_sec: float
    total_duration_sec: float
    learner_turn_count: int
    total_learner_words: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_session_metrics(transcript: ParsedTranscript) -> SessionMetrics:
    """Compute all session metrics from a ParsedTranscript."""
    learner_turns: List[Turn] = transcript.learner_turns
    learner_talk_sec = transcript.learner_talk_sec
    tutor_talk_sec = transcript.tutor_talk_sec
    total_talk_sec = transcript.total_talk_sec

    # ── Talk-time ratio ────────────────────────────────────────────────────────
    talk_time_ratio = (
        learner_talk_sec / total_talk_sec if total_talk_sec > 0 else 0.0
    )

    # ── Words per turn & Sentence Length ──────────────────────────────────────
    word_counts = []
    english_sentence_lengths = []
    total_learner_words = 0

    for t in learner_turns:
        if t.word_count > 0:
            word_counts.append(t.word_count)
            total_learner_words += t.word_count

            sentences = re.split(r'[.!?]+', t.text)
            for s in sentences:
                tokens = s.strip().split()
                if not tokens:
                    continue
                flags = hindi_token_flags(tokens)
                eng_count = sum(
                    1 for tok, is_hindi in zip(tokens, flags)
                    if _clean_token(tok) and not is_hindi
                )
                if eng_count > 0:
                    english_sentence_lengths.append(eng_count)

    words_per_turn = (
        sum(word_counts) / len(word_counts) if word_counts else 0.0
    )
    average_words_per_sentence = (
        sum(english_sentence_lengths) / len(english_sentence_lengths) if english_sentence_lengths else 0.0
    )

    # ── Speaking rate (WPM) ───────────────────────────────────────────────────
    learner_talk_min = learner_talk_sec / 60.0
    speaking_rate_wpm = (
        total_learner_words / learner_talk_min if learner_talk_min > 0 else 0.0
    )

    # ── Code-switch ratio ─────────────────────────────────────────────────────
    code_switch_ratio = _compute_code_switch_ratio(
        transcript.learner_text, total_learner_words
    )

    return SessionMetrics(
        talk_time_ratio=round(talk_time_ratio, 4),
        words_per_turn=round(words_per_turn, 2),
        average_words_per_sentence=round(average_words_per_sentence, 2),
        speaking_rate_wpm=round(speaking_rate_wpm, 2),
        code_switch_ratio=round(code_switch_ratio, 4),
        learner_talk_sec=round(learner_talk_sec, 2),
        tutor_talk_sec=round(tutor_talk_sec, 2),
        total_duration_sec=round(transcript.duration_sec, 2),
        learner_turn_count=len(learner_turns),
        total_learner_words=total_learner_words,
    )


def _compute_code_switch_ratio(learner_text: str, total_words: int) -> float:
    """Estimate fraction of learner tokens that are Hindi/Hinglish."""
    if not learner_text or total_words == 0:
        return 0.0

    tokens = learner_text.split()
    hindi_count = sum(hindi_token_flags(tokens))
    return hindi_count / total_words
