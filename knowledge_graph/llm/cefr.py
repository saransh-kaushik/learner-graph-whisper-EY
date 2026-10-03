"""
knowledge_graph/llm/cefr.py
────────────────────────────
CEFR level lookup for vocabulary words.

Priority:
  1. Check local cefr_wordlist.json (fast, free)
  2. OpenAI fallback classification (costs ~$0.001/batch)
  3. Cache OpenAI results back to file

CEFR levels: A1 < A2 < B1 < B2 < C1 < C2
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from knowledge_graph.config import settings

logger = logging.getLogger(__name__)

_CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

# In-memory cache (loaded from file on first use)
_wordlist: Optional[Dict[str, str]] = None

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _load_wordlist() -> Dict[str, str]:
    global _wordlist
    if _wordlist is not None:
        return _wordlist

    path = Path(settings.cefr_reference_path)
    if path.exists():
        _wordlist = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded CEFR wordlist: %d entries from %s", len(_wordlist), path)
    else:
        logger.warning("CEFR wordlist not found at %s — using empty dict", path)
        _wordlist = {}
    return _wordlist


def _save_wordlist() -> None:
    path = Path(settings.cefr_reference_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_wordlist, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def lookup_cefr(word: str) -> str:
    """Return the CEFR level for *word*, querying OpenAI if not cached."""
    word = word.lower().strip()
    wl = _load_wordlist()
    if word in wl:
        return wl[word]

    level = _classify_via_openai([word]).get(word, "B1")
    wl[word] = level
    _save_wordlist()
    return level


def lookup_cefr_batch(words: List[str]) -> Dict[str, str]:
    """Return a dict mapping each word to its CEFR level."""
    wl = _load_wordlist()
    result: Dict[str, str] = {}
    missing: List[str] = []

    for w in words:
        w_clean = w.lower().strip()
        if w_clean in wl:
            result[w_clean] = wl[w_clean]
        else:
            missing.append(w_clean)

    if missing:
        classified = _classify_via_openai(missing)
        for w, level in classified.items():
            result[w] = level
            wl[w] = level
        _save_wordlist()

    return result


def _classify_via_openai(words: List[str]) -> Dict[str, str]:
    """Ask GPT-4o to classify a list of words by CEFR level."""
    word_list_str = "\n".join(f"- {w}" for w in words)
    prompt = (
        "Classify each of the following English words by CEFR level "
        "(A1, A2, B1, B2, C1, or C2). "
        "Return a JSON object mapping each word (lowercase) to its level.\n\n"
        f"{word_list_str}"
    )
    try:
        response = _get_client().chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert English language teacher. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content or "{}")
        # Validate levels
        return {
            k.lower(): v if v in _CEFR_LEVELS else "B1"
            for k, v in data.items()
            if isinstance(v, str)
        }
    except Exception as exc:
        logger.error("CEFR OpenAI classification failed: %s", exc)
        return {w: "B1" for w in words}
