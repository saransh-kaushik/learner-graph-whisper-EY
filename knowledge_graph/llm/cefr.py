"""
knowledge_graph/llm/cefr.py
────────────────────────────
CEFR level lookup for vocabulary words and phrases.

Priority:
  1. Local cefr_wordlist.json cache (fast, free, consistent across sessions)
  2. Level hint from the extraction model (already paid for)
  3. OpenAI classification for anything still unknown
  4. Cache results back to the file

Words that cannot be classified get "unknown" (never a silent default level).
The cache is shared by concurrent ingestion threads, so access is locked and
the file is written atomically.

CEFR levels: A1 < A2 < B1 < B2 < C1 < C2
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from openai import OpenAI

from knowledge_graph.config import settings
from knowledge_graph.taxonomy import CEFR_LEVELS

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"

# In-memory cache (loaded from file on first use)
_wordlist: Optional[Dict[str, str]] = None
_lock = threading.RLock()

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key, max_retries=3)
    return _client


def _normalize(word: str) -> str:
    return " ".join(word.lower().split())


def _load_wordlist() -> Dict[str, str]:
    global _wordlist
    with _lock:
        if _wordlist is not None:
            return _wordlist

        path = Path(settings.cefr_reference_path)
        _wordlist = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                _wordlist = {
                    _normalize(k): v for k, v in raw.items() if v in CEFR_LEVELS
                }
                logger.info("Loaded CEFR wordlist: %d entries from %s", len(_wordlist), path)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("CEFR wordlist at %s is unreadable (%s) — starting empty", path, exc)
        else:
            logger.warning("CEFR wordlist not found at %s — starting empty", path)
        return _wordlist


def _save_wordlist() -> None:
    """Atomically write the cache (temp file + rename) so concurrent readers
    and crashes never see a half-written file."""
    with _lock:
        path = Path(settings.cefr_reference_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(sorted(_wordlist.items())), indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".cefr_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise


def lookup_cefr(word: str) -> str:
    """Return the CEFR level for *word* (or "unknown")."""
    return lookup_cefr_batch([word]).get(_normalize(word), UNKNOWN)


def lookup_cefr_batch(
    words: List[str],
    hints: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """
    Return a dict mapping each (normalised) word/phrase to its CEFR level.

    *hints* maps words to the level the extraction model already estimated;
    they are used (and cached) before making another API call.
    """
    hints = {_normalize(k): v for k, v in (hints or {}).items() if v in CEFR_LEVELS}
    with _lock:
        wl = _load_wordlist()
        result: Dict[str, str] = {}
        missing: List[str] = []
        dirty = False

        for w in dict.fromkeys(_normalize(w) for w in words if w and w.strip()):
            if w in wl:
                result[w] = wl[w]
            elif w in hints:
                result[w] = wl[w] = hints[w]
                dirty = True
            else:
                missing.append(w)

    if missing:
        classified = _classify_via_openai(missing)
        with _lock:
            for w in missing:
                level = classified.get(w)
                if level in CEFR_LEVELS:
                    result[w] = wl[w] = level
                    dirty = True
                else:
                    result[w] = UNKNOWN

    if dirty:
        try:
            _save_wordlist()
        except OSError as exc:
            logger.error("Could not save CEFR wordlist: %s", exc)

    return result


def _classify_via_openai(words: List[str]) -> Dict[str, str]:
    """Ask the model to classify words/phrases by CEFR level. Never raises."""
    word_list_str = "\n".join(f"- {w}" for w in words)
    prompt = (
        "Classify each English word or phrase below by the CEFR level at which a "
        "learner typically acquires it (A1, A2, B1, B2, C1 or C2). "
        "Return a JSON object mapping each item exactly as written (lowercase) to its level.\n\n"
        f"{word_list_str}"
    )
    try:
        response = _get_client().chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert English language assessor. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return {
            _normalize(k): v.upper()
            for k, v in data.items()
            if isinstance(v, str) and v.upper() in CEFR_LEVELS
        }
    except Exception as exc:
        logger.error("CEFR OpenAI classification failed: %s", exc)
        return {}
