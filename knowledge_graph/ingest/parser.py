"""
knowledge_graph/ingest/parser.py
─────────────────────────────────
Parses Whisper pipeline output (session_stt.json) into structured dataclasses.

The Whisper pipeline emits a list of segments, each with:
  { "speaker": "SPEAKER_00", "start": float, "end": float,
    "text": str, "words": [...] }

The parser:
  1. Splits segments into learner vs. tutor turns.
  2. Computes basic token/word counts per turn.
  3. Returns a ParsedTranscript dataclass ready for metric computation.

Speaker assignment heuristics (in priority order)
--------------------------------------------------
  a) Explicit mapping passed in via `speaker_map` param:
         {"SPEAKER_00": "learner", "SPEAKER_01": "tutor"}
  b) If exactly 2 speakers found, the one with less total talk-time
     is assigned "learner" (tutors usually dominate early sessions).
  c) Otherwise all segments are labelled "unknown".
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    speaker_label: str          # raw label from Whisper, e.g. "SPEAKER_00"
    role: str                   # "learner" | "tutor" | "unknown"
    start: float
    end: float
    text: str
    words: List[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def word_count(self) -> int:
        tokens = self.text.split()
        return len(tokens)


@dataclass
class ParsedTranscript:
    session_path: str
    turns: List[Turn]
    learner_label: Optional[str]    # e.g. "SPEAKER_00"
    tutor_label: Optional[str]      # e.g. "SPEAKER_01"

    # ── Convenience accessors ──────────────────────────────────────────────────

    @property
    def learner_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "learner"]

    @property
    def tutor_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "tutor"]

    @property
    def full_text(self) -> str:
        return " ".join(t.text for t in self.turns)

    @property
    def learner_text(self) -> str:
        return " ".join(t.text for t in self.learner_turns)

    @property
    def tutor_text(self) -> str:
        return " ".join(t.text for t in self.tutor_turns)

    @property
    def duration_sec(self) -> float:
        if not self.turns:
            return 0.0
        return max(t.end for t in self.turns)

    @property
    def learner_talk_sec(self) -> float:
        return sum(t.duration for t in self.learner_turns)

    @property
    def tutor_talk_sec(self) -> float:
        return sum(t.duration for t in self.tutor_turns)

    @property
    def total_talk_sec(self) -> float:
        return self.learner_talk_sec + self.tutor_talk_sec


# ── Parser ────────────────────────────────────────────────────────────────────

class TranscriptParser:
    """Parse a Whisper diarized-transcript JSON file into a ParsedTranscript."""

    def __init__(self, speaker_map: Optional[Dict[str, str]] = None) -> None:
        """
        Parameters
        ----------
        speaker_map:
            Optional explicit mapping of raw speaker labels to roles.
            E.g. {"SPEAKER_00": "learner", "SPEAKER_01": "tutor"}
        """
        self.speaker_map = speaker_map or {}

    # ── Public ────────────────────────────────────────────────────────────────

    def parse_file(self, path: str | Path) -> ParsedTranscript:
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self._parse(raw, session_path=str(path))

    def parse_dict(self, data: dict, session_path: str = "") -> ParsedTranscript:
        return self._parse(data, session_path=session_path)

    # ── Private ───────────────────────────────────────────────────────────────

    def _parse(self, data: dict, session_path: str) -> ParsedTranscript:
        # Support both {"segments": [...]} and bare list formats
        if isinstance(data, list):
            raw_segments = data
        else:
            raw_segments = data.get("segments", data.get("output", []))

        turns: List[Turn] = []
        for seg in raw_segments:
            label = str(seg.get("speaker", "")).strip()
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = str(seg.get("text", "")).strip()
            words = seg.get("words", [])
            turns.append(Turn(
                speaker_label=label,
                role="unknown",   # resolved below
                start=start,
                end=end,
                text=text,
                words=words,
            ))

        learner_label, tutor_label = self._resolve_roles(turns)

        for turn in turns:
            if self.speaker_map:
                turn.role = self.speaker_map.get(turn.speaker_label, "unknown")
            elif turn.speaker_label == learner_label:
                turn.role = "learner"
            elif turn.speaker_label == tutor_label:
                turn.role = "tutor"

        logger.info(
            "Parsed %d turns — learner=%s tutor=%s (path=%s)",
            len(turns), learner_label, tutor_label, session_path,
        )
        return ParsedTranscript(
            session_path=session_path,
            turns=turns,
            learner_label=learner_label,
            tutor_label=tutor_label,
        )

    def _resolve_roles(
        self, turns: List[Turn]
    ) -> tuple[Optional[str], Optional[str]]:
        """Auto-detect learner/tutor labels from talk-time heuristic."""
        if self.speaker_map:
            # Derive canonical labels from explicit map
            learner = next(
                (k for k, v in self.speaker_map.items() if v == "learner"), None
            )
            tutor = next(
                (k for k, v in self.speaker_map.items() if v == "tutor"), None
            )
            return learner, tutor

        # Aggregate talk-time per speaker
        talk: Dict[str, float] = {}
        for t in turns:
            if t.speaker_label:
                talk[t.speaker_label] = talk.get(t.speaker_label, 0.0) + t.duration

        unique = sorted(talk.keys())
        if len(unique) == 2:
            # Learner = shorter talk time
            learner, tutor = sorted(unique, key=lambda s: talk[s])
            return learner, tutor
        elif len(unique) == 1:
            return unique[0], None
        else:
            logger.warning(
                "Found %d speakers — cannot auto-assign roles. "
                "Pass explicit speaker_map.",
                len(unique),
            )
            return None, None
