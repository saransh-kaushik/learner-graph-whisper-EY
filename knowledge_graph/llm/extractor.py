"""
knowledge_graph/llm/extractor.py
─────────────────────────────────
OpenAI-powered NLP extraction from diarized transcripts.

Extracts (via strict JSON-schema structured output):
  • grammar_errors     — fixed categories, verbatim learner quotes + corrections
  • vocabulary         — words *and* phrases with meaning, level and a work example
  • strengths          — what the learner did well, with quotes
  • learner_goals      — the learner's own communication goals
  • lesson_topics      — what the session practised
  • skills_covered     — skills practised in the session
  • estimated_level    — holistic CEFR estimate of the learner's speaking
  • session_summary    — 2 sentences addressed to the learner

Also provides `identify_learner_label`, a cheap call that decides which
diarized speaker is the tutor when no explicit speaker map is given.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

from openai import BadRequestError, OpenAI

from knowledge_graph.config import settings
from knowledge_graph.taxonomy import CEFR_LEVELS, GRAMMAR_CATEGORIES, GRAMMAR_CATEGORY_IDS

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key, max_retries=3, timeout=180)
    return _client


class ExtractionError(RuntimeError):
    """Raised when the LLM extraction cannot produce a usable result."""


# ── Result dataclasses ─────────────────────────────────────────────────────────

VocabSource = Literal["learner_spontaneous", "learner_after_prompt", "learner_repeated", "tutor_only"]
GoalStatus = Literal["mentioned", "practiced", "demonstrated"]


@dataclass
class GrammarError:
    category: str               # one of taxonomy.GRAMMAR_CATEGORY_IDS
    description: str            # specific, one line
    example_wrong: str          # verbatim learner quote
    example_correct: str
    explanation: str = ""       # short rule in plain English
    count: int = 1

    @property
    def pattern(self) -> str:   # backwards-compatible name
        return self.category


@dataclass
class VocabItem:
    term: str                   # base form; may be a multi-word phrase
    kind: str = "word"          # word | phrase | phrasal_verb | idiom | collocation
    part_of_speech: str = ""
    level: str = "B1"           # model's CEFR estimate for the term
    meaning: str = ""
    workplace_example: str = ""
    context_sentence: str = ""  # verbatim line from the transcript
    source: VocabSource = "learner_spontaneous"
    introduced_by_tutor: bool = False
    used_correctly: bool = True
    correction: str = ""

    @property
    def word(self) -> str:
        return self.term

    @property
    def used_by_learner(self) -> bool:
        return self.source != "tutor_only"

    @property
    def spontaneous(self) -> bool:
        return self.source == "learner_spontaneous"

    @property
    def mastery(self) -> Optional[str]:
        return {
            "learner_spontaneous": "spontaneous",
            "learner_after_prompt": "prompted",
            "learner_repeated": "repeated",
        }.get(self.source)


@dataclass
class GoalItem:
    description: str
    status: GoalStatus = "mentioned"


@dataclass
class Strength:
    area: str
    quote: str
    note: str


@dataclass
class SkillItem:
    name: str
    category: str   # speaking | vocabulary | grammar | fluency | listening


@dataclass
class ExtractionResult:
    grammar_errors: List[GrammarError] = field(default_factory=list)
    vocabulary: List[VocabItem] = field(default_factory=list)
    learner_goals: List[GoalItem] = field(default_factory=list)
    lesson_topics: List[str] = field(default_factory=list)
    strengths: List[Strength] = field(default_factory=list)
    skills_covered: List[SkillItem] = field(default_factory=list)
    estimated_level: Optional[str] = None
    session_summary: str = ""


# ── Prompt ────────────────────────────────────────────────────────────────────

_GRAMMAR_LIST = "\n".join(
    f"  - {cid}: {desc}" for cid, (_label, desc) in GRAMMAR_CATEGORIES.items()
)

_SYSTEM_PROMPT = f"""You analyse one-to-one spoken English coaching sessions for EnglishYaari.

WHO THE LEARNER IS
The learner is a working professional in India improving spoken English for work:
meetings, client calls, interviews, presentations, stand-ups and small talk with
colleagues. They will read a short progress report built from your analysis, so
everything you return must be accurate, specific and useful to an adult professional.

WHAT YOU RECEIVE
A transcript produced by automatic speech recognition (ASR) and speaker diarization.
Lines are labelled LEARNER or TUTOR. Expect ASR noise: missing punctuation,
misheard words, broken sentences at segment boundaries and filler words.

GROUND RULES
- Analyse only LEARNER lines for errors, strengths and learner vocabulary.
  TUTOR lines are context only.
- Quote the learner verbatim (copy the words exactly, trimming to the relevant clause).
  Never invent or "clean up" a quote.
- Do NOT treat these as grammar errors: ASR mistakes, missing punctuation or
  capitalisation, fillers (um, uh, like, you know), false starts the learner
  self-corrected, or Hindi/Hinglish words (code-switching is measured separately).
- Do not comment on pronunciation or accent — you cannot hear the audio.
- If something is not clearly in the transcript, leave it out. Empty lists are fine.

GRAMMAR ERRORS
Use exactly one of these categories for each error:
{_GRAMMAR_LIST}
- Group repeated instances of the same specific mistake into one item and set
  "count" to the number of distinct learner utterances that contain it.
- "example_wrong": the learner's exact words. "example_correct": the same sentence
  corrected with the smallest change a native speaker would make.
- "explanation": one short rule in plain English (max 20 words), no jargon.
- Report at most 8 items, the most frequent and most important for work first.

VOCABULARY
List words AND multi-word expressions (phrasal verbs, collocations, idioms,
workplace phrases such as "circle back", "on the same page", "take ownership").
Include an item only if it is useful for a professional AND it is one of:
  a) used by the learner and at B1 level or above (or a useful phrase at any level),
  b) taught, suggested or corrected by the tutor,
  c) misused by the learner (set used_correctly=false and give the correction).
Exclude: names of people, companies, products and places; Hindi words; filler;
very basic words (A1–A2 single words like "good", "work", "meeting"); technical
jargon of the learner's job unless the tutor was teaching it.
For each item:
- "term": dictionary/base form, lowercase ("negotiate", "follow up on").
- "level": CEFR level of the term itself.
- "meaning": simple definition, max 15 words, for a non-native professional.
- "workplace_example": one natural new sentence using the term at work.
- "context_sentence": the transcript line where it appeared (verbatim).
- "source":
    learner_spontaneous  = learner used it on their own, without being prompted
    learner_after_prompt = learner used it after the tutor asked them to / modelled it
    learner_repeated     = learner only repeated it straight after the tutor
    tutor_only           = only the tutor said it (taught or suggested it)
- "introduced_by_tutor": true if the tutor taught, suggested or explained it.
- "correction": the correct usage if used_correctly=false, else "".
Return at most 20 items, most useful first.

STRENGTHS
Up to 4 specific things the learner did well, each with a verbatim quote that shows it
(e.g. a well-formed complex sentence, a precise word choice, a confident long answer,
self-correcting an error). No generic praise.

GOALS AND TOPICS
- "learner_goals": goals the LEARNER expresses for their own communication or career
  (e.g. "Speak confidently in client calls", "Clear a job interview in English").
  Not session logistics, not internet or audio problems, not lesson activities.
  status: mentioned (only talked about) | practiced (worked on in the session) |
  demonstrated (learner clearly did it well, unprompted).
- "lesson_topics": short labels for what the session practised
  (e.g. "Describing your role", "Past tense in project updates"). Max 5.

LEVEL
"estimated_level": your holistic CEFR estimate of the learner's SPOKEN English in
this session, based on range, accuracy, fluency and coherence.

SUMMARY
"session_summary": 2 short sentences addressed to the learner as "you", saying what
was practised and one notable observation. Professional, warm, no exaggeration.
"""

_USER_TEMPLATE = """SESSION TRANSCRIPT
==================
{transcript}
"""


def _json_schema() -> dict:
    s = {"type": "string"}
    b = {"type": "boolean"}

    def obj(props: dict) -> dict:
        return {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),
            "additionalProperties": False,
        }

    return obj({
        "grammar_errors": {"type": "array", "items": obj({
            "category": {"type": "string", "enum": list(GRAMMAR_CATEGORY_IDS)},
            "description": s,
            "example_wrong": s,
            "example_correct": s,
            "explanation": s,
            "count": {"type": "integer"},
        })},
        "vocabulary": {"type": "array", "items": obj({
            "term": s,
            "kind": {"type": "string", "enum": ["word", "phrase", "phrasal_verb", "idiom", "collocation"]},
            "part_of_speech": s,
            "level": {"type": "string", "enum": list(CEFR_LEVELS)},
            "meaning": s,
            "workplace_example": s,
            "context_sentence": s,
            "source": {"type": "string", "enum": [
                "learner_spontaneous", "learner_after_prompt", "learner_repeated", "tutor_only",
            ]},
            "introduced_by_tutor": b,
            "used_correctly": b,
            "correction": s,
        })},
        "strengths": {"type": "array", "items": obj({
            "area": {"type": "string", "enum": [
                "grammar", "vocabulary", "fluency", "clarity", "confidence", "interaction",
            ]},
            "quote": s,
            "note": s,
        })},
        "learner_goals": {"type": "array", "items": obj({
            "description": s,
            "status": {"type": "string", "enum": ["mentioned", "practiced", "demonstrated"]},
        })},
        "lesson_topics": {"type": "array", "items": s},
        "skills_covered": {"type": "array", "items": obj({
            "name": s,
            "category": {"type": "string", "enum": [
                "speaking", "vocabulary", "grammar", "fluency", "listening",
            ]},
        })},
        "estimated_level": {"type": "string", "enum": list(CEFR_LEVELS)},
        "session_summary": s,
    })


def _chat_json(messages: list, schema_name: str, schema: dict, temperature: float) -> dict:
    """Call the chat API with a strict JSON schema, falling back to JSON mode
    for models that don't support structured outputs."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            messages=messages,
            temperature=temperature,
        )
    except BadRequestError as exc:
        logger.warning("Structured output not supported (%s); falling back to JSON mode", exc)
        response = client.chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=messages + [{
                "role": "system",
                "content": "Return ONLY a JSON object matching this schema:\n" + json.dumps(schema),
            }],
            temperature=temperature,
        )
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise ExtractionError(f"Model refused: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise ExtractionError("Model output was truncated (transcript too long?)")
    return json.loads(choice.message.content or "{}")


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_from_transcript(transcript_text: str) -> ExtractionResult:
    """
    Extract grammar errors, vocabulary, strengths, goals and topics from a
    formatted transcript ("LEARNER: …" / "TUTOR: …" lines).

    Raises ExtractionError if no usable result can be produced, so callers
    never store an empty analysis that would look like "zero errors".
    """
    if not transcript_text.strip():
        raise ExtractionError("Transcript is empty")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(transcript=transcript_text)},
    ]

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            logger.info("Calling OpenAI extraction (model=%s, attempt=%d)…", settings.openai_model, attempt + 1)
            data = _chat_json(messages, "session_analysis", _json_schema(), temperature=0.1)
            return _parse_extraction(data)
        except (json.JSONDecodeError, ExtractionError) as exc:
            last_exc = exc
            logger.warning("Extraction attempt %d failed: %s", attempt + 1, exc)
    raise ExtractionError(f"Extraction failed after retries: {last_exc}")


def _str(item: dict, key: str) -> str:
    return str(item.get(key) or "").strip()


def _parse_extraction(data: dict) -> ExtractionResult:
    result = ExtractionResult()

    for item in data.get("grammar_errors", []):
        try:
            category = _str(item, "category")
            if category not in GRAMMAR_CATEGORIES:
                category = "other"
            wrong = _str(item, "example_wrong")
            if not wrong:
                continue
            result.grammar_errors.append(GrammarError(
                category=category,
                description=_str(item, "description"),
                example_wrong=wrong,
                example_correct=_str(item, "example_correct"),
                explanation=_str(item, "explanation"),
                count=max(1, int(item.get("count") or 1)),
            ))
        except Exception as exc:
            logger.warning("Skipping grammar_error item: %s", exc)

    seen_terms = set()
    for item in data.get("vocabulary", []):
        try:
            term = " ".join(_str(item, "term").lower().split())
            if not term or term in seen_terms:
                continue
            seen_terms.add(term)
            source = _str(item, "source")
            if source not in ("learner_spontaneous", "learner_after_prompt", "learner_repeated", "tutor_only"):
                source = "tutor_only"
            level = _str(item, "level").upper()
            result.vocabulary.append(VocabItem(
                term=term,
                kind=_str(item, "kind") or "word",
                part_of_speech=_str(item, "part_of_speech"),
                level=level if level in CEFR_LEVELS else "B1",
                meaning=_str(item, "meaning"),
                workplace_example=_str(item, "workplace_example"),
                context_sentence=_str(item, "context_sentence"),
                source=source,  # type: ignore[arg-type]
                introduced_by_tutor=bool(item.get("introduced_by_tutor", False)),
                used_correctly=bool(item.get("used_correctly", True)),
                correction=_str(item, "correction"),
            ))
        except Exception as exc:
            logger.warning("Skipping vocabulary item: %s", exc)

    for item in data.get("strengths", []):
        quote = _str(item, "quote")
        if quote:
            result.strengths.append(Strength(
                area=_str(item, "area") or "fluency", quote=quote, note=_str(item, "note"),
            ))

    for item in data.get("learner_goals", []):
        desc = _str(item, "description")
        if not desc:
            continue
        status = _str(item, "status")
        if status not in ("mentioned", "practiced", "demonstrated"):
            status = "mentioned"
        result.learner_goals.append(GoalItem(description=desc, status=status))  # type: ignore[arg-type]

    result.lesson_topics = [t.strip() for t in data.get("lesson_topics", []) if str(t).strip()][:5]

    for item in data.get("skills_covered", []):
        name = _str(item, "name")
        if name:
            result.skills_covered.append(SkillItem(name=name, category=_str(item, "category") or "speaking"))

    level = _str(data, "estimated_level").upper()
    result.estimated_level = level if level in CEFR_LEVELS else None
    result.session_summary = _str(data, "session_summary")
    return result


# ── Speaker role identification ───────────────────────────────────────────────

_ROLE_PROMPT = """Below is the start of a diarized transcript of a one-to-one English
coaching session. One speaker is the TUTOR (asks questions, explains, corrects,
gives feedback, runs the lesson). The other is the LEARNER (answers, practises,
makes more errors). Identify the tutor's speaker label.

Return JSON: {"tutor_label": "<one of the labels>", "confident": true|false}"""


def identify_learner_label(turns: Sequence[tuple[str, str]], labels: Sequence[str]) -> Optional[str]:
    """
    Given (speaker_label, text) turns and exactly two speaker labels, ask the
    model which speaker is the tutor and return the *learner's* label.
    Returns None if unsure or on any failure (caller falls back to heuristics).
    """
    if len(labels) != 2:
        return None
    sample_lines = []
    total = 0
    for label, text in turns:
        line = f"{label}: {text.strip()}"
        sample_lines.append(line)
        total += len(line)
        if total > 6000 or len(sample_lines) >= 80:
            break
    schema = {
        "type": "object",
        "properties": {
            "tutor_label": {"type": "string", "enum": list(labels)},
            "confident": {"type": "boolean"},
        },
        "required": ["tutor_label", "confident"],
        "additionalProperties": False,
    }
    try:
        data = _chat_json(
            [
                {"role": "system", "content": _ROLE_PROMPT},
                {"role": "user", "content": "\n".join(sample_lines)},
            ],
            "speaker_roles",
            schema,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("Speaker role identification failed: %s", exc)
        return None

    tutor = data.get("tutor_label")
    if tutor not in labels or not data.get("confident", False):
        return None
    return next(l for l in labels if l != tutor)
