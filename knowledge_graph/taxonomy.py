"""
knowledge_graph/taxonomy.py
───────────────────────────
Shared, fixed vocabularies used by extraction, aggregation and reporting:

  • CEFR levels with plain-language descriptions for working professionals
  • A closed set of grammar categories (so patterns don't fragment across sessions)
  • Metric definitions (label, unit, which direction is "better")
"""
from __future__ import annotations

import re
from typing import Dict, Optional

# ── CEFR ──────────────────────────────────────────────────────────────────────

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_RANK = {lvl: i for i, lvl in enumerate(CEFR_LEVELS, start=1)}

# (short name, what it means for someone using English at work)
CEFR_DESCRIPTIONS: Dict[str, tuple[str, str]] = {
    "A1": ("Beginner", "Uses basic, familiar phrases for simple needs."),
    "A2": ("Elementary", "Handles short, routine exchanges on familiar work topics."),
    "B1": ("Intermediate", "Gets the main point across in familiar work situations."),
    "B2": ("Upper-intermediate", "Takes part in most meetings and calls with confidence."),
    "C1": ("Advanced", "Speaks fluently and precisely, even in complex discussions."),
    "C2": ("Proficient", "Communicates with near-native ease and nuance."),
}

# How a word at each level reads to a learner
WORD_LEVEL_HINTS: Dict[str, str] = {
    "A1": "Basic",
    "A2": "Everyday",
    "B1": "Common at work",
    "B2": "Professional",
    "C1": "Advanced",
    "C2": "Expert",
}


def describe_level(level: Optional[str]) -> dict:
    if level not in CEFR_DESCRIPTIONS:
        return {"value": None, "label": None, "description": None}
    name, desc = CEFR_DESCRIPTIONS[level]
    return {"value": level, "label": name, "description": desc}


# ── Grammar categories ───────────────────────────────────────────────────────

# id -> (learner-facing label, description given to the extraction model)
GRAMMAR_CATEGORIES: Dict[str, tuple[str, str]] = {
    "verb_tense": ("Verb tenses", "Wrong tense for the time referred to (e.g. 'I go yesterday', present perfect vs past simple)."),
    "subject_verb_agreement": ("Subject–verb agreement", "Verb does not agree with its subject (e.g. 'he go', 'the team are agree')."),
    "verb_form": ("Verb forms", "Wrong form of the verb: infinitive/-ing/participle (e.g. 'I am agree', 'looking forward to meet')."),
    "articles": ("Articles (a / an / the)", "Missing, extra or wrong article."),
    "prepositions": ("Prepositions", "Missing, extra or wrong preposition (e.g. 'discuss about', 'married with')."),
    "plurals_and_countability": ("Plurals & countable nouns", "Singular/plural errors and uncountable nouns (e.g. 'informations', 'many work')."),
    "pronouns": ("Pronouns", "Wrong or missing pronoun (he/she, him/her, it, they)."),
    "word_order": ("Word order", "Words in the wrong order, including indirect questions."),
    "question_formation": ("Asking questions", "Incorrectly formed direct questions (e.g. 'You are coming?', 'Why you said that?')."),
    "auxiliary_and_modals": ("Helping & modal verbs", "Missing or wrong do/does/did, be, have, can/could/should/would."),
    "missing_word": ("Missing words", "A required subject, object or verb is missing (e.g. 'Is very important')."),
    "word_choice": ("Word choice & collocations", "Wrong word or unnatural combination (e.g. 'do a mistake', 'open the light')."),
    "sentence_structure": ("Sentence structure", "Run-on sentences, fragments or sentences that lose their thread."),
    "comparatives": ("Comparisons", "Errors with comparatives/superlatives (e.g. 'more better')."),
    "conditionals": ("If-sentences", "Errors in conditional sentences (e.g. 'If I will get time')."),
    "other": ("Other grammar", "Anything that does not fit the categories above."),
}

GRAMMAR_CATEGORY_IDS = tuple(GRAMMAR_CATEGORIES.keys())

# Free-form names produced by the earlier extraction prompt → fixed categories
_LEGACY_GRAMMAR_MAP: Dict[str, str] = {
    "incorrect_past_tense_usage": "verb_tense",
    "incorrect_verb_tense": "verb_tense",
    "incorrect_past_tense": "verb_tense",
    "present_perfect_vs_simple_past": "verb_tense",
    "past_tense": "verb_tense",
    "article_and_plural_errors": "articles",
    "article_usage": "articles",
    "article": "articles",
    "missing_subject_or_object": "missing_word",
    "missing_subject": "missing_word",
    "incorrect_preposition_usage": "prepositions",
    "unnecessary_preposition_with_home": "prepositions",
    "preposition_usage": "prepositions",
    "preposition": "prepositions",
    "incorrect_word_order": "word_order",
    "word_order": "word_order",
    "incorrect_verb_form": "verb_form",
    "incorrect_past_participle": "verb_form",
    "verb_form": "verb_form",
    "pluralization": "plurals_and_countability",
    "plural": "plurals_and_countability",
    "incorrect_verb_collocation": "word_choice",
    "collocation": "word_choice",
    "noun_usage": "word_choice",
    "vocabulary_choice": "word_choice",
    "incorrect_auxiliary_usage": "auxiliary_and_modals",
    "incorrect_modal_usage": "auxiliary_and_modals",
    "auxiliary": "auxiliary_and_modals",
    "modal": "auxiliary_and_modals",
    "run_on_sentence": "sentence_structure",
    "sentence_fragment": "sentence_structure",
    "run_on": "sentence_structure",
    "fragment": "sentence_structure",
}

# Keyword fallback for any other legacy name (checked in order)
_KEYWORD_RULES = (
    ("agreement", "subject_verb_agreement"),
    ("tense", "verb_tense"),
    ("article", "articles"),
    ("preposition", "prepositions"),
    ("plural", "plurals_and_countability"),
    ("countab", "plurals_and_countability"),
    ("pronoun", "pronouns"),
    ("question", "question_formation"),
    ("order", "word_order"),
    ("modal", "auxiliary_and_modals"),
    ("auxiliar", "auxiliary_and_modals"),
    ("missing", "missing_word"),
    ("collocation", "word_choice"),
    ("word_choice", "word_choice"),
    ("vocab", "word_choice"),
    ("run_on", "sentence_structure"),
    ("fragment", "sentence_structure"),
    ("structure", "sentence_structure"),
    ("compar", "comparatives"),
    ("superlative", "comparatives"),
    ("conditional", "conditionals"),
    ("verb", "verb_form"),
)


def normalize_grammar_category(name: Optional[str]) -> str:
    """Map any grammar pattern name (new or legacy) to a fixed category id."""
    if not name:
        return "other"
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if key in GRAMMAR_CATEGORIES:
        return key
    if key in _LEGACY_GRAMMAR_MAP:
        return _LEGACY_GRAMMAR_MAP[key]
    for needle, category in _KEYWORD_RULES:
        if needle in key:
            return category
    return "other"


def grammar_label(category: str) -> str:
    return GRAMMAR_CATEGORIES.get(category, GRAMMAR_CATEGORIES["other"])[0]


# ── Session metrics ───────────────────────────────────────────────────────────

# better: "higher" | "lower" | "band" (closer to target band is better)
METRIC_INFO: Dict[str, dict] = {
    "talk_time_ratio": {
        "label": "Your share of the conversation",
        "help": "How much of the session you spoke, compared with your tutor.",
        "unit": "%",
        "better": "higher",
    },
    "words_per_turn": {
        "label": "Answer length",
        "help": "Average number of words each time you speak.",
        "unit": "words",
        "better": "higher",
    },
    "average_words_per_sentence": {
        "label": "Sentence length",
        "help": "Average number of English words per sentence.",
        "unit": "words",
        "better": "higher",
    },
    "speaking_rate_wpm": {
        "label": "Speaking pace",
        "help": "Words per minute. 110–150 is a clear, comfortable pace for meetings.",
        "unit": "wpm",
        "better": "band",
        "band": (110.0, 150.0),
    },
    "code_switch_ratio": {
        "label": "Hindi mixed in",
        "help": "Share of your words that were Hindi. Lower means more English-only speaking.",
        "unit": "%",
        "better": "lower",
    },
}
