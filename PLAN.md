# 📚 Learner Progress Knowledge Graph — Implementation Plan

> **Goal:** Build a knowledge-graph-powered system that processes diarized session transcripts and tracks each learner's English-learning progress across sessions and tutors.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 0 — Project Scaffolding & Configuration](#2-phase-0--project-scaffolding--configuration)
3. [Phase 1 — Neo4j Schema & Seed Data](#3-phase-1--neo4j-schema--seed-data)
4. [Phase 2 — Transcript Ingestion Pipeline](#4-phase-2--transcript-ingestion-pipeline)
5. [Phase 3 — Session Metrics Computation](#5-phase-3--session-metrics-computation)
6. [Phase 4 — NLP Analysis (OpenAI)](#6-phase-4--nlp-analysis-openai)
7. [Phase 5 — Vector Embeddings & Semantic Search](#7-phase-5--vector-embeddings--semantic-search)
8. [Phase 6 — Progress Report Generation](#8-phase-6--progress-report-generation)
9. [Phase 7 — API Endpoints](#9-phase-7--api-endpoints)
10. [Phase 8 — Dashboard Frontend](#10-phase-8--dashboard-frontend)
11. [Phase 9 — Testing & Deployment](#11-phase-9--testing--deployment)
12. [File Structure](#12-file-structure)
13. [Dependency List](#13-dependency-list)

---

## 1. Architecture Overview

```
Audio File ──> Whisper Diarization Pipeline ──> Diarized Transcript JSON
                                                        │
                                                        ▼
                                              Transcript Processor
                                                        │
                                                        ▼
                                             OpenAI GPT-4o Analysis
                                          ┌─────────────┼─────────────┐
                                          │             │             │
                                    Grammar Errors  Vocabulary   Goals & Skills
                                          │             │             │
                                          └─────────────┼─────────────┘
                                                        ▼
                                              Neo4j Knowledge Graph
                                                        │
                                                        ▼
                                           Progress Report Generator
                                                        │
                                                        ▼
                                               API  /  Dashboard
```

### Knowledge Graph Schema

```
  ┌──────────┐          ┌──────────┐
  │  Learner │─ATTENDED─▶│ Session  │◀─CONDUCTED─┌──────────┐
  └────┬─────┘          └────┬─────┘             │  Tutor   │
       │                     │                    └──────────┘
       │              COVERED_SKILL──▶ ┌──────────┐
       │              INTRODUCED_WORD─▶│  Skill   │
       │                               └──────────┘
       │
       ├──HAS_GOAL──▶ ┌──────────┐
       │               │   Goal   │
       │               └──────────┘
       │
       ├──MADE_ERROR─▶ ┌────────────────┐
       │               │ GrammarPattern │
       │               └────────────────┘
       │
       └──USED_WORD──▶ ┌──────────┐
                       │   Word   │
                       └──────────┘

  7 Node Types: Learner, Tutor, Session, Goal, GrammarPattern, Word, Skill
```

---

## 2. Phase 0 — Project Scaffolding & Configuration

### Step 0.1 — Environment Variables

Create `.env.example` in project root:

```env
# ─── OpenAI ───────────────────────────────────────────────
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# ─── Neo4j ────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password-here
NEO4J_DATABASE=neo4j

# ─── App Settings ─────────────────────────────────────────
LOG_LEVEL=INFO
CEFR_REFERENCE_PATH=./knowledge_graph/data/cefr_wordlist.json
ROLLING_BASELINE_SESSIONS=5
```

### Step 0.2 — Config Module

Create `knowledge_graph/config.py` using `pydantic-settings`:
- Load all env vars into a typed `Settings` dataclass
- Single import `from knowledge_graph.config import settings` used everywhere

### Step 0.3 — Install New Dependencies

New packages: `neo4j`, `openai`, `pydantic-settings`, `python-dotenv`, `tiktoken`

### Step 0.4 — Neo4j Setup (Docker)

```bash
docker run -d \
  --name neo4j-learner \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-neo4j-password-here \
  -e NEO4J_PLUGINS='["apoc","graph-data-science"]' \
  -v neo4j_data:/data \
  neo4j:5-community
```

Also create a `docker-compose.neo4j.yml` for convenience.

---

## 3. Phase 1 — Neo4j Schema & Seed Data

### Step 1.1 — Constraints & Indexes

Create `knowledge_graph/schema.py` — run once on fresh DB:

```cypher
CREATE CONSTRAINT learner_id   IF NOT EXISTS FOR (l:Learner)        REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT tutor_id     IF NOT EXISTS FOR (t:Tutor)           REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT session_id   IF NOT EXISTS FOR (s:Session)         REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT goal_id      IF NOT EXISTS FOR (g:Goal)            REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT grammar_id   IF NOT EXISTS FOR (gp:GrammarPattern) REQUIRE gp.id IS UNIQUE;
CREATE CONSTRAINT word_lemma   IF NOT EXISTS FOR (w:Word)            REQUIRE w.lemma IS UNIQUE;
CREATE CONSTRAINT skill_name   IF NOT EXISTS FOR (sk:Skill)          REQUIRE sk.name IS UNIQUE;

CREATE VECTOR INDEX session_embedding IF NOT EXISTS
  FOR (s:Session) ON (s.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX goal_embedding IF NOT EXISTS
  FOR (g:Goal) ON (g.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX grammar_embedding IF NOT EXISTS
  FOR (gp:GrammarPattern) ON (gp.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
```

### Step 1.2 — Node Properties

| Node              | Properties                                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| **Learner**       | `id`, `name`, `native_language`, `current_cefr_level`, `created_at`                                |
| **Tutor**         | `id`, `name`, `specialization`, `created_at`                                                       |
| **Session**       | `id`, `learner_id`, `tutor_id`, `date`, `duration_sec`, `talk_time_ratio`, `words_per_turn`, `speaking_rate_wpm`, `code_switch_ratio`, `transcript_summary`, `embedding` |
| **Goal**          | `id`, `description`, `category`, `embedding`, `created_at`                                         |
| **GrammarPattern**| `id`, `pattern_name`, `description`, `example_wrong`, `example_correct`, `embedding`               |
| **Word**          | `lemma`, `cefr_level`, `pos` (part of speech)                                                      |
| **Skill**         | `name`, `category` (e.g. reading, speaking, vocabulary, fluency)                                   |

### Step 1.3 — Edge (Relationship) Properties

| Relationship                                 | Properties                                                                                                     |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `(Learner)-[:ATTENDED]->(Session)`           | `role: "learner"`                                                                                              |
| `(Tutor)-[:CONDUCTED]->(Session)`            | `role: "tutor"`                                                                                                |
| `(Session)-[:COVERED_SKILL]->(Skill)`        | `focus_duration_sec`, `notes`                                                                                  |
| `(Session)-[:INTRODUCED_WORD]->(Word)`       | `context_sentence`                                                                                             |
| `(Learner)-[:HAS_GOAL]->(Goal)`             | `status` in {mentioned, practiced, demonstrated}, `updated_at`                                                 |
| `(Learner)-[:MADE_ERROR]->(GrammarPattern)` | `session_id`, `error_count`, `example_utterance`, `date`                                                       |
| `(Learner)-[:USED_WORD]->(Word)`            | `session_id`, `correct`, `spontaneous`, `mastery` in {introduced, prompted, spontaneous}, `date`               |

### Step 1.4 — CEFR Reference Data

Create `knowledge_graph/data/cefr_wordlist.json` — start with ~3000 curated words with CEFR levels (A1-C2). For unknown words, call OpenAI to estimate.

---

## 4. Phase 2 — Transcript Ingestion Pipeline

### Step 2.1 — Neo4j Client Wrapper (`knowledge_graph/db.py`)
- Singleton Neo4j driver using `neo4j.GraphDatabase.driver()`
- Context-manager for sessions
- Helper methods: `run_query()`, `run_write()`, `close()`

### Step 2.2 — Transcript Parser (`knowledge_graph/ingest/parser.py`)
- Input: `session_stt.json` (our Whisper pipeline output)
- Parse each segment: extract `speaker`, `text`, `start`, `end`, `words[]`
- Separate speakers into learner vs tutor (via config or heuristic)
- Output: `ParsedTranscript` dataclass with separated learner/tutor turns

### Step 2.3 — Session Ingestion Orchestrator (`knowledge_graph/ingest/orchestrator.py`)
```python
def ingest_session(transcript_path, learner_id, tutor_id, session_date) -> session_id:
    # 1. Parse transcript
    # 2. Compute session metrics (Phase 3)
    # 3. Run NLP analysis via OpenAI (Phase 4)
    # 4. Write everything to Neo4j
```

---

## 5. Phase 3 — Session Metrics Computation

### Step 3.1 — Extend Existing Metrics (`knowledge_graph/metrics.py`)

Builds on `core/metrics_utils.py`:

| Metric                    | Computation                                                          |
| ------------------------- | -------------------------------------------------------------------- |
| **Talk-time ratio**       | `learner_talk_sec / total_talk_sec` (already in AudioMetrics)        |
| **Words per turn**        | Count words in each learner segment, compute mean                    |
| **Speaking rate (WPM)**   | `total_learner_words / learner_talk_minutes`                         |
| **Code-switch ratio**     | Hindi/Hinglish tokens via regex + langdetect / total_learner_tokens  |

### Step 3.2 — Hindi/Hinglish Detection
- Curated Hindi word list + Devanagari regex
- OpenAI fallback for uncertain tokens
- Store as `code_switch_ratio` on Session node

---

## 6. Phase 4 — NLP Analysis (OpenAI)

### Step 4.1 — LLM Extraction Service (`knowledge_graph/llm/extractor.py`)

Uses OpenAI structured output (JSON mode) to extract:

1. `grammar_errors`: list of {pattern, example_wrong, example_correct, count}
2. `vocabulary`: list of {word, context_sentence, introduced_by_tutor, used_by_learner, correct, spontaneous}
3. `goals_discussed`: list of {description, status: mentioned|practiced|demonstrated}
4. `skills_covered`: list of {name, category}
5. `session_summary`: 2-3 sentence summary

### Step 4.2 — CEFR Level Lookup
1. Check `cefr_wordlist.json`
2. If not found, call OpenAI for classification
3. Cache result back

### Step 4.3 — Determine Mastery State

| Condition                                   | Mastery       |
| ------------------------------------------- | ------------- |
| Tutor introduced, learner hasn't used       | `introduced`  |
| Learner used after tutor prompt             | `prompted`    |
| Learner used without prompting              | `spontaneous` |

### Step 4.4 — Graph Writer (`knowledge_graph/ingest/graph_writer.py`)
- Cypher MERGE queries to upsert nodes/relationships
- Transactional writes for atomicity

---

## 7. Phase 5 — Vector Embeddings & Semantic Search

### Step 5.1 — Embedding Generation (`knowledge_graph/llm/embeddings.py`)
- Use OpenAI `text-embedding-3-small` (1536 dimensions)
- Embed: Session summaries, Goal descriptions, GrammarPattern descriptions
- Store on respective nodes

### Step 5.2 — Semantic Search (`knowledge_graph/search.py`)
- `find_similar_errors(pattern_text, top_k)` — group similar grammar mistakes
- `find_similar_sessions(summary, top_k)` — find related past sessions
- `find_tutors_for_learner_profile(learner_id, top_k)` — match tutors to learners
- Uses Neo4j vector index: `db.index.vector.queryNodes()`

---

## 8. Phase 6 — Progress Report Generation

### Step 6.1 — Data Aggregation (`knowledge_graph/reports/aggregator.py`)
Query Neo4j for learner's last N sessions:
- Metrics trend over time (talk_time, wpm, words/turn, code-switch)
- Rolling baseline (avg of each metric)
- New words introduced/mastered
- Fading vs persistent grammar errors
- Goal status transitions

### Step 6.2 — Trend Detection
- **Improving**: current > baseline x 1.1
- **Stable**: within +/- 10%
- **Declining**: current < baseline x 0.9

### Step 6.3 — LLM Summary (`knowledge_graph/reports/generator.py`)
- Feed aggregated data to GPT-4o
- Generate: 3-4 sentence progress summary + "Practice Next" recommendation

### Step 6.4 — Report JSON Output
```json
{
  "learner_id": "learner_001",
  "learner_name": "Kiran Paunikar",
  "report_date": "2026-10-01",
  "sessions_analyzed": 5,
  "metrics": {
    "talk_time_ratio":   {"current": 0.42, "baseline": 0.38, "trend": "improving"},
    "words_per_turn":    {"current": 12.3, "baseline": 10.1, "trend": "improving"},
    "speaking_rate_wpm": {"current": 78,   "baseline": 72,   "trend": "improving"},
    "code_switch_ratio": {"current": 0.05, "baseline": 0.08, "trend": "improving"}
  },
  "vocabulary": {
    "new_words": ["acquire", "harnessing", "deriving"],
    "mastery_upgrades": [{"word": "enthusiasm", "from": "prompted", "to": "spontaneous"}]
  },
  "grammar": {
    "fading_errors":     [{"pattern": "tense_consistency", "trend": [5, 3, 1]}],
    "persistent_errors": [{"pattern": "article_usage",     "trend": [3, 4, 3]}]
  },
  "goals": [{"description": "Introduce self fluently", "status": "practiced"}],
  "llm_summary": "Kiran has shown consistent improvement in speaking fluency...",
  "practice_next": "Focus on article usage (a/an/the) when describing objects."
}
```

---

## 9. Phase 7 — API Endpoints

Add `api/knowledge_graph_routes.py`, include in `api/server.py`:

| Method | Endpoint                       | Description                                     |
| ------ | ------------------------------ | ----------------------------------------------- |
| POST   | `/kg/ingest`                   | Ingest a session transcript into the KG          |
| GET    | `/kg/learners`                 | List all learners                                |
| GET    | `/kg/learners/{id}`            | Learner details + current stats                  |
| GET    | `/kg/learners/{id}/report`     | Generate progress report (last N sessions)       |
| GET    | `/kg/learners/{id}/sessions`   | List sessions for a learner                      |
| GET    | `/kg/sessions/{id}`            | Session details with metrics                     |
| GET    | `/kg/search/errors`            | Semantic search for similar grammar errors       |
| GET    | `/kg/search/sessions`          | Semantic search for similar sessions             |

Create Pydantic request/response models in `api/models/kg_models.py`.

---

## 10. Phase 8 — Dashboard Frontend

Build in `api/static/` (extend existing frontend):

- **Learner Selector** — dropdown to pick a learner
- **Metrics Cards** — talk-time ratio, WPM, words/turn, code-switch with trend arrows
- **Vocabulary Panel** — words table with CEFR badges and mastery status
- **Grammar Panel** — error patterns with sparkline trends
- **Goal Tracker** — progress bars (mentioned -> practiced -> demonstrated)
- **Session Timeline** — scrollable session list with highlights
- **AI Summary** — LLM-generated progress summary + "practice next"
- **Graph Visualizer** — interactive subgraph view using `neovis.js` or `d3-force`

---

## 11. Phase 9 — Testing & Deployment

### Unit Tests
- Metric calculations with known fixtures
- Cypher queries against test Neo4j
- OpenAI extraction with mocked responses

### Integration Test
- End-to-end: audio -> Whisper -> KG ingestion -> report
- Use `session_stt.json` as golden fixture

### Docker Compose
- Add Neo4j to existing compose setup
- Pass through OpenAI keys

---

## 12. File Structure

```
whisper-diarization-advanced/
├── .env.example
├── docker-compose.neo4j.yml
├── PLAN.md
├── core/                                 # Existing STT pipeline
│   ├── pipeline.py
│   ├── preprocess.py
│   └── metrics_utils.py
├── api/                                  # Existing API
│   ├── server.py                         # Extended with KG routes
│   ├── knowledge_graph_routes.py         # NEW
│   ├── models/
│   │   └── kg_models.py                  # NEW
│   └── static/                           # Extended dashboard
├── knowledge_graph/                      # NEW MODULE
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── schema.py
│   ├── metrics.py
│   ├── search.py
│   ├── data/
│   │   └── cefr_wordlist.json
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── orchestrator.py
│   │   └── graph_writer.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   └── embeddings.py
│   └── reports/
│       ├── __init__.py
│       ├── aggregator.py
│       └── generator.py
└── tests/
    ├── test_metrics.py
    ├── test_parser.py
    ├── test_graph_writer.py
    └── test_report.py
```

---

## 13. Dependency List

| Package            | Purpose                    | Version            |
| ------------------ | -------------------------- | ------------------ |
| `neo4j`            | Graph database driver      | `>=5.20.0`         |
| `openai`           | LLM extraction + embeddings | `>=1.30.0`       |
| `pydantic-settings`| Config from .env           | `>=2.2.0`          |
| `python-dotenv`    | Load .env file             | `>=1.0.0`          |
| `tiktoken`         | Token counting for prompts | `>=0.7.0`          |
| `numpy`            | Numerical operations       | already installed  |
| `pandas`           | Data manipulation          | already installed  |

---

## Notes

- **Implementation order:** Phases 0-1-2-3-4 are sequential. Phases 5-8 can overlap once Phase 4 is done.
- **Cost:** ~$0.01-0.03 per session ingestion with GPT-4o. Batch embeddings where possible.
- **Existing code:** Reuses `core/metrics_utils.py` for talk-time computation, `session_stt.json` format as input.
