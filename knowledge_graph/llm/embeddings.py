"""
knowledge_graph/llm/embeddings.py
──────────────────────────────────
Generate and retrieve OpenAI vector embeddings.

Uses text-embedding-3-small (1536 dimensions, ~$0.00002/1K tokens).

Usage
-----
    from knowledge_graph.llm.embeddings import embed_text, embed_batch

    vec = embed_text("Kiran struggled with article usage today.")
    vecs = embed_batch(["sentence one", "sentence two"])
"""
from __future__ import annotations

import logging
from typing import List, Optional

from openai import OpenAI

from knowledge_graph.config import settings

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None
_EMBEDDING_DIM = 1536


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_text(text: str) -> List[float]:
    """Return a 1536-dim embedding vector for *text*."""
    if not text or not text.strip():
        return [0.0] * _EMBEDDING_DIM

    response = _get_client().embeddings.create(
        model=settings.openai_embedding_model,
        input=text.strip(),
    )
    return response.data[0].embedding


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a batch of texts in a single API call."""
    if not texts:
        return []

    clean = [t.strip() if t else " " for t in texts]
    response = _get_client().embeddings.create(
        model=settings.openai_embedding_model,
        input=clean,
    )
    # API returns results in the same order as input
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
