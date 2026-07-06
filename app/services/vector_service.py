"""
Vector Service — semantic retrieval over ingested document chunks via Qdrant
+ sentence-transformers. Degrades gracefully to a no-op (returns 0 / []) if
Qdrant isn't reachable or the optional dependencies aren't installed —
callers never need to check availability themselves.
"""

import uuid
from typing import List, Optional

from loguru import logger

from app.config import get_settings
from app.services.nlp_service import _chunk_text

settings = get_settings()
COLLECTION_NAME = "neorag_documents"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

_client = None
_embedder = None
_unavailable = False  # set once we know Qdrant/sentence-transformers can't be used


def get_client():
    global _client, _unavailable
    if _client is not None or _unavailable:
        return _client
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        try:
            _client.get_collection(COLLECTION_NAME)
        except Exception:
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
    except Exception as e:
        logger.warning(f"Qdrant unavailable: {e}. Semantic search will be disabled.")
        _client = None
        _unavailable = True
    return _client


def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"sentence-transformers unavailable: {e}")
    return _embedder


def index_document(doc_id: str, title: str, content: str) -> int:
    """Chunks and indexes a document for later semantic retrieval. Returns
    the number of chunks indexed (0 if Qdrant/embedder unavailable)."""
    client = get_client()
    if not client:
        return 0
    embedder = get_embedder()
    if not embedder:
        return 0

    from qdrant_client.http.models import PointStruct

    chunks = _chunk_text(content, chunk_size=500, overlap=50)
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"doc_id": doc_id, "title": title, "chunk_index": i, "text": chunk},
        )
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def semantic_search(query: str, top_k: int = 5) -> List[str]:
    """Returns the top_k most semantically similar document chunks to `query`,
    or [] if Qdrant/embedder are unavailable."""
    client = get_client()
    if not client:
        return []
    embedder = get_embedder()
    if not embedder:
        return []

    query_vec = embedder.encode([query], show_progress_bar=False)[0].tolist()
    hits = client.search(collection_name=COLLECTION_NAME, query_vector=query_vec, limit=top_k)
    return [h.payload["text"] for h in hits]
