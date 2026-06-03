"""
Unit tests for NeoRAG API
Run: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Helper ────────────────────────────────────────────────────────────────────

def get_token():
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "NeoRAG" in r.json()["name"]


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_login_success():
    r = client.post("/api/v1/auth/token",
                    data={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    assert "access_token" in r.json()

def test_login_invalid():
    r = client.post("/api/v1/auth/token",
                    data={"username": "wrong", "password": "wrong"})
    assert r.status_code == 401

def test_protected_without_token():
    r = client.post("/api/v1/nlp/analyze", json={"text": "hello"})
    assert r.status_code == 401


# ── NLP ───────────────────────────────────────────────────────────────────────

def test_nlp_analyze():
    token = get_token()
    r = client.post(
        "/api/v1/nlp/analyze",
        json={"text": "Apple Inc was founded by Steve Jobs in Cupertino, California."},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    data = r.json()
    assert "entities"   in data
    assert "sentiment"  in data
    assert "word_count" in data

def test_nlp_empty_text():
    token = get_token()
    r = client.post(
        "/api/v1/nlp/analyze",
        json={"text": ""},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 400

def test_sentiment_positive():
    token = get_token()
    r = client.post(
        "/api/v1/nlp/sentiment",
        json={"text": "This is an excellent and amazing product!"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    assert "label" in r.json()


# ── Document Ingestion ────────────────────────────────────────────────────────

def test_ingest_document():
    token = get_token()
    r = client.post(
        "/api/v1/documents/ingest",
        json={
            "title": "Test Document",
            "content": "Elon Musk is the CEO of Tesla and SpaceX, based in Austin, Texas.",
            "metadata": {"source": "test"}
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    data = r.json()
    assert "id"              in data
    assert "entity_count"    in data
    assert "sentiment"       in data

def test_ingest_json():
    token = get_token()
    r = client.post(
        "/api/v1/documents/ingest-json",
        json={
            "json_data": {
                "name": "Sundar Pichai",
                "company": "Google",
                "city": "Mountain View",
                "products": ["Search", "Gmail", "YouTube"]
            },
            "extract_entities": True,
            "build_graph": True
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    data = r.json()
    assert "entities_found"  in data
    assert "nodes_created"   in data


# ── Graph ─────────────────────────────────────────────────────────────────────

def test_graph_stats():
    token = get_token()
    r = client.get(
        "/api/v1/graph/stats",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    data = r.json()
    assert "total_nodes"         in data
    assert "total_relationships" in data

def test_cypher_dangerous_blocked():
    token = get_token()
    r = client.post(
        "/api/v1/graph/query",
        json={"cypher": "MATCH (n) DELETE n"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 403


# ── LLM ───────────────────────────────────────────────────────────────────────

def test_ask_question():
    token = get_token()
    r = client.post(
        "/api/v1/llm/ask",
        json={"question": "Who founded Apple?", "use_graph": True, "max_tokens": 256},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer"   in data
    assert "question" in data

def test_summarize():
    token = get_token()
    r = client.post(
        "/api/v1/llm/summarize",
        json={
            "text": "FastAPI is a modern, fast web framework for building APIs with Python. "
                    "It is based on standard Python type hints and provides automatic "
                    "interactive documentation. FastAPI is one of the fastest Python frameworks.",
            "max_length": 50
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    assert "summary" in r.json()
