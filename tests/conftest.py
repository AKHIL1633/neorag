import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.core.pg import get_db_session
from app.main import app
from app.models.user_model import Base


@pytest.fixture(scope="session", autouse=True)
def _test_llm():
    """Uses a small local model (distilgpt2, ~350MB) instead of the
    production default (facebook/opt-1.3b, ~5GB float32) for the test
    session — this environment doesn't have enough free RAM to load the
    full model without heavy swapping. Mutates the cached Settings
    singleton in place so every module that already holds a reference to
    it (e.g. `settings = get_settings()` at import time in llm_service.py)
    sees the override too."""
    settings = get_settings()
    settings.llm_model_name = "distilgpt2"
    settings.llm_fallback_model = "distilgpt2"
    # CPU generation time scales ~linearly with max_new_tokens — 512 takes
    # ~45s per call even on the small test model; 24 keeps each call to a
    # few seconds, which matters a lot with 20 benchmark questions x a
    # multi-call agent pipeline with retries.
    settings.llm_max_new_tokens = 24
    yield


@pytest.fixture(scope="session", autouse=True)
def _test_db():
    """Overrides the Postgres-backed auth dependency with an in-memory
    SQLite database for the whole test session — tests must not require a
    real Postgres instance to run."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)

    def _override():
        session = TestSessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = _override
    yield
    app.dependency_overrides.pop(get_db_session, None)
    test_engine.dispose()


@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def token(test_client):
    # Not asserted — may already be registered from a prior test module
    # sharing the same session-scoped in-memory DB.
    test_client.post(
        "/api/v1/auth/register",
        json={"username": "admin", "email": "admin@example.com", "password": "TestPass123"},
    )
    r = test_client.post("/api/v1/auth/token", data={"username": "admin", "password": "TestPass123"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]
