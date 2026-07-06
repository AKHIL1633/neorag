# NeoRAG — Knowledge Graph + LLM Q&A API

> A FastAPI backend built with Neo4j, LangGraph, LangChain, Qdrant, and PyTorch that transforms
> unstructured text and JSON into a queryable knowledge graph, then answers questions over it
> through an agentic pipeline with a self-critique retry loop.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Neo4j](https://img.shields.io/badge/Neo4j-5.20-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 What is NeoRAG?

NeoRAG is a knowledge graph-powered Retrieval Augmented Generation (RAG) system that:

1. **Ingests** text documents and semi-structured JSON
2. **Extracts** named entities via HuggingFace Transformers (with a spaCy fallback), sentiment, and simplified coreference resolution
3. **Builds** a Neo4j knowledge graph of entities and their relationships, and indexes document chunks into Qdrant for semantic search
4. **Answers** questions through a **LangGraph agent**: classify the question → extract entities → retrieve graph context → retrieve semantic (vector) context → synthesize an answer → **self-critique the answer with an LLM-as-judge, and retry retrieval with a widened search if the score is too low** (bounded to 2 retries)
5. **Guards** every LLM call with input screening (prompt-injection patterns, length limits) and output PII redaction
6. **Exposes** everything via a documented REST API, with real user accounts (register/login backed by PostgreSQL) instead of a hardcoded demo user

---

## 🏗️ Architecture

```
                          ┌─────────────────────────────┐
  POST /llm/ask  ───────▶ │        FastAPI app           │
                          │  (rate-limited, request-ID   │
                          │   middleware, PII guardrails) │
                          └──────────────┬───────────────┘
                                         │
                              LangGraph agent pipeline
                                         │
        ┌──────────┬──────────┬─────────┼──────────┬───────────┬────────────┐
        ▼          ▼          ▼         ▼          ▼           ▼            │
  understand   extract-    retrieve-  retrieve-  synthesize  critique       │
  question     entities    graph      vectors    (LLM)       (LLM-judge)    │
                  │           │          │                       │          │
                  ▼           ▼          ▼                       │          │
              HuggingFace   Neo4j     Qdrant +          score < 0.6?        │
              NER/spaCy     Graph DB  sentence-           retry (max 2) ────┘
                                      transformers         with widened
                                                            search
```

Auth, NLP, graph, ingestion, LLM Q&A, and evaluation each live in their own service module — see [Project Structure](#️-project-structure) below.

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/AKHIL1633/neorag
cd neorag

cp .env.example .env
# Edit .env — at minimum set SECRET_KEY (openssl rand -hex 32) and HUGGINGFACE_TOKEN if using a gated model

docker-compose up -d
# Starts Neo4j, PostgreSQL, Redis, Qdrant, and the API

docker-compose exec app alembic upgrade head
# Creates the users table

# API is live at: http://localhost:8000
# Swagger docs:   http://localhost:8000/docs
# Neo4j browser:  http://localhost:7474
```

### Option 2 — Local Development

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start dependencies (or point at your own via .env)
docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.20
docker run -d -p 5432:5432 -e POSTGRES_USER=neorag -e POSTGRES_PASSWORD=neorag -e POSTGRES_DB=neorag postgres:16-alpine
docker run -d -p 6333:6333 qdrant/qdrant:v1.9.2

cp .env.example .env
alembic upgrade head

uvicorn app.main:app --reload --port 8000
```

Then register a user and log in:

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "email": "alice@example.com", "password": "AlicePass123"}'

curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=alice&password=AlicePass123"
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/register` | Create a user account (first user ever registered is auto-promoted to admin) |
| `POST` | `/api/v1/auth/token` | Get JWT token |
| `POST` | `/api/v1/nlp/analyze` | NER + Sentiment + Coreference |
| `POST` | `/api/v1/nlp/entities` | Named entity extraction only |
| `POST` | `/api/v1/nlp/sentiment` | Sentiment analysis only |
| `POST` | `/api/v1/documents/ingest` | Ingest text → knowledge graph + vector index |
| `POST` | `/api/v1/documents/ingest-json` | Ingest JSON → knowledge graph |
| `GET`  | `/api/v1/graph/stats` | Knowledge graph statistics |
| `POST` | `/api/v1/graph/query` | Run an allowlisted Cypher query on Neo4j |
| `GET`  | `/api/v1/graph/entity/{name}/context` | Get entity relationships |
| `POST` | `/api/v1/graph/node` | Create entity node (label must be on the allowlist) |
| `POST` | `/api/v1/graph/relationship` | Create relationship (type must be on the allowlist) |
| `POST` | `/api/v1/llm/ask` | Agentic Q&A: graph + vector retrieval, synthesis, self-critique (rate-limited: 30/min) |
| `POST` | `/api/v1/llm/summarize` | Text summarization (rate-limited: 30/min) |
| `POST` | `/api/v1/llm/generate` | Free-form text generation (rate-limited: 30/min) |
| `POST` | `/api/v1/eval/run` | Run the 20-question benchmark, return per-question + mean judge scores |
| `GET`  | `/health` | Reports real Neo4j connectivity + whether NER/sentiment models have loaded |

---

## 🔧 Usage Examples

### 1. Register and get a token

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "email": "alice@example.com", "password": "AlicePass123"}'

curl -X POST http://localhost:8000/api/v1/auth/token -d "username=alice&password=AlicePass123"
# Response: {"access_token": "eyJ...", "token_type": "bearer"}
```

### 2. Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Tech Companies",
    "content": "Elon Musk founded SpaceX in 2002 and Tesla in 2003. Both companies are based in Texas. Sundar Pichai leads Google, headquartered in Mountain View, California."
  }'
```

This builds the Neo4j graph **and** chunks + embeds the document into Qdrant for semantic retrieval.

### 3. Ask a Question (Agentic Q&A)

```bash
curl -X POST http://localhost:8000/api/v1/llm/ask \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What companies did Elon Musk found?",
    "use_graph": true,
    "max_tokens": 256
  }'
```

Runs the LangGraph pipeline: entity extraction → graph + vector retrieval → synthesis → self-critique (retries with a widened search if the judge scores the answer below 0.6, up to 2 times).

### 4. Query the Knowledge Graph (Cypher)

```bash
curl -X POST http://localhost:8000/api/v1/graph/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "cypher": "MATCH (p:PERSON)-[:WORKS_FOR]->(o:ORG) RETURN p.name, o.name LIMIT 10"
  }'
```

### 5. Run the Evaluation Benchmark

```bash
curl -X POST http://localhost:8000/api/v1/eval/run -H "Authorization: Bearer <token>"
```

Ingests 20 hand-curated seed documents, asks the paired question for each, and scores the answer with an LLM-as-judge — see [Honest Metrics](#-honest-metrics) for what this actually measured.

---

## 🗂️ Project Structure

```
neorag/
├── app/
│   ├── main.py                  # FastAPI app + lifespan, CORS, rate limiter, request-ID middleware
│   ├── config.py                # Pydantic v2 Settings
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response DTOs
│   │   └── user_model.py        # SQLAlchemy User model
│   ├── core/
│   │   ├── auth.py              # JWT + Postgres-backed register/authenticate
│   │   ├── pg.py                # SQLAlchemy engine + session factory
│   │   ├── database.py          # Neo4j connection manager
│   │   ├── guardrails.py        # Prompt-injection input screening + PII output redaction
│   │   ├── observability.py     # Structured JSON tracing per LLM call, request-ID correlation
│   │   └── rate_limit.py        # Shared slowapi Limiter instance
│   ├── services/
│   │   ├── nlp_service.py       # NER + Sentiment + Coreference (chunked, not truncated to 512 chars)
│   │   ├── graph_service.py     # Neo4j CRUD + Cypher + graph builder (allowlisted labels/relations)
│   │   ├── llm_service.py       # LLM loading + direct-call fallback path (LCEL)
│   │   ├── agent_pipeline.py    # LangGraph agent: understand → extract → retrieve (graph+vector) → synthesize → critique
│   │   ├── vector_service.py    # Qdrant + sentence-transformers semantic search
│   │   ├── evaluation.py        # LLM-as-judge scorer + the 20-question benchmark
│   │   └── ingestion.py         # Text/JSON extraction → NLP → graph → vector index pipeline
│   └── api/routes/
│       ├── auth.py, nlp.py, graph.py, documents.py, qa.py, eval.py
├── alembic/                      # DB migrations for the users table
├── tests/
│   ├── conftest.py               # In-memory SQLite override for auth; lightweight LLM override for tests
│   ├── test_api.py               # 22 tests with real behavioral assertions (not just "did it return 200")
│   └── test_evaluation.py        # The 20-question benchmark (see Honest Metrics)
├── .github/workflows/ci.yml
├── docker-compose.yml            # Neo4j + PostgreSQL + Redis + Qdrant + App, all with healthchecks
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🎯 Honest Metrics

Ran via `pytest tests/test_evaluation.py` — a hand-curated 20-question benchmark (`app/services/evaluation.py::QA_BENCHMARK`). Each item establishes a `PERSON → ORG` fact via a seed document, asks a question mentioning the person, and checks whether the organization surfaces in the agent's answer; an LLM-as-judge separately scores each answer 0.0–1.0 against the retrieved context.

| Metric | Result |
|---|---|
| Mean judge score | **0.72** |
| Entity hit rate | **0.35** (7/20) |
| Questions | 20 |

**Methodology note (read before citing these numbers):** this run used `distilgpt2` as both the answer-generator and the judge, not the configured production default (`facebook/opt-1.3b`) — the sandboxed environment this was measured in has only ~3.7GB of free RAM, not enough to comfortably load a 5GB+ float32 model without heavy swapping. `distilgpt2` is a much weaker model, so these numbers are a **floor**, not a ceiling — expect a real deployment on adequate hardware with `facebook/opt-1.3b` (or a stronger model) to score higher on both metrics. Re-run the benchmark yourself via `POST /api/v1/eval/run` or `pytest tests/test_evaluation.py -v -s` against your own configured model to get numbers representative of your deployment.

No claim on this page is asserted without this benchmark (or the test suite) backing it — there is no "70% hallucination reduction," "92% accuracy," or "60% latency improvement" claimed anywhere, because no such baseline was ever measured.

---

## 🤖 LLM Configuration

By default, NeoRAG uses `facebook/opt-1.3b` (no token needed, runs on CPU — needs ~5GB+ free RAM to load comfortably).

**To use Llama 3.2** (or another gated model):

1. Get access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Get your HF token: https://huggingface.co/settings/tokens
3. Update `.env`:
```env
HUGGINGFACE_TOKEN=hf_your_token_here
LLM_MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
```

**Running on constrained hardware:** set `LLM_MODEL_NAME` and `LLM_FALLBACK_MODEL` to something smaller (e.g. `distilgpt2`), and lower `LLM_MAX_NEW_TOKENS` — CPU generation time scales roughly linearly with it. This is exactly what `tests/conftest.py` does for the automated test suite.

---

## 🧪 Running Tests

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest tests/ -v
```

`tests/conftest.py`:
- Overrides the Postgres-backed auth dependency with an in-memory SQLite database for the whole session — no real Postgres needed to run the suite.
- Overrides the LLM to `distilgpt2` with a small `max_new_tokens`, so the suite doesn't require ~5GB of free RAM or take an impractically long time per call.

Neo4j and Qdrant are optional for testing — both services fall back gracefully (in-memory graph storage; semantic search disabled) when unreachable, and the test suite exercises both paths.

`tests/test_evaluation.py::test_benchmark_accuracy` runs the full 20-question benchmark and takes several minutes (each question round-trips through NER, graph retrieval, LLM synthesis, and LLM-as-judge scoring, with up to 2 retries) — see [Honest Metrics](#-honest-metrics) for what it measures and why.

---

## 🌐 Deployment on AWS

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
docker build -t neorag .
docker tag neorag:latest <ecr-url>/neorag:latest
docker push <ecr-url>/neorag:latest

docker-compose -f docker-compose.yml up -d
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **API Framework** | FastAPI 0.111 |
| **Agent Orchestration** | LangGraph 0.2 (self-critique retry loop) |
| **LLM Orchestration** | LangChain 0.2 (LCEL) |
| **LLM** | HuggingFace (`facebook/opt-1.3b` default; swap for Llama 3.2 or others) |
| **Graph Database** | Neo4j 5.20 + Cypher (allowlisted labels/relationship types) |
| **Vector Database** | Qdrant + sentence-transformers (`all-MiniLM-L6-v2`) |
| **Auth Database** | PostgreSQL + SQLAlchemy + Alembic migrations |
| **NLP / NER** | HuggingFace Transformers + spaCy fallback |
| **Sentiment Analysis** | RoBERTa (cardiffnlp) |
| **Deep Learning** | PyTorch |
| **Authentication** | JWT (python-jose) + bcrypt, real Postgres-backed user accounts |
| **Rate Limiting** | slowapi |
| **Observability** | Structured JSON tracing per LLM call, request-ID correlation via loguru |
| **Guardrails** | Prompt-injection input screening + PII output redaction |
| **Containerisation** | Docker + Docker Compose (Neo4j, PostgreSQL, Redis, Qdrant, App — all with healthchecks) |
| **CI/CD** | GitHub Actions |
| **Cloud** | AWS (EC2, ECS, Lambda) |

Redis is provisioned in `docker-compose.yml` and `.env.example` for future async/background-task use, but isn't yet consumed by any application code.

---

## 👤 Author

**P Akhil** — AI/ML Engineer
- GitHub: [@AKHIL1633](https://github.com/AKHIL1633)
- LinkedIn: [linkedin.com/in/p-akhil](https://linkedin.com/in/p-akhil)
- Email: akhil.kartik371@gmail.com

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
