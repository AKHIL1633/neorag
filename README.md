# NeoRAG — Knowledge Graph + LLM Q&A API

> **Production-grade AI backend** built with FastAPI, Neo4j, Llama 3.2, LangChain, and PyTorch.
> Transforms unstructured text and JSON into queryable knowledge graphs with LLM-powered Q&A.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Neo4j](https://img.shields.io/badge/Neo4j-5.20-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 What is NeoRAG?

NeoRAG is a knowledge graph-powered Retrieval Augmented Generation (RAG) system that:

1. **Ingests** text documents and semi-structured JSON
2. **Extracts** named entities using Stanford NLP-equivalent techniques (NER, sentiment, coreference)
3. **Builds** a Neo4j knowledge graph of entities and their relationships
4. **Answers** questions using Llama 3.2 grounded by graph context
5. **Exposes** everything via a clean, documented REST API

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                         │
│  /auth  /nlp  /documents  /graph  /llm                      │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────────┐  ┌──────────────┐
   │ NLP      │   │ Graph        │  │ LLM          │
   │ Service  │   │ Service      │  │ Service      │
   │          │   │              │  │              │
   │ • NER    │   │ • Neo4j CRUD │  │ • Llama 3.2  │
   │ • Senti- │   │ • Cypher     │  │ • LangChain  │
   │   ment   │   │ • Graph      │  │ • Q&A        │
   │ • Coref  │   │   Builder    │  │ • Summarize  │
   └──────────┘   └──────────────┘  └──────────────┘
         │               │               │
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────────┐  ┌──────────────┐
   │HuggingFace│  │   Neo4j      │  │  HuggingFace │
   │ + spaCy  │   │   Graph DB   │  │  Llama 3.2   │
   └──────────┘   └──────────────┘  └──────────────┘
```

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/AKHIL1633/neorag
cd neorag

# Copy environment config
cp .env.example .env
# Edit .env with your HuggingFace token

# Start all services (Neo4j + Redis + PostgreSQL + App)
docker-compose up -d

# API is live at: http://localhost:8000
# Swagger docs:   http://localhost:8000/docs
# Neo4j browser:  http://localhost:7474
```

### Option 2 — Local Development

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start Neo4j (Docker)
docker run -d -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5.20

# Copy and configure env
cp .env.example .env

# Run the API
uvicorn app.main:app --reload --port 8000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/token` | Get JWT token |
| `POST` | `/api/v1/nlp/analyze` | NER + Sentiment + Coreference |
| `POST` | `/api/v1/nlp/entities` | Named entity extraction only |
| `POST` | `/api/v1/nlp/sentiment` | Sentiment analysis only |
| `POST` | `/api/v1/documents/ingest` | Ingest text → knowledge graph |
| `POST` | `/api/v1/documents/ingest-json` | Ingest JSON → knowledge graph |
| `GET`  | `/api/v1/graph/stats` | Knowledge graph statistics |
| `POST` | `/api/v1/graph/query` | Run Cypher query on Neo4j |
| `GET`  | `/api/v1/graph/entity/{name}/context` | Get entity relationships |
| `POST` | `/api/v1/graph/node` | Create entity node |
| `POST` | `/api/v1/graph/relationship` | Create relationship |
| `POST` | `/api/v1/llm/ask` | Graph-grounded Q&A (Llama 3.2) |
| `POST` | `/api/v1/llm/summarize` | Text summarization |
| `POST` | `/api/v1/llm/generate` | Free-form text generation |

---

## 🔧 Usage Examples

### 1. Get Authentication Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=admin123"

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

### 3. Ingest JSON Data

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest-json \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "json_data": {
      "name": "Tim Cook",
      "company": "Apple Inc",
      "city": "Cupertino",
      "products": ["iPhone", "MacBook", "iPad"]
    },
    "extract_entities": true,
    "build_graph": true
  }'
```

### 4. Ask a Question (Graph-Grounded)

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

### 5. Query the Knowledge Graph (Cypher)

```bash
curl -X POST http://localhost:8000/api/v1/graph/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "cypher": "MATCH (p:PERSON)-[:WORKS_FOR]->(o:ORG) RETURN p.name, o.name LIMIT 10"
  }'
```

### 6. Run NLP Analysis

```bash
curl -X POST http://localhost:8000/api/v1/nlp/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple Inc was founded by Steve Jobs and Steve Wozniak in 1976 in Cupertino, California. He later returned as CEO."
  }'

# Returns: entities (NER), sentiment, coreference hints ("He" → "Steve Jobs")
```

---

## 🗂️ Project Structure

```
neorag/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── config.py                # Pydantic settings
│   ├── models/
│   │   └── schemas.py           # All Pydantic request/response models
│   ├── core/
│   │   ├── auth.py              # JWT authentication
│   │   └── database.py          # Neo4j connection manager
│   ├── services/
│   │   ├── nlp_service.py       # NER + Sentiment + Coreference
│   │   ├── graph_service.py     # Neo4j CRUD + Cypher + graph builder
│   │   ├── llm_service.py       # Llama 3.2 via LangChain + HuggingFace
│   │   └── ingestion.py         # JSON extraction + document pipeline
│   └── api/routes/
│       ├── auth.py              # /auth endpoints
│       ├── nlp.py               # /nlp endpoints
│       ├── graph.py             # /graph endpoints
│       ├── documents.py         # /documents endpoints
│       └── qa.py                # /llm endpoints
├── tests/
│   └── test_api.py              # pytest test suite
├── .github/workflows/ci.yml     # GitHub Actions CI/CD
├── docker-compose.yml           # Neo4j + Redis + PostgreSQL + App
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🤖 LLM Configuration

By default, NeoRAG uses `facebook/opt-1.3b` (no token needed, runs on CPU).

**To use Llama 3.2** (recommended for production):

1. Get access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Get your HF token: https://huggingface.co/settings/tokens
3. Update `.env`:
```env
HUGGINGFACE_TOKEN=hf_your_token_here
LLM_MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
```

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html
```

---

## 🌐 Deployment on AWS

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
docker build -t neorag .
docker tag neorag:latest <ecr-url>/neorag:latest
docker push <ecr-url>/neorag:latest

# Deploy on ECS / EC2 with docker-compose
docker-compose -f docker-compose.yml up -d
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **API Framework** | FastAPI 0.111 |
| **Graph Database** | Neo4j 5.20 + Cypher |
| **LLM** | Llama 3.2 (HuggingFace) |
| **LLM Orchestration** | LangChain 0.2 |
| **NLP / NER** | HuggingFace Transformers + spaCy |
| **Deep Learning** | PyTorch |
| **Sentiment Analysis** | RoBERTa (cardiffnlp) |
| **Authentication** | JWT (python-jose) |
| **Async Tasks** | Celery + Redis |
| **Database** | PostgreSQL + SQLAlchemy |
| **Vector DB** | Qdrant |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Cloud** | AWS (EC2, ECS, Lambda) |

---

## 👤 Author

**P Akhil** — AI/ML Engineer
- GitHub: [@AKHIL1633](https://github.com/AKHIL1633)
- LinkedIn: [linkedin.com/in/p-akhil](https://linkedin.com/in/p-akhil)
- Email: akhil.kartik371@gmail.com

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
