from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.config import get_settings
from app.core.database import close_driver
from app.api.routes import auth, nlp, graph, documents, qa

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info("📡 Connecting to Neo4j...")
    yield
    logger.info("🛑 Shutting down — closing Neo4j connection")
    close_driver()


app = FastAPI(
    title="NeoRAG — Knowledge Graph + LLM Q&A API",
    description="""
## NeoRAG: Knowledge Graph-Powered LLM API

A production-grade AI backend built with **FastAPI**, **Neo4j**, **Llama 3.2**, and **LangChain**.

### Features
- 🧠 **LLM Q&A** — Graph-grounded question answering using Llama 3.2
- 🔍 **NLP Analysis** — Named Entity Recognition, Sentiment Analysis, Coreference Resolution
- 🕸️ **Knowledge Graph** — Neo4j graph construction from text/JSON with Cypher queries
- 📄 **Document Ingestion** — Full pipeline: text/JSON → NLP → knowledge graph
- 🔐 **JWT Authentication** — Secure token-based access

### Quick Start
1. Get a token: `POST /api/v1/auth/token` (admin/admin123)
2. Ingest a document: `POST /api/v1/documents/ingest`
3. Ask a question: `POST /api/v1/llm/ask`
4. Query the graph: `POST /api/v1/graph/query`

### Tech Stack
`FastAPI` · `Neo4j` · `Llama 3.2 (HuggingFace)` · `LangChain` · `spaCy` · `PyTorch` · `Docker` · `AWS`
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
PREFIX = "/api/v1"
app.include_router(auth.router,      prefix=PREFIX)
app.include_router(nlp.router,       prefix=PREFIX)
app.include_router(graph.router,     prefix=PREFIX)
app.include_router(documents.router, prefix=PREFIX)
app.include_router(qa.router,        prefix=PREFIX)


@app.get("/", tags=["Health"])
async def root():
    return {
        "name":    settings.app_name,
        "version": settings.app_version,
        "status":  "running",
        "docs":    "/docs",
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "app": settings.app_name}
