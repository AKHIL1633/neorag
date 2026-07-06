import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.config import get_settings
from app.core.database import close_driver
from app.core.rate_limit import limiter
from app.core.observability import request_id_ctx
from app.api.routes import auth, nlp, graph, documents, qa, eval as eval_routes

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

A knowledge graph-powered AI backend built with **FastAPI**, **Neo4j**, **LangGraph**, and **LangChain**.
Demo project — see the README's "Honest Metrics" section for what's actually been measured.

### Features
- 🧠 **Agentic Q&A** — LangGraph pipeline: question understanding, entity extraction,
  graph + vector retrieval, answer synthesis, and a self-critique retry loop
- 🔍 **NLP Analysis** — Named Entity Recognition, Sentiment Analysis, Coreference Resolution
- 🕸️ **Knowledge Graph** — Neo4j graph construction from text/JSON with Cypher queries
- 🔎 **Semantic Retrieval** — Qdrant + sentence-transformers over ingested document chunks
- 📄 **Document Ingestion** — Full pipeline: text/JSON → NLP → knowledge graph → vector index
- 🔐 **JWT Authentication** — Real Postgres-backed user accounts (register + login)
- 🛡️ **Guardrails** — prompt-injection input screening + PII output redaction
- 📊 **Evaluation** — LLM-as-judge benchmark suite (`POST /eval/run`)

### Quick Start
1. Register: `POST /api/v1/auth/register`
2. Get a token: `POST /api/v1/auth/token`
3. Ingest a document: `POST /api/v1/documents/ingest`
4. Ask a question: `POST /api/v1/llm/ask`
5. Query the graph: `POST /api/v1/graph/query`

### Tech Stack
`FastAPI` · `Neo4j` · `LangGraph` · `LangChain` · `Qdrant` · `spaCy` · `PyTorch` · `PostgreSQL` · `Docker`
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — no wildcard when credentials are enabled (browsers reject that combo)
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    token = request_id_ctx.set(uuid.uuid4().hex[:16])
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    response.headers["X-Request-ID"] = request_id_ctx.get()
    return response


# Routers
PREFIX = "/api/v1"
app.include_router(auth.router,        prefix=PREFIX)
app.include_router(nlp.router,         prefix=PREFIX)
app.include_router(graph.router,       prefix=PREFIX)
app.include_router(documents.router,   prefix=PREFIX)
app.include_router(qa.router,          prefix=PREFIX)
app.include_router(eval_routes.router, prefix=PREFIX)


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
    from app.core.database import get_driver
    from app.services.nlp_service import _ner_pipeline, _sentiment_pipe

    neo4j_ok = False
    try:
        driver = get_driver()
        if driver:
            driver.verify_connectivity()
            neo4j_ok = True
    except Exception:
        neo4j_ok = False

    return {
        "status": "healthy" if neo4j_ok else "degraded",
        "app": settings.app_name,
        "dependencies": {
            "neo4j": "ok" if neo4j_ok else "unavailable (using in-memory fallback)",
            # NLP models are lazy-loaded — checking availability here would
            # trigger a load, so we only report whether they've been loaded
            # by a prior request, not whether they *can* load.
            "ner_model_loaded": _ner_pipeline is not None,
            "sentiment_model_loaded": _sentiment_pipe is not None,
        }
    }
