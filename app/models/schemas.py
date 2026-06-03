from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ── Auth ──────────────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Document Ingestion ────────────────────────────────────────────────────────

class DocumentIngest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Raw text content")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Optional metadata")

class DocumentResponse(BaseModel):
    id: str
    title: str
    entity_count: int
    relationship_count: int
    sentiment: str
    processing_time_ms: float
    message: str = "Document processed and knowledge graph updated"


# ── NLP ───────────────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    PERSON  = "PERSON"
    ORG     = "ORG"
    GPE     = "GPE"       # Geo-political entity
    LOC     = "LOC"
    PRODUCT = "PRODUCT"
    EVENT   = "EVENT"
    DATE    = "DATE"
    MISC    = "MISC"

class Entity(BaseModel):
    text:        str
    label:       str
    start:       int
    end:         int
    confidence:  float

class SentimentResult(BaseModel):
    label:   str          # positive / negative / neutral
    score:   float
    text:    str

class NLPAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyse")

class NLPAnalysisResponse(BaseModel):
    entities:          List[Entity]
    sentiment:         SentimentResult
    coreference_hints: List[str]   # simplified coreference
    word_count:        int
    processing_time_ms: float


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id:         str
    label:      str          # Entity type
    name:       str
    properties: Dict[str, Any] = {}

class GraphRelationship(BaseModel):
    source:      str
    target:      str
    relation:    str
    weight:      float = 1.0

class GraphQueryRequest(BaseModel):
    cypher: str = Field(..., description="Cypher query to run against Neo4j")
    params: Optional[Dict[str, Any]] = {}

class GraphQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    count:   int
    query:   str

class GraphStatsResponse(BaseModel):
    total_nodes:         int
    total_relationships: int
    node_types:          Dict[str, int]
    relationship_types:  Dict[str, int]


# ── LLM Q&A ───────────────────────────────────────────────────────────────────

class QARequest(BaseModel):
    question:    str = Field(..., description="Question to ask over the knowledge graph")
    use_graph:   bool = Field(default=True, description="Whether to use Neo4j graph context")
    max_tokens:  int  = Field(default=512, ge=64, le=2048)

class QAResponse(BaseModel):
    question:    str
    answer:      str
    context:     List[str]     # Graph nodes/facts used
    model_used:  str
    processing_time_ms: float

class SummarizeRequest(BaseModel):
    document_id: Optional[str] = None
    text:        Optional[str] = None
    max_length:  int = Field(default=200, ge=50, le=1000)

class SummarizeResponse(BaseModel):
    summary:            str
    original_length:    int
    summary_length:     int
    processing_time_ms: float


# ── JSON Extraction ───────────────────────────────────────────────────────────

class JSONExtractionRequest(BaseModel):
    json_data:   Dict[str, Any] = Field(..., description="Semi-structured JSON to process")
    extract_entities: bool = True
    build_graph:      bool = True

class JSONExtractionResponse(BaseModel):
    extracted_text:  str
    entities_found:  int
    nodes_created:   int
    relationships:   int
    processing_time_ms: float
