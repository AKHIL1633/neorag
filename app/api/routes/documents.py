from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import (
    DocumentIngest, DocumentResponse,
    JSONExtractionRequest, JSONExtractionResponse
)
from app.services.ingestion import process_document, process_json_document
from app.core.auth import get_current_user

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post(
    "/ingest",
    response_model=DocumentResponse,
    summary="Ingest a text document → NLP → Knowledge Graph"
)
async def ingest_document(
    request: DocumentIngest,
    current_user: dict = Depends(get_current_user)
):
    """
    **Full document ingestion pipeline:**

    1. Run NLP analysis (NER, sentiment, coreference)
    2. Extract named entities (PERSON, ORG, GPE, etc.)
    3. Build Neo4j knowledge graph nodes + relationships
    4. Return processing summary

    The document's entities and relationships become queryable
    via the `/graph/query` endpoint using Cypher.
    """
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Document content cannot be empty")
    if len(request.content) > 50_000:
        raise HTTPException(status_code=413, detail="Document too large (max 50,000 chars)")

    result = process_document(request.title, request.content, request.metadata or {})
    return DocumentResponse(**result)


@router.post(
    "/ingest-json",
    response_model=JSONExtractionResponse,
    summary="Extract entities from semi-structured JSON → Knowledge Graph"
)
async def ingest_json(
    request: JSONExtractionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    **JSON data extraction and knowledge graph building:**

    1. Recursively extract all text values from JSON
    2. Map JSON keys to entity types (name→PERSON, company→ORG, etc.)
    3. Build Neo4j knowledge graph from extracted entities
    4. Return extraction summary

    Handles nested objects, arrays, and mixed-type JSON.

    **Example JSON input:**
    ```json
    {
      "name": "Tim Cook",
      "company": "Apple Inc",
      "city": "Cupertino",
      "products": ["iPhone", "MacBook"]
    }
    ```
    """
    result = process_json_document(
        request.json_data,
        extract_entities=request.extract_entities,
        build_graph=request.build_graph,
    )
    return JSONExtractionResponse(**result)
