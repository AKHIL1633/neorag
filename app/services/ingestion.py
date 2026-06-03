"""
Document Ingestion Service
Handles JSON extraction, text preprocessing, and feeding into NLP + Graph pipeline
"""

import time
import json
import uuid
from typing import Dict, Any, List, Optional
from loguru import logger

from app.services.nlp_service import analyze_text
from app.services.graph_service import build_knowledge_graph


# ── JSON Extraction ───────────────────────────────────────────────────────────

def extract_text_from_json(data: Dict[str, Any]) -> str:
    """
    Recursively extract all string values from semi-structured JSON.
    Handles nested objects, arrays, and mixed types.
    """
    texts = []
    _extract_recursive(data, texts)
    return " ".join(texts)


def _extract_recursive(obj: Any, texts: List[str], depth: int = 0):
    """Recursively walk JSON tree extracting string content."""
    if depth > 10:  # prevent infinite recursion
        return

    if isinstance(obj, str) and len(obj.strip()) > 2:
        texts.append(obj.strip())
    elif isinstance(obj, dict):
        for key, value in obj.items():
            # Add key as context
            if isinstance(key, str):
                texts.append(key.replace("_", " ").replace("-", " "))
            _extract_recursive(value, texts, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _extract_recursive(item, texts, depth + 1)
    elif isinstance(obj, (int, float, bool)):
        texts.append(str(obj))


def extract_entities_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured entity information from JSON data.
    Maps JSON fields to entity types intelligently.
    """
    entities = {}

    # Common patterns in JSON data
    person_keys   = {"name", "author", "user", "person", "ceo", "founder", "employee"}
    org_keys      = {"company", "organization", "org", "employer", "brand", "client"}
    location_keys = {"city", "country", "location", "address", "region", "state"}
    date_keys     = {"date", "created_at", "updated_at", "timestamp", "year", "month"}

    def _find_entities(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = key.lower()
                full_path = f"{path}.{key}" if path else key

                if isinstance(value, str) and value.strip():
                    if key_lower in person_keys:
                        entities[full_path] = {"text": value, "label": "PERSON"}
                    elif key_lower in org_keys:
                        entities[full_path] = {"text": value, "label": "ORG"}
                    elif key_lower in location_keys:
                        entities[full_path] = {"text": value, "label": "GPE"}
                    elif key_lower in date_keys:
                        entities[full_path] = {"text": value, "label": "DATE"}

                _find_entities(value, full_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _find_entities(item, f"{path}[{i}]")

    _find_entities(data)
    return entities


# ── Document Processing Pipeline ─────────────────────────────────────────────

def process_document(title: str, content: str, metadata: Dict = {}) -> Dict[str, Any]:
    """
    End-to-end document processing pipeline:
    1. NLP analysis (NER + sentiment + coreference)
    2. Knowledge graph construction
    3. Return processing results
    """
    start    = time.time()
    doc_id   = str(uuid.uuid4())[:12]

    logger.info(f"Processing document: '{title}' ({len(content)} chars)")

    # Step 1: NLP analysis
    nlp_result = analyze_text(content)

    # Step 2: Build knowledge graph from entities
    entities_dict = [e.model_dump() for e in nlp_result.entities]
    graph_result  = build_knowledge_graph(entities_dict, title)

    elapsed = round((time.time() - start) * 1000, 2)
    logger.info(f"Document '{title}' processed in {elapsed}ms — "
                f"{graph_result['nodes_created']} nodes, "
                f"{graph_result['relationships']} relationships")

    return {
        "id":                  doc_id,
        "title":               title,
        "entity_count":        len(nlp_result.entities),
        "relationship_count":  graph_result["relationships"],
        "sentiment":           nlp_result.sentiment.label,
        "processing_time_ms":  elapsed,
    }


def process_json_document(json_data: Dict[str, Any], extract_entities: bool = True,
                           build_graph: bool = True) -> Dict[str, Any]:
    """
    Process a JSON document through the full pipeline.
    Extracts text, runs NLP, builds knowledge graph.
    """
    start = time.time()

    # Extract text from JSON
    extracted_text = extract_text_from_json(json_data)

    # Extract structured entities from JSON keys/values
    json_entities = extract_entities_from_json(json_data)

    nodes_created = 0
    relationships = 0
    entities_found = len(json_entities)

    if build_graph and json_entities:
        entity_list = [
            {"text": v["text"], "label": v["label"], "confidence": 0.95}
            for v in json_entities.values()
        ]
        title = json_data.get("title", json_data.get("name", "JSON Document"))
        graph_result  = build_knowledge_graph(entity_list, str(title))
        nodes_created = graph_result["nodes_created"]
        relationships = graph_result["relationships"]

    elapsed = round((time.time() - start) * 1000, 2)

    return {
        "extracted_text":      extracted_text[:500],
        "entities_found":      entities_found,
        "nodes_created":       nodes_created,
        "relationships":       relationships,
        "processing_time_ms":  elapsed,
    }
