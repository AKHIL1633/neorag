"""
Graph Service — Neo4j knowledge graph operations
Covers: node creation, relationship building, Cypher queries, graph stats
"""

import re
import time
import uuid
from typing import List, Dict, Any, Optional
from loguru import logger
from app.core.database import Neo4jSession
from app.models.schemas import (
    GraphNode, GraphRelationship, GraphStatsResponse, GraphQueryResponse
)

# ── In-memory fallback (when Neo4j is unavailable) ───────────────────────────
_memory_nodes: Dict[str, Dict] = {}
_memory_edges: List[Dict] = []

# Neo4j does not support parameterizing labels/relationship types, so they
# must be validated in Python before being interpolated into Cypher. A
# well-formed identifier alone isn't enough defense-in-depth — restrict to a
# fixed allowlist matching the entity/relation vocabulary this service
# actually produces (see EntityType in schemas.py and infer_relationship
# below).
_VALID_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "DATE", "MISC", "Document"}
_ALLOWED_RELATIONS = {
    "WORKS_FOR", "LOCATED_IN", "FROM", "ASSOCIATED_WITH", "RELATED_TO",
    "MADE_BY", "OCCURRED_IN", "CO_OCCURS_WITH", "MENTIONS",
}


def _validate_label(label: str) -> str:
    if not _VALID_IDENT.match(label):
        raise ValueError(f"Invalid label: {label!r}")
    if label not in _ALLOWED_LABELS:
        raise ValueError(f"Label not in allowlist: {label!r}")
    return label


def _validate_relation(rel: str) -> str:
    normalized = rel.upper().replace(" ", "_")
    if not _VALID_IDENT.match(normalized):
        raise ValueError(f"Invalid relation: {rel!r}")
    if normalized not in _ALLOWED_RELATIONS:
        raise ValueError(f"Relation not in allowlist: {rel!r}")
    return normalized


# ── Node Operations ───────────────────────────────────────────────────────────

def create_entity_node(name: str, label: str, properties: Optional[Dict] = None) -> str:
    """Create or merge an entity node in Neo4j."""
    properties = properties or {}
    label = _validate_label(label)
    node_id = str(uuid.uuid4())[:8]

    cypher = f"""
    MERGE (n:{label} {{name: $name}})
    ON CREATE SET n.id = $node_id, n.created_at = timestamp()
    ON MATCH  SET n.updated_at = timestamp()
    SET n += $props
    RETURN n.id AS id
    """

    with Neo4jSession() as session:
        if session.session is not None:
            results = session.run_write(cypher, name=name, node_id=node_id, props=properties)
            if results:
                return results[0].get("id", node_id)

    # In-memory fallback
    _memory_nodes[name] = {"id": node_id, "label": label, "name": name, **properties}
    return node_id


def create_relationship(source: str, target: str, relation: str, weight: float = 1.0):
    """Create a relationship between two entity nodes."""
    relation = _validate_relation(relation)

    cypher = f"""
    MATCH (a {{name: $source}}), (b {{name: $target}})
    MERGE (a)-[r:{relation}]->(b)
    ON CREATE SET r.weight = $weight, r.created_at = timestamp()
    ON MATCH  SET r.weight = r.weight + $weight
    RETURN type(r) AS rel_type
    """

    with Neo4jSession() as session:
        if session.session is not None:
            session.run_write(cypher, source=source, target=target, weight=weight)
            return

    # In-memory fallback — only reached when Neo4j is unavailable
    _memory_edges.append({"source": source, "target": target, "relation": relation})


# ── Knowledge Graph Builder ───────────────────────────────────────────────────

def build_knowledge_graph(entities: List[Dict], document_title: str) -> Dict[str, int]:
    """
    Build a knowledge graph from extracted NLP entities.
    Creates nodes for each entity and relationships between co-occurring entities.
    """
    nodes_created = 0
    relationships = 0

    # Create document node
    create_entity_node(document_title, "Document", {"type": "source_document"})
    nodes_created += 1

    # Create entity nodes
    entity_names = []
    for entity in entities:
        name = entity.get("text", "").strip()
        label = entity.get("label", "MISC")
        if label not in _ALLOWED_LABELS:
            label = "MISC"
        if name and len(name) > 1:
            create_entity_node(name, label, {"confidence": entity.get("confidence", 0.8)})
            entity_names.append((name, label))

            # Link entity to document
            create_relationship(document_title, name, "MENTIONS")
            nodes_created += 1
            relationships += 1

    # Create co-occurrence relationships between entities
    for i in range(len(entity_names)):
        for j in range(i + 1, min(i + 4, len(entity_names))):  # sliding window
            src_name, src_label = entity_names[i]
            tgt_name, tgt_label = entity_names[j]

            # Determine relationship type based on entity types
            rel = infer_relationship(src_label, tgt_label)
            create_relationship(src_name, tgt_name, rel, weight=0.5)
            relationships += 1

    return {"nodes_created": nodes_created, "relationships": relationships}


def infer_relationship(src_label: str, tgt_label: str) -> str:
    """Infer relationship type from entity label combination."""
    mapping = {
        ("PERSON", "ORG"):     "WORKS_FOR",
        ("ORG",    "GPE"):     "LOCATED_IN",
        ("PERSON", "GPE"):     "FROM",
        ("PERSON", "PERSON"):  "ASSOCIATED_WITH",
        ("ORG",    "ORG"):     "RELATED_TO",
        ("PRODUCT","ORG"):     "MADE_BY",
        ("EVENT",  "GPE"):     "OCCURRED_IN",
    }
    return mapping.get((src_label, tgt_label),
           mapping.get((tgt_label, src_label), "CO_OCCURS_WITH"))


# ── Query Operations ──────────────────────────────────────────────────────────

def run_cypher_query(query: str, params: Optional[Dict] = None) -> GraphQueryResponse:
    """Run arbitrary Cypher query against Neo4j."""
    params = params or {}
    with Neo4jSession() as session:
        if session.session is not None:
            results = session.run(query, **params)
            return GraphQueryResponse(results=results, count=len(results), query=query)
    # Fallback — Neo4j unavailable, return an empty result set rather than crashing
    return GraphQueryResponse(results=[], count=0, query=query)


def get_entity_context(entity_name: str, depth: int = 2) -> List[str]:
    """Get graph context for an entity — used for LLM Q&A grounding."""
    cypher = """
    MATCH path = (n {name: $name})-[r*1..2]-(m)
    UNWIND relationships(path) AS rel
    RETURN n.name AS source,
           type(rel) AS relation,
           endNode(rel).name AS target,
           labels(endNode(rel))[0] AS target_type
    LIMIT 20
    """
    with Neo4jSession() as session:
        if session.session is not None:
            results = session.run(cypher, name=entity_name)
            context = [
                f"{r['source']} --[{r['relation']}]--> {r['target']} ({r['target_type']})"
                for r in results
                if all(r.get(k) for k in ["source", "relation", "target"])
            ]
            return context

    # Genuine in-memory fallback — reached when Neo4j is unavailable
    context = []
    for edge in _memory_edges:
        if edge["source"] == entity_name or edge["target"] == entity_name:
            context.append(f"{edge['source']} --[{edge['relation']}]--> {edge['target']}")
    return context


def get_graph_stats() -> GraphStatsResponse:
    """Get statistics about the knowledge graph."""
    with Neo4jSession() as session:
        if session.session is not None:
            node_result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt")
            rel_result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS cnt")

            node_types = {r["label"]: r["cnt"] for r in node_result if r.get("label")}
            rel_types = {r["type"]: r["cnt"] for r in rel_result if r.get("type")}

            return GraphStatsResponse(
                total_nodes=sum(node_types.values()),
                total_relationships=sum(rel_types.values()),
                node_types=node_types,
                relationship_types=rel_types,
            )

    # Fallback — reached when Neo4j is unavailable
    node_types: Dict[str, int] = {}
    for node in _memory_nodes.values():
        node_types[node["label"]] = node_types.get(node["label"], 0) + 1
    rel_types: Dict[str, int] = {}
    for edge in _memory_edges:
        rel_types[edge["relation"]] = rel_types.get(edge["relation"], 0) + 1

    return GraphStatsResponse(
        total_nodes=len(_memory_nodes),
        total_relationships=len(_memory_edges),
        node_types=node_types,
        relationship_types=rel_types,
    )
