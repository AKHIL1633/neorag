from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import (
    GraphQueryRequest, GraphQueryResponse, GraphStatsResponse
)
from app.services.graph_service import (
    run_cypher_query, get_graph_stats, get_entity_context,
    create_entity_node, create_relationship
)
from app.core.auth import get_current_user

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


@router.get(
    "/stats",
    response_model=GraphStatsResponse,
    summary="Get knowledge graph statistics"
)
async def graph_stats(current_user: dict = Depends(get_current_user)):
    """
    Get statistics about the Neo4j knowledge graph:
    - Total nodes and their types (PERSON, ORG, GPE, etc.)
    - Total relationships and their types (WORKS_FOR, LOCATED_IN, etc.)
    """
    return get_graph_stats()


@router.post(
    "/query",
    response_model=GraphQueryResponse,
    summary="Run Cypher query against Neo4j"
)
async def cypher_query(
    request: GraphQueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Execute a **Cypher query** directly against the Neo4j knowledge graph.

    Example queries:
    ```cypher
    MATCH (n:PERSON) RETURN n.name LIMIT 10
    MATCH (p:PERSON)-[:WORKS_FOR]->(o:ORG) RETURN p.name, o.name
    MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 20
    ```
    """
    # Basic security — prevent destructive queries
    dangerous = ["DELETE", "DETACH", "DROP", "REMOVE"]
    if any(kw in request.cypher.upper() for kw in dangerous):
        raise HTTPException(
            status_code=403,
            detail="Destructive Cypher operations are not allowed via API"
        )
    return run_cypher_query(request.cypher, request.params or {})


@router.get(
    "/entity/{entity_name}/context",
    summary="Get graph context for an entity"
)
async def entity_context(
    entity_name: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get all graph relationships for a named entity.
    Returns connected nodes up to 2 hops away.

    Example: `/graph/entity/Apple/context` returns all
    companies, people, and locations related to Apple.
    """
    context = get_entity_context(entity_name)
    return {
        "entity":        entity_name,
        "context_facts": context,
        "fact_count":    len(context),
    }


@router.post(
    "/node",
    summary="Manually create an entity node"
)
async def create_node(
    name:  str,
    label: str = "MISC",
    current_user: dict = Depends(get_current_user)
):
    """Manually create or merge an entity node in the knowledge graph."""
    node_id = create_entity_node(name, label)
    return {"node_id": node_id, "name": name, "label": label, "status": "created"}


@router.post(
    "/relationship",
    summary="Create a relationship between two entities"
)
async def add_relationship(
    source:   str,
    target:   str,
    relation: str,
    current_user: dict = Depends(get_current_user)
):
    """Create a directed relationship between two entity nodes."""
    create_relationship(source, target, relation)
    return {
        "source":   source,
        "relation": relation,
        "target":   target,
        "status":   "created",
    }
