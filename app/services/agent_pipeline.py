"""
Agent Pipeline — LangGraph multi-agent Q&A with a self-critique retry loop.

Nodes:
  1. understand        — classify question type (factual / analytical / comparative)
  2. extract_entities   — pull entities via NER (see nlp_service.extract_entities)
  3. retrieve_graph      — for each entity, get_entity_context() from Neo4j
  4. retrieve_vectors    — semantic search over document chunks (Qdrant)
  5. synthesize           — LLM generates an answer from the combined context
  6. critique              — LLM-as-judge scores the answer; on a low score,
                              loops back to retrieval with a widened search
                              (more entities considered, more facts per
                              entity, higher vector top_k), bounded to 2 retries

This module is imported lazily by llm_service.answer_question, inside a
try/except — if `langgraph` isn't installed, the ImportError is caught there
and callers transparently fall back to the direct (non-agentic) LLM path.
Importing this module directly always requires langgraph to be installed.
"""

from typing import List, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from app.config import get_settings
from app.core.observability import LLMCallTrace
from app.services.graph_service import get_entity_context
from app.services.nlp_service import extract_entities
from app.services.vector_service import semantic_search

settings = get_settings()

_STOPWORDS = {
    "what", "who", "where", "when", "how", "is", "are", "the", "a", "an",
    "in", "of", "for", "and", "or", "to", "that", "this", "it", "was",
}
_COMPARATIVE_MARKERS = ("compare", "versus", " vs ", "difference between", "better")


class AgentState(TypedDict):
    question: str
    question_type: str
    entities: List[str]
    graph_context: List[str]
    vector_context: List[str]
    answer: str
    model_used: str
    critique_score: float
    retry_count: int
    use_graph: bool
    max_tokens: int


def _fallback_terms(question: str) -> List[str]:
    words = [w.strip("?.,!") for w in question.split()]
    return [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]


def understand_question_node(state: AgentState) -> dict:
    q_lower = state["question"].lower()
    if any(m in q_lower for m in _COMPARATIVE_MARKERS):
        question_type = "comparative"
    elif q_lower.startswith(("why", "how")) or "analyze" in q_lower or "explain" in q_lower:
        question_type = "analytical"
    else:
        question_type = "factual"
    return {"question_type": question_type}


def extract_entities_node(state: AgentState) -> dict:
    entities = [e.text for e in extract_entities(state["question"])]
    return {"entities": entities}


def retrieve_graph_node(state: AgentState) -> dict:
    if not state.get("use_graph", True):
        return {"graph_context": []}
    terms = state["entities"] or _fallback_terms(state["question"])
    retry = state.get("retry_count", 0)
    # Widen the search on each retry rather than repeating the exact same query.
    term_limit = 3 + retry * 2
    facts_per_term = 5 + retry * 3
    context: List[str] = []
    for term in terms[:term_limit]:
        context.extend(get_entity_context(term)[:facts_per_term])
    return {"graph_context": context}


def retrieve_vectors_node(state: AgentState) -> dict:
    retry = state.get("retry_count", 0)
    top_k = 5 + retry * 3
    context = semantic_search(state["question"], top_k=top_k)
    return {"vector_context": context}


def synthesize_answer_node(state: AgentState) -> dict:
    from langchain.prompts import PromptTemplate

    from app.services.llm_service import _load_llm, _rule_based_answer

    combined_context = state["graph_context"] + state["vector_context"]
    context_str = "\n".join(combined_context) if combined_context else "No context available."

    with LLMCallTrace("agent_synthesize", settings.llm_model_name, state["question"]) as trace:
        llm = _load_llm()
        if llm:
            try:
                prompt = PromptTemplate.from_template(
                    "You are a helpful AI assistant with access to a knowledge graph.\n\n"
                    "Knowledge Graph / Document Context:\n{context}\n\n"
                    "Question type: {question_type}\n\n"
                    "Based on the above context and your knowledge, answer the following "
                    "question concisely:\n\nQuestion: {question}\n\nAnswer:"
                )
                chain = prompt | llm
                raw = chain.invoke({
                    "context": context_str,
                    "question": state["question"],
                    "question_type": state["question_type"],
                })
                answer = str(raw).strip()
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                answer = answer[: state.get("max_tokens", 512)]
                model_used = settings.llm_model_name
                if not answer:
                    # A weak (or occasionally even a strong) model can end
                    # its completion right at/before the "Answer:" marker,
                    # leaving nothing after the split — never hand the user
                    # a blank response.
                    answer = _rule_based_answer(state["question"], combined_context)
                    model_used = "rule-based-fallback"
            except Exception as e:
                logger.warning(f"Agent synthesis LLM call failed: {e}")
                answer = _rule_based_answer(state["question"], combined_context)
                model_used = "rule-based-fallback"
        else:
            answer = _rule_based_answer(state["question"], combined_context)
            model_used = "rule-based-fallback"
        trace.set(tokens_out=len(answer.split()))

    return {"answer": answer, "model_used": model_used}


def critique_node(state: AgentState) -> dict:
    from app.services.evaluation import judge_answer

    combined_context = state["graph_context"] + state["vector_context"]
    score = judge_answer(state["question"], state["answer"], combined_context)
    return {"critique_score": score}


def _increment_retry_node(state: AgentState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def should_retry(state: AgentState) -> str:
    if state["critique_score"] >= 0.6 or state.get("retry_count", 0) >= 2:
        return "end"
    return "retry"


_compiled_graph = None


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("understand", understand_question_node)
    graph.add_node("extract_entities", extract_entities_node)
    graph.add_node("retrieve_graph", retrieve_graph_node)
    graph.add_node("retrieve_vectors", retrieve_vectors_node)
    graph.add_node("synthesize", synthesize_answer_node)
    graph.add_node("critique", critique_node)
    graph.add_node("increment_retry", _increment_retry_node)

    graph.add_edge("understand", "extract_entities")
    graph.add_edge("extract_entities", "retrieve_graph")
    graph.add_edge("retrieve_graph", "retrieve_vectors")
    graph.add_edge("retrieve_vectors", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_conditional_edges(
        "critique",
        should_retry,
        {"retry": "increment_retry", "end": END},
    )
    graph.add_edge("increment_retry", "retrieve_graph")
    graph.set_entry_point("understand")
    return graph.compile()


def _get_agent():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


def run_agent(question: str, use_graph: bool = True, max_tokens: int = 512):
    """Runs the full agent graph. Returns (answer, context, model_used) — the
    same tuple shape as llm_service's legacy direct-call path, so callers
    don't need to know which path actually served the request."""
    agent = _get_agent()
    initial_state: AgentState = {
        "question": question,
        "question_type": "",
        "entities": [],
        "graph_context": [],
        "vector_context": [],
        "answer": "",
        "model_used": "",
        "critique_score": 0.0,
        "retry_count": 0,
        "use_graph": use_graph,
        "max_tokens": max_tokens,
    }
    final_state = agent.invoke(initial_state)
    return (
        final_state["answer"],
        final_state["graph_context"] + final_state["vector_context"],
        final_state["model_used"],
    )
