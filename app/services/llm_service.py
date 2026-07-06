"""
LLM Service — Llama 3.2 (via HuggingFace) for Q&A, Summarization, Text Generation

`answer_question` prefers the LangGraph multi-agent pipeline
(app.services.agent_pipeline) — question understanding, entity extraction,
graph + vector retrieval, synthesis, and a self-critique retry loop. If
`langgraph` isn't installed, or the agent raises, it falls back to the
direct single-call path below (now using LCEL instead of the deprecated
LLMChain) — the graceful-degradation path required so this module never
crashes on import in an environment without the optional agent dependencies.
"""

from typing import List, Optional, Tuple
from loguru import logger

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline as hf_pipeline
import torch

from app.config import get_settings
from app.core.guardrails import check_input, redact_output
from app.core.observability import LLMCallTrace
from app.services.graph_service import get_entity_context
from app.services.nlp_service import extract_entities

settings = get_settings()

_llm_pipeline = None
_langchain_llm = None


# ── Model Loading ─────────────────────────────────────────────────────────────

def _load_llm():
    global _llm_pipeline, _langchain_llm
    if _langchain_llm:
        return _langchain_llm

    try:
        logger.info(f"Loading LLM: {settings.llm_model_name}")
        pipe = hf_pipeline(
            "text-generation",
            model=settings.llm_model_name,
            token=settings.huggingface_token or None,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            max_new_tokens=settings.llm_max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=2,
        )
        _langchain_llm = HuggingFacePipeline(pipeline=pipe)
        logger.info(f"✅ LLM loaded: {settings.llm_model_name}")
    except Exception as e:
        logger.warning(f"Primary LLM failed ({e}), trying fallback: {settings.llm_fallback_model}")
        try:
            pipe = hf_pipeline(
                "text-generation",
                model=settings.llm_fallback_model,
                max_new_tokens=min(settings.llm_max_new_tokens, 256),
                do_sample=True,
                temperature=0.7,
                pad_token_id=1,
            )
            _langchain_llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"✅ Fallback LLM loaded: {settings.llm_fallback_model}")
        except Exception as e2:
            logger.error(f"Both LLMs failed: {e2}. Using rule-based fallback.")

    return _langchain_llm


# ── Q&A over Knowledge Graph ──────────────────────────────────────────────────

def answer_question(question: str, use_graph: bool = True, max_tokens: int = 512) -> Tuple[str, List[str], str]:
    """
    Answer a question, grounded by Neo4j graph context and (if available)
    semantic vector retrieval. Returns: (answer, context_used, model_name)
    """
    ok, reason = check_input(question)
    if not ok:
        raise ValueError(reason)

    with LLMCallTrace("answer_question", settings.llm_model_name, question) as trace:
        try:
            from app.services.agent_pipeline import run_agent

            answer, context, model_name = run_agent(question, use_graph=use_graph, max_tokens=max_tokens)
            trace.set(path="agent_pipeline", tokens_out=len(answer.split()))
            return redact_output(answer), context, model_name
        except ImportError:
            logger.info("LangGraph not installed — using direct LLM call instead of the agent pipeline")
        except Exception as e:
            logger.warning(f"Agent pipeline failed ({e}) — falling back to direct LLM call")

        # ── Direct (non-agentic) fallback path ───────────────────────────────
        context_facts = []
        if use_graph:
            key_terms = _extract_key_terms(question)
            for term in key_terms[:3]:
                facts = get_entity_context(term)
                context_facts.extend(facts[:5])

        context_str = "\n".join(context_facts) if context_facts else "No graph context available."

        llm = _load_llm()
        if llm:
            try:
                prompt = PromptTemplate.from_template(
                    "You are a helpful AI assistant with access to a knowledge graph.\n\n"
                    "Knowledge Graph Context:\n{context}\n\n"
                    "Based on the above context and your knowledge, answer the following "
                    "question concisely:\n\nQuestion: {question}\n\nAnswer:"
                )
                chain = prompt | llm
                raw = chain.invoke({"context": context_str, "question": question})
                answer = str(raw).strip()

                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()

                if not answer:
                    # The model can end its completion right at/before the
                    # "Answer:" marker, leaving nothing after the split —
                    # never hand the user a blank response.
                    raise ValueError("LLM produced an empty completion")

                model_name = settings.llm_model_name
                trace.set(path="direct_llm", tokens_out=len(answer.split()))
                return redact_output(answer[:max_tokens]), context_facts, model_name
            except Exception as e:
                logger.warning(f"LLM inference failed: {e}")

        # Rule-based fallback
        answer = _rule_based_answer(question, context_facts)
        trace.set(path="rule_based")
        return redact_output(answer), context_facts, "rule-based-fallback"


def _extract_key_terms(question: str) -> List[str]:
    """Extract key terms from question for graph lookup.

    Runs NER first so multi-word entities ("Steve Jobs") are kept intact
    instead of being split into single tokens that won't exact-match the
    corresponding graph node's name.
    """
    entities = extract_entities(question)
    entity_texts = [e.text for e in entities]
    if entity_texts:
        return entity_texts

    # Fallback — noun-like tokens, only reached when no NER backend is available
    stopwords = {"what", "who", "where", "when", "how", "is", "are", "the", "a", "an",
                 "in", "of", "for", "and", "or", "to", "that", "this", "it", "was"}
    words = [w.strip("?.,!") for w in question.split()]
    return [w for w in words if w.lower() not in stopwords and len(w) > 2]


def _rule_based_answer(question: str, context: List[str]) -> str:
    if context:
        return f"Based on the knowledge graph: {context[0]}. " \
               f"Found {len(context)} related facts in the graph."
    return "I couldn't find relevant information in the knowledge graph for this question."


# ── Summarization ─────────────────────────────────────────────────────────────

def summarize_text(text: str, max_length: int = 200) -> Tuple[str, str]:
    """
    Summarize text using Llama 3.2 / HuggingFace.
    Returns: (summary, model_used)
    """
    ok, reason = check_input(text)
    if not ok:
        raise ValueError(reason)

    llm = _load_llm()

    if llm:
        try:
            prompt = PromptTemplate.from_template(
                "Summarize the following text in approximately {max_length} words.\n"
                "Be concise, accurate, and capture the key points.\n\n"
                "Text to summarize:\n{text}\n\nSummary:"
            )
            chain = prompt | llm
            raw = chain.invoke({"text": text[:2000], "max_length": max_length})
            summary = str(raw).strip()

            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()

            return redact_output(summary[:max_length * 6]), settings.llm_model_name  # ~6 chars/word

        except Exception as e:
            logger.warning(f"Summarization failed: {e}")

    # Extractive fallback — return first N sentences
    sentences = text.split(". ")[:5]
    return redact_output(". ".join(sentences) + "."), "extractive-fallback"


# ── Text Generation ───────────────────────────────────────────────────────────

def generate_text(prompt: str, max_tokens: int = 256) -> Tuple[str, str]:
    """Generate text from a prompt using the loaded LLM."""
    ok, reason = check_input(prompt)
    if not ok:
        raise ValueError(reason)

    llm = _load_llm()

    if llm:
        try:
            lc_prompt = PromptTemplate.from_template("{prompt}")
            chain = lc_prompt | llm
            raw = chain.invoke({"prompt": prompt})
            return redact_output(str(raw).strip()), settings.llm_model_name
        except Exception as e:
            logger.warning(f"Text generation failed: {e}")

    return redact_output(f"Generated response for: {prompt[:50]}..."), "fallback"
