"""
LLM Service — Llama 3.2 (via HuggingFace) for Q&A, Summarization, Text Generation
Uses LangChain for orchestration with graph context grounding
"""

import time
from typing import List, Optional, Tuple
from loguru import logger

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline as hf_pipeline, AutoTokenizer
import torch

from app.config import get_settings
from app.services.graph_service import get_entity_context

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
            max_new_tokens=512,
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
                max_new_tokens=256,
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
    Answer a question using LLM grounded by Neo4j knowledge graph context.
    Returns: (answer, context_used, model_name)
    """
    context_facts = []

    # Extract key terms from question for graph lookup
    if use_graph:
        key_terms = _extract_key_terms(question)
        for term in key_terms[:3]:
            facts = get_entity_context(term)
            context_facts.extend(facts[:5])

    context_str = "\n".join(context_facts) if context_facts else "No graph context available."

    # Try LLM
    llm = _load_llm()
    if llm:
        try:
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful AI assistant with access to a knowledge graph.

Knowledge Graph Context:
{context}

Based on the above context and your knowledge, answer the following question concisely:

Question: {question}

Answer:"""
            )
            chain  = LLMChain(llm=llm, prompt=prompt)
            result = chain.invoke({"context": context_str, "question": question})
            answer = result.get("text", "").strip()

            # Clean up generated text
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            model_name = settings.llm_model_name
            return answer[:max_tokens], context_facts, model_name

        except Exception as e:
            logger.warning(f"LLM inference failed: {e}")

    # Rule-based fallback
    answer = _rule_based_answer(question, context_facts)
    return answer, context_facts, "rule-based-fallback"


def _extract_key_terms(question: str) -> List[str]:
    """Extract key terms from question for graph lookup."""
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
    llm = _load_llm()

    if llm:
        try:
            prompt = PromptTemplate(
                input_variables=["text", "max_length"],
                template="""Summarize the following text in approximately {max_length} words.
Be concise, accurate, and capture the key points.

Text to summarize:
{text}

Summary:"""
            )
            chain  = LLMChain(llm=llm, prompt=prompt)
            result = chain.invoke({"text": text[:2000], "max_length": max_length})
            summary = result.get("text", "").strip()

            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()

            return summary[:max_length * 6], settings.llm_model_name  # ~6 chars/word

        except Exception as e:
            logger.warning(f"Summarization failed: {e}")

    # Extractive fallback — return first N sentences
    sentences = text.split(". ")[:5]
    return ". ".join(sentences) + ".", "extractive-fallback"


# ── Text Generation ───────────────────────────────────────────────────────────

def generate_text(prompt: str, max_tokens: int = 256) -> Tuple[str, str]:
    """Generate text from a prompt using the loaded LLM."""
    llm = _load_llm()

    if llm:
        try:
            lc_prompt = PromptTemplate(input_variables=["prompt"], template="{prompt}")
            chain     = LLMChain(llm=llm, prompt=lc_prompt)
            result    = chain.invoke({"prompt": prompt})
            return result.get("text", "").strip(), settings.llm_model_name
        except Exception as e:
            logger.warning(f"Text generation failed: {e}")

    return f"Generated response for: {prompt[:50]}...", "fallback"
