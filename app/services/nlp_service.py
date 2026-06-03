"""
NLP Service — Stanford NLP-equivalent capabilities via HuggingFace + spaCy
Covers: Named Entity Recognition, Sentiment Analysis, Coreference Resolution hints
"""

import time
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Any, Optional
from loguru import logger
from app.models.schemas import Entity, SentimentResult, NLPAnalysisResponse
from app.config import get_settings

settings = get_settings()

# ── Lazy-loaded models ────────────────────────────────────────────────────────
_nlp_spacy       = None
_ner_pipeline    = None
_sentiment_pipe  = None


def _get_spacy():
    global _nlp_spacy
    if _nlp_spacy is None:
        try:
            _nlp_spacy = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy en_core_web_sm loaded")
        except Exception as e:
            logger.warning(f"spaCy not available: {e}")
    return _nlp_spacy


def _get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            _ner_pipeline = pipeline(
                "ner",
                model=settings.ner_model,
                aggregation_strategy="simple",
                device=-1,   # CPU; set to 0 for GPU
            )
            logger.info(f"✅ NER model loaded: {settings.ner_model}")
        except Exception as e:
            logger.warning(f"HuggingFace NER not available: {e}. Using spaCy fallback.")
    return _ner_pipeline


def _get_sentiment_pipeline():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        try:
            _sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=settings.sentiment_model,
                device=-1,
            )
            logger.info(f"✅ Sentiment model loaded: {settings.sentiment_model}")
        except Exception as e:
            logger.warning(f"Sentiment model not available: {e}")
    return _sentiment_pipe


# ── NER ───────────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> List[Entity]:
    """
    Extract named entities — Stanford NLP NER equivalent.
    Detects: PERSON, ORG, GPE, LOC, DATE, PRODUCT, EVENT
    """
    entities = []

    # Try HuggingFace NER first (more accurate)
    ner = _get_ner_pipeline()
    if ner:
        try:
            results = ner(text[:512])  # truncate for performance
            for r in results:
                entities.append(Entity(
                    text=r["word"],
                    label=r["entity_group"],
                    start=r.get("start", 0),
                    end=r.get("end", 0),
                    confidence=round(r["score"], 4),
                ))
            return entities
        except Exception as e:
            logger.warning(f"HuggingFace NER failed: {e}. Falling back to spaCy.")

    # spaCy fallback
    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.85,
            ))

    return entities


# ── Sentiment Analysis ────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> SentimentResult:
    """
    Sentiment analysis — positive / negative / neutral.
    Uses RoBERTa-based model for high accuracy.
    """
    pipe = _get_sentiment_pipeline()
    if pipe:
        try:
            result = pipe(text[:512])[0]
            label  = result["label"].lower()
            # Normalize labels
            if "pos" in label:
                label = "positive"
            elif "neg" in label:
                label = "negative"
            else:
                label = "neutral"
            return SentimentResult(label=label, score=round(result["score"], 4), text=text[:100])
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")

    # Simple rule-based fallback
    positive_words = {"good", "great", "excellent", "amazing", "positive", "success"}
    negative_words = {"bad", "poor", "terrible", "negative", "fail", "loss"}
    words = set(text.lower().split())
    if words & positive_words:
        return SentimentResult(label="positive", score=0.7, text=text[:100])
    elif words & negative_words:
        return SentimentResult(label="negative", score=0.7, text=text[:100])
    return SentimentResult(label="neutral", score=0.6, text=text[:100])


# ── Coreference Resolution (simplified) ──────────────────────────────────────

def extract_coreference_hints(text: str, entities: List[Entity]) -> List[str]:
    """
    Simplified coreference resolution — identifies pronouns and maps them
    to the nearest named entity (Stanford NLP CoreNLP-style approach).
    """
    pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their"}
    words    = text.split()
    hints    = []

    entity_names = [e.text for e in entities if e.label in ("PERSON", "ORG")]

    for i, word in enumerate(words):
        if word.lower() in pronouns and entity_names:
            # Find nearest preceding entity
            context = " ".join(words[max(0, i-10):i])
            for name in entity_names:
                if name in context:
                    hints.append(f'"{word}" likely refers to "{name}"')
                    break

    return hints[:5]  # Return top 5 hints


# ── Full Analysis ─────────────────────────────────────────────────────────────

def analyze_text(text: str) -> NLPAnalysisResponse:
    """Run full NLP pipeline: NER + Sentiment + Coreference."""
    start = time.time()

    entities    = extract_entities(text)
    sentiment   = analyze_sentiment(text)
    coref_hints = extract_coreference_hints(text, entities)

    return NLPAnalysisResponse(
        entities=entities,
        sentiment=sentiment,
        coreference_hints=coref_hints,
        word_count=len(text.split()),
        processing_time_ms=round((time.time() - start) * 1000, 2),
    )
