from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import NLPAnalysisRequest, NLPAnalysisResponse
from app.services.nlp_service import analyze_text
from app.core.auth import get_current_user

router = APIRouter(prefix="/nlp", tags=["NLP Analysis"])


@router.post(
    "/analyze",
    response_model=NLPAnalysisResponse,
    summary="Full NLP analysis — NER + Sentiment + Coreference"
)
async def analyze(
    request: NLPAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Run complete NLP pipeline on input text:

    - **Named Entity Recognition (NER)** — detects PERSON, ORG, GPE, LOC, DATE
    - **Sentiment Analysis** — positive / negative / neutral classification
    - **Coreference Resolution** — maps pronouns to named entities

    *Stanford NLP-equivalent capabilities via HuggingFace Transformers + spaCy.*
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    return analyze_text(request.text)


@router.post(
    "/entities",
    summary="Extract named entities only"
)
async def extract_entities_only(
    request: NLPAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Extract only named entities from text (faster than full analysis)."""
    from app.services.nlp_service import extract_entities
    entities = extract_entities(request.text)
    return {
        "entities":     [e.model_dump() for e in entities],
        "count":        len(entities),
        "text_preview": request.text[:100],
    }


@router.post(
    "/sentiment",
    summary="Sentiment analysis only"
)
async def sentiment_only(
    request: NLPAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Run sentiment analysis on input text."""
    from app.services.nlp_service import analyze_sentiment
    result = analyze_sentiment(request.text)
    return result.model_dump()
