import time
from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.schemas import (
    QARequest, QAResponse, SummarizeRequest, SummarizeResponse
)
from app.models.user_model import User
from app.services.llm_service import answer_question, summarize_text, generate_text
from app.core.auth import get_current_user
from app.core.rate_limit import limiter

router = APIRouter(prefix="/llm", tags=["LLM — Q&A & Generation"])


@router.post(
    "/ask",
    response_model=QAResponse,
    summary="Ask a question — answered using a LangGraph agent grounded by the Knowledge Graph"
)
@limiter.limit("30/minute")
async def ask_question(
    request: Request,
    payload: QARequest,
    current_user: User = Depends(get_current_user)
):
    """
    **Graph-grounded, agentic Q&A:**

    1. Classify the question and extract entities
    2. Retrieve context from Neo4j (graph) and Qdrant (semantic search)
    3. Synthesize an answer, then self-critique and (if needed) retry with a
       widened search — see app.services.agent_pipeline
    4. Return the grounded answer

    Set `use_graph=false` to skip graph-based context.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    try:
        answer, context, model = answer_question(
            payload.question,
            use_graph=payload.use_graph,
            max_tokens=payload.max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    elapsed = round((time.time() - start) * 1000, 2)

    return QAResponse(
        question=payload.question,
        answer=answer,
        context=context,
        model_used=model,
        processing_time_ms=elapsed,
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize text using Llama 3.2"
)
@limiter.limit("30/minute")
async def summarize(
    request: Request,
    payload: SummarizeRequest,
    current_user: User = Depends(get_current_user)
):
    """
    **Text summarization using Llama 3.2.**

    Provide either:
    - `text`: direct text to summarize
    - `document_id`: ID of a previously ingested document
    """
    text = payload.text
    if not text:
        raise HTTPException(status_code=400, detail="Provide text or document_id")

    start = time.time()
    try:
        summary, model = summarize_text(text, max_length=payload.max_length)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    elapsed = round((time.time() - start) * 1000, 2)

    return SummarizeResponse(
        summary=summary,
        original_length=len(text.split()),
        summary_length=len(summary.split()),
        processing_time_ms=elapsed,
    )


@router.post(
    "/generate",
    summary="Free-form text generation using Llama 3.2"
)
@limiter.limit("30/minute")
async def generate(
    request: Request,
    prompt: str,
    max_tokens: int = 256,
    current_user: User = Depends(get_current_user)
):
    """Generate text from a free-form prompt using the loaded LLM."""
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        result, model = generate_text(prompt, max_tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"generated_text": result, "model_used": model, "prompt": prompt}
