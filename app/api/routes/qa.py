import time
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import (
    QARequest, QAResponse, SummarizeRequest, SummarizeResponse
)
from app.services.llm_service import answer_question, summarize_text, generate_text
from app.core.auth import get_current_user

router = APIRouter(prefix="/llm", tags=["LLM — Q&A & Generation"])


@router.post(
    "/ask",
    response_model=QAResponse,
    summary="Ask a question — answered using Llama 3.2 + Knowledge Graph"
)
async def ask_question(
    request: QARequest,
    current_user: dict = Depends(get_current_user)
):
    """
    **Graph-grounded Q&A using Llama 3.2:**

    1. Extract key entities from the question
    2. Query Neo4j graph for relevant context (up to 2 hops)
    3. Pass graph context + question to Llama 3.2
    4. Return grounded, accurate answer

    Set `use_graph=false` to get a pure LLM answer without graph context.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    answer, context, model = answer_question(
        request.question,
        use_graph=request.use_graph,
        max_tokens=request.max_tokens,
    )
    elapsed = round((time.time() - start) * 1000, 2)

    return QAResponse(
        question=request.question,
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
async def summarize(
    request: SummarizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    **Text summarization using Llama 3.2.**

    Provide either:
    - `text`: direct text to summarize
    - `document_id`: ID of a previously ingested document
    """
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Provide text or document_id")

    start   = time.time()
    summary, model = summarize_text(text, max_length=request.max_length)
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
async def generate(
    prompt: str,
    max_tokens: int = 256,
    current_user: dict = Depends(get_current_user)
):
    """Generate text from a free-form prompt using the loaded LLM."""
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    result, model = generate_text(prompt, max_tokens)
    return {"generated_text": result, "model_used": model, "prompt": prompt}
