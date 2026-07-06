from fastapi import APIRouter, Depends

from app.core.auth import get_current_user
from app.models.schemas import BenchmarkRunResponse
from app.models.user_model import User
from app.services.evaluation import run_benchmark

router = APIRouter(prefix="/eval", tags=["Evaluation"])


@router.post(
    "/run",
    response_model=BenchmarkRunResponse,
    summary="Run the Q&A benchmark and return per-question + mean judge scores",
)
async def run_eval(current_user: User = Depends(get_current_user)):
    """
    Runs the hand-curated 20-question benchmark (app.services.evaluation.QA_BENCHMARK):
    ingests each seed document, asks the paired question, scores the answer
    with the LLM-as-judge, and reports whether the expected entity actually
    appears in the answer.
    """
    return run_benchmark()
