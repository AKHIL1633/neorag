"""
Evaluation — LLM-as-judge scoring for agent/LLM answers, plus a small
hand-curated Q&A benchmark. QA_BENCHMARK here is the single source of truth
used by both POST /api/v1/eval/run and tests/test_evaluation.py, so the
numbers quoted in README.md's "Honest Metrics" section reflect exactly what
this benchmark measures.
"""

import re
from typing import Any, Dict, List

from loguru import logger

JUDGE_PROMPT = """You are an impartial evaluator. Score the following answer on a scale of 0.0 to 1.0.

Question: {question}
Retrieved Context: {context}
Answer: {answer}

Score criteria:
- 1.0: Answer directly addresses the question and is fully supported by context
- 0.7: Answer addresses the question but relies partly on external knowledge
- 0.4: Answer is partially relevant but contains speculation
- 0.0: Answer is off-topic, incorrect, or hallucinated

Respond with ONLY a number between 0.0 and 1.0. Nothing else.
Score:"""


def judge_answer(question: str, answer: str, context: List[str]) -> float:
    from langchain.prompts import PromptTemplate

    from app.services.llm_service import _load_llm

    llm = _load_llm()
    if not llm:
        return 0.5  # neutral when judge unavailable
    try:
        prompt = PromptTemplate.from_template(JUDGE_PROMPT)
        chain = prompt | llm
        raw = chain.invoke({"question": question, "answer": answer, "context": "\n".join(context)})
        raw_str = str(raw)
        # HuggingFacePipeline's text-generation returns the full prompt +
        # completion by default (return_full_text defaults to True), and
        # JUDGE_PROMPT itself contains the literal text "0.0 to 1.0" — a
        # naive regex search over the whole string matches that instead of
        # the model's actual score. Only search what comes after the
        # trailing "Score:" marker the model was asked to continue from.
        if "Score:" in raw_str:
            raw_str = raw_str.rsplit("Score:", 1)[-1]
        m = re.search(r"([01](?:\.\d+)?)", raw_str)
        return float(m.group(1)) if m else 0.5
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return 0.5


# ── Benchmark ─────────────────────────────────────────────────────────────────
# Each item establishes a PERSON -> ORG fact in `seed_doc`, then asks a
# question that mentions the PERSON verbatim (so entity extraction on the
# question matches the exact graph node created from the seed doc) and
# expects the ORG to surface in the answer via graph-grounded retrieval.

QA_BENCHMARK: List[Dict[str, str]] = [
    {"question": "What company did Steve Jobs found?", "expected_entity": "Apple Inc",
     "seed_doc": "Steve Jobs founded Apple Inc in Cupertino."},
    {"question": "What company did Elon Musk found?", "expected_entity": "SpaceX",
     "seed_doc": "Elon Musk founded SpaceX in Hawthorne."},
    {"question": "What company did Bill Gates found?", "expected_entity": "Microsoft",
     "seed_doc": "Bill Gates founded Microsoft in Redmond."},
    {"question": "What company did Jeff Bezos found?", "expected_entity": "Amazon",
     "seed_doc": "Jeff Bezos founded Amazon in Seattle."},
    {"question": "What company did Mark Zuckerberg found?", "expected_entity": "Facebook",
     "seed_doc": "Mark Zuckerberg founded Facebook in Menlo Park."},
    {"question": "What company did Larry Page found?", "expected_entity": "Google",
     "seed_doc": "Larry Page founded Google in Mountain View."},
    {"question": "Who does Sundar Pichai work for?", "expected_entity": "Google",
     "seed_doc": "Sundar Pichai works for Google in California."},
    {"question": "Who does Tim Cook work for?", "expected_entity": "Apple Inc",
     "seed_doc": "Tim Cook works for Apple Inc in Cupertino."},
    {"question": "Who does Satya Nadella work for?", "expected_entity": "Microsoft",
     "seed_doc": "Satya Nadella works for Microsoft in Redmond."},
    {"question": "Who does Sam Altman work for?", "expected_entity": "OpenAI",
     "seed_doc": "Sam Altman works for OpenAI in San Francisco."},
    {"question": "What company did Warren Buffett found?", "expected_entity": "Berkshire Hathaway",
     "seed_doc": "Warren Buffett founded Berkshire Hathaway in Omaha."},
    {"question": "What company did Jack Ma found?", "expected_entity": "Alibaba",
     "seed_doc": "Jack Ma founded Alibaba in Hangzhou."},
    {"question": "What company did Reed Hastings found?", "expected_entity": "Netflix",
     "seed_doc": "Reed Hastings founded Netflix in Los Gatos."},
    {"question": "What company did Daniel Ek found?", "expected_entity": "Spotify",
     "seed_doc": "Daniel Ek founded Spotify in Stockholm."},
    {"question": "What company did Brian Chesky found?", "expected_entity": "Airbnb",
     "seed_doc": "Brian Chesky founded Airbnb in San Francisco."},
    {"question": "What company did Travis Kalanick found?", "expected_entity": "Uber",
     "seed_doc": "Travis Kalanick founded Uber in San Francisco."},
    {"question": "What company did Evan Spiegel found?", "expected_entity": "Snapchat",
     "seed_doc": "Evan Spiegel founded Snapchat in Los Angeles."},
    {"question": "What company did Jensen Huang found?", "expected_entity": "Nvidia",
     "seed_doc": "Jensen Huang founded Nvidia in Santa Clara."},
    {"question": "What company did Larry Ellison found?", "expected_entity": "Oracle",
     "seed_doc": "Larry Ellison founded Oracle in Redwood City."},
    {"question": "What company did Michael Dell found?", "expected_entity": "Dell",
     "seed_doc": "Michael Dell founded Dell in Austin."},
]


def run_benchmark_item(item: Dict[str, str]) -> Dict[str, Any]:
    from app.services.ingestion import process_document
    from app.services.llm_service import answer_question

    process_document(title=f"benchmark: {item['question'][:40]}", content=item["seed_doc"])
    answer, context, _ = answer_question(item["question"])
    score = judge_answer(item["question"], answer, context)
    return {
        "question": item["question"],
        "answer": answer,
        "expected_entity": item["expected_entity"],
        "entity_found": item["expected_entity"].lower() in answer.lower(),
        "judge_score": score,
    }


def run_benchmark() -> Dict[str, Any]:
    results = [run_benchmark_item(item) for item in QA_BENCHMARK]
    mean_score = sum(r["judge_score"] for r in results) / len(results)
    hit_rate = sum(r["entity_found"] for r in results) / len(results)
    return {
        "results": results,
        "mean_judge_score": round(mean_score, 4),
        "entity_hit_rate": round(hit_rate, 4),
        "item_count": len(results),
    }
