from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from schema import AssistantAnswer, CriticReport, Evidence, Plan, SubQuestion, ToolName, WorkerResult

from rag import answer_local
from stackoverflow_tool import answer_stackoverflow

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

load_dotenv(BASE_DIR / ".env")


def plan(question: str, forced_type: str | None = None) -> Plan:
    q = question.lower()
    if forced_type == "local_only":
        route: list[ToolName] = ["local_rag"]
    elif forced_type == "stackoverflow":
        route = ["stackoverflow_search"]
    elif forced_type in {"mixed", "out_of_scope"}:
        route = ["local_rag", "stackoverflow_search"]
    elif any(word in q for word in ["сравни", "сопостав", "сочета", "дополни"]) or (
        "статья" in q and any(word in q for word in ["совет", "stackoverflow", "stack overflow"])
    ) or (
        "habr" in q and any(word in q for word in ["stackoverflow", "stack overflow"])
    ):
        route = ["local_rag", "stackoverflow_search"]
    elif any(word in q for word in ["автор", "claude code", "в статье", "статье", "habr"]):
        route = ["local_rag"]
    elif any(
        word in q
        for word in [
            "stackoverflow",
            "stack overflow",
            "python",
            "api",
            "chunk size",
            "hybrid",
            "bm25",
            "цитат",
            "метрик",
            "eval",
            "evaluation",
            "retrieval",
            "generation",
        ]
    ):
        route = ["stackoverflow_search"]
    else:
        route = ["local_rag"]

    subquestions = [
        SubQuestion(id=i, question=question, tool=tool)
        for i, tool in enumerate(route, start=1)
    ]
    return Plan(reasoning=f"Вопрос требует инструментов: {', '.join(route)}.", subquestions=subquestions)


def run_worker(subquestion: SubQuestion) -> WorkerResult:
    if subquestion.tool == "local_rag":
        raw = answer_local(subquestion.question)
    else:
        raw = answer_stackoverflow(subquestion.question)
    return WorkerResult(
        subquestion_id=subquestion.id,
        tool=subquestion.tool,
        answer=raw["answer"],
        evidence=[Evidence(**item) for item in raw["evidence"]],
        raw=raw,
    )


def run_workers(plan_obj: Plan) -> list[WorkerResult]:
    if len(plan_obj.subquestions) <= 1:
        return [run_worker(sq) for sq in plan_obj.subquestions]
    results: list[WorkerResult] = []
    with ThreadPoolExecutor(max_workers=len(plan_obj.subquestions)) as pool:
        futures = {pool.submit(run_worker, sq): sq for sq in plan_obj.subquestions}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.subquestion_id)


def local_rag_needs_retry(worker: WorkerResult) -> bool:
    if worker.tool != "local_rag":
        return False
    hits = worker.raw.get("hits", [])
    if not hits:
        return True
    return float(hits[0].get("score", 0.0)) < 15.0


def make_local_retry_question(question: str, worker: WorkerResult) -> str:
    hits = worker.raw.get("hits", [])
    titles = ", ".join(dict.fromkeys(hit.get("title", "") for hit in hits[:3] if hit.get("title")))
    return (
        f"{question}\n"
        "Повторный поиск: найди более конкретные факты, определения, числа и ограничения; "
        f"проверь соседние фрагменты в найденных источниках: {titles}."
    )


def maybe_retry_local_rag(question: str, workers: list[WorkerResult]) -> list[WorkerResult]:
    retry_workers: list[WorkerResult] = []
    next_id = max((worker.subquestion_id for worker in workers), default=0) + 1
    for worker in workers:
        if not local_rag_needs_retry(worker):
            continue
        retry_question = make_local_retry_question(question, worker)
        raw = answer_local(retry_question)
        raw["retry"] = True
        raw["retry_reason"] = "first local_rag top BM25 score below 15.0"
        raw["retry_question"] = retry_question
        retry_workers.append(
            WorkerResult(
                subquestion_id=next_id,
                tool="local_rag",
                answer=raw["answer"],
                evidence=[Evidence(**item) for item in raw["evidence"]],
                raw=raw,
            )
        )
        next_id += 1
    return workers + retry_workers


def synthesize(question: str, route: list[ToolName], workers: list[WorkerResult]) -> AssistantAnswer:
    answer_parts = [worker.answer for worker in workers]
    evidence = [item for worker in workers for item in worker.evidence]
    answer = " ".join(answer_parts)
    if len(route) > 1:
        answer = (
            "Двухуровневый ответ: локальный корпус Habr дает предметный контекст, "
            "а Stack Overflow fallback добавляет практическую инженерную проверку. "
            + answer
        )
    status = "answered" if evidence else "insufficient"
    return AssistantAnswer(
        status=status,
        answer=answer,
        route=route,
        evidence=evidence[:6],
        confidence=(0.72 if len(route) > 1 else 0.66) if evidence else 0.2,
        limitations=["Без LLM-ключа ответ собран извлекательно из найденных фрагментов."]
        if evidence
        else ["Инструменты не вернули проверяемых источников."],
    )


def verify_grounding(answer: AssistantAnswer, workers: list[WorkerResult], expected_route: list[ToolName] | None = None) -> CriticReport:
    haystack_parts = []
    for worker in workers:
        if worker.tool == "local_rag":
            for hit in worker.raw.get("hits", []):
                haystack_parts.append(hit.get("text", ""))
        else:
            for row in worker.raw.get("raw", {}).get("results", []):
                haystack_parts.append(row.get("summary", ""))
    haystack = re.sub(r"\s+", " ", "\n".join(haystack_parts)).lower()

    ghost_quotes = 0
    notes = []
    for evidence in answer.evidence:
        quote = re.sub(r"\s+", " ", evidence.quote).lower()
        if quote and quote not in haystack:
            ghost_quotes += 1
            notes.append(f"Ghost quote: {evidence.source}")

    unsupported_numbers = 0
    answer_numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", answer.answer))
    context_numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", haystack))
    for number in sorted(answer_numbers - context_numbers):
        unsupported_numbers += 1
        notes.append(f"Unsupported number: {number}")

    missing: list[ToolName] = []
    if expected_route:
        for tool in expected_route:
            if tool not in answer.route:
                missing.append(tool)

    return CriticReport(
        ok=ghost_quotes == 0 and unsupported_numbers == 0 and not missing,
        ghost_quotes=ghost_quotes,
        unsupported_numbers=unsupported_numbers,
        missing_expected_tools=missing,
        notes=notes,
    )


def parse_with_response_model(client: Any, *, model: str, messages: list[dict[str, str]], response_model: Any, max_retries: int) -> Any:
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_model,
                temperature=0,
                max_completion_tokens=1200,
            )
            return completion.choices[0].message.parsed
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("structured parsing failed")


def maybe_llm_refine(question: str, draft: AssistantAnswer) -> AssistantAnswer:
    if not os.environ.get("OPENAI_API_KEY"):
        return draft
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            "Перепиши ответ кратко на русском, сохрани все route/evidence/confidence. "
            "Не добавляй факты вне evidence.\n\n"
            f"Вопрос: {question}\nЧерновик JSON:\n{draft.model_dump_json(indent=2)}"
        )
        parsed = parse_with_response_model(
            client,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_model=AssistantAnswer,
            max_retries=2,
        )
        return parsed or draft
    except Exception:
        return draft


def run_agent(
    question: str,
    *,
    forced_type: str | None = None,
    expected_route: list[ToolName] | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    started = time.time()
    plan_obj = plan(question, forced_type=forced_type)
    workers = run_workers(plan_obj)
    workers = maybe_retry_local_rag(question, workers)
    route = list(dict.fromkeys(worker.tool for worker in workers))
    answer = synthesize(question, route, workers)
    if use_llm:
        answer = maybe_llm_refine(question, answer)
    critic = verify_grounding(answer, workers, expected_route=expected_route)
    token_estimate = estimate_tokens(
        question
        + "\n"
        + answer.answer
        + "\n"
        + "\n".join(e.quote for e in answer.evidence)
    )
    return {
        "question": question,
        "plan": plan_obj.model_dump(),
        "workers": [worker.model_dump() for worker in workers],
        "answer": answer.model_dump(),
        "critic": critic.model_dump(),
        "steps": len(workers),
        "token_estimate": token_estimate,
        "cost_usd_estimate": 0.0,
        "elapsed_sec": round(time.time() - started, 3),
    }


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--type", choices=["local_only", "stackoverflow", "mixed", "out_of_scope"], default=None)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--trace", type=Path, default=None)
    args = parser.parse_args()

    result = run_agent(args.question, forced_type=args.type, use_llm=args.llm)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
