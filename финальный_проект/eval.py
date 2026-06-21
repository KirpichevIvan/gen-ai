from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from agent import run_agent, verify_grounding
from judge import judge_answer
from rag import build_index, stats
from schema import AssistantAnswer, EvalCase, EvalRow, Evidence, WorkerResult

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
GOLD_PATH = INPUT_DIR / "gold.json"

load_dotenv(BASE_DIR / ".env")


def norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = text.replace("‐", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def count_keyword_hits(answer: str, keywords: list[str]) -> int:
    text = norm(answer)
    return sum(1 for keyword in keywords if norm(keyword) in text)


def count_source_hits(result: dict, gold_sources: list[str]) -> int:
    sources = set()
    for evidence in result["answer"]["evidence"]:
        source = evidence["source"]
        if source.startswith("habr_"):
            sources.add("_".join(source.split("_")[:2]))
    return sum(1 for source in gold_sources if source in sources)


def route_ok(expected: list[str], actual: list[str]) -> bool:
    return set(expected).issubset(set(actual)) and len(actual) == len(expected)


def search_calls(result: dict) -> int:
    return sum(1 for worker in result["workers"] if worker["tool"] in {"local_rag", "stackoverflow_search"})


def stackoverflow_mode(result: dict) -> str | None:
    modes = []
    for worker in result["workers"]:
        if worker["tool"] == "stackoverflow_search":
            raw = worker.get("raw", {}).get("raw", {})
            modes.append(raw.get("mode", "unknown"))
    return ",".join(modes) if modes else None


def evaluate_case(case: EvalCase, *, use_llm_answer: bool, use_llm_judge: bool) -> tuple[EvalRow, dict]:
    result = run_agent(
        case.question,
        forced_type=None,
        expected_route=case.expected_route,
        use_llm=use_llm_answer,
    )
    answer_text = result["answer"]["answer"]
    answer_obj = AssistantAnswer(**result["answer"])
    keyword_hits = count_keyword_hits(answer_text, case.expected_keywords)
    source_hits = count_source_hits(result, case.gold_sources)
    actual_route = result["answer"]["route"]
    r_ok = route_ok(case.expected_route, actual_route)
    keyword_min = max(1, len(case.expected_keywords) // 2) if case.expected_keywords else 0
    source_ok = source_hits == len(case.gold_sources)
    keyword_ok = keyword_hits >= keyword_min
    ghost_quotes = result["critic"]["ghost_quotes"]
    unsupported_numbers = result["critic"].get("unsupported_numbers", 0)
    evidence_texts = []
    for worker in result["workers"]:
        if worker["tool"] == "local_rag":
            evidence_texts.extend(hit.get("text", "") for hit in worker["raw"].get("hits", []))
        else:
            evidence_texts.extend(row.get("summary", "") for row in worker["raw"].get("raw", {}).get("results", []))
    judge = judge_answer(
        case.question,
        answer_obj,
        evidence_texts,
        expect_abstain=case.expect_abstain,
        use_llm=use_llm_judge,
    )
    if case.expect_abstain:
        passed = (
            r_ok
            and answer_obj.status == "insufficient"
            and judge.verdict == "abstain_ok"
            and ghost_quotes == 0
            and unsupported_numbers == 0
        )
    else:
        passed = (
            r_ok
            and keyword_ok
            and source_ok
            and ghost_quotes == 0
            and unsupported_numbers == 0
            and judge.verdict in {"supported", "partially_supported"}
        )

    row = EvalRow(
        id=case.id,
        type=case.type,
        question=case.question,
        expected_route=case.expected_route,
        actual_route=actual_route,
        route_ok=r_ok,
        keyword_hits=keyword_hits,
        keyword_total=len(case.expected_keywords),
        source_hits=source_hits,
        source_total=len(case.gold_sources),
        passed=passed,
        steps=result["steps"],
        ghost_quotes=ghost_quotes,
        unsupported_numbers=unsupported_numbers,
        judge_verdict=judge.verdict,
        status=answer_obj.status,
        token_estimate=result.get("token_estimate", 0),
        cost_usd_estimate=result.get("cost_usd_estimate", 0.0),
        hit_at_5=round(source_hits / len(case.gold_sources), 3) if case.gold_sources else 0.0,
        search_calls=search_calls(result),
        latency_sec=result.get("elapsed_sec", 0.0),
        so_mode=stackoverflow_mode(result),
        answer=answer_text,
    )
    result["judge"] = judge.model_dump()
    return row, result


def write_csv(rows: list[EvalRow], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].model_dump().keys()))
        writer.writeheader()
        for row in rows:
            data = row.model_dump()
            data["expected_route"] = ",".join(data["expected_route"])
            data["actual_route"] = ",".join(data["actual_route"])
            writer.writerow(data)


def write_markdown(rows: list[EvalRow], path: Path) -> None:
    lines = [
        "| id | type | pass | tools | hit@5 | judge | status | ghosts | nums | steps | search | tokens | lat,s | so |",
        "|---|---|:---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        passed = "yes" if row.passed else "no"
        tools = "+".join(row.actual_route)
        so_mode = row.so_mode or ""
        lines.append(
            f"| {row.id} | {row.type} | {passed} | {tools} | {row.hit_at_5:.2f} | "
            f"{row.judge_verdict} | {row.status} | {row.ghost_quotes} | {row.unsupported_numbers} | "
            f"{row.steps} | {row.search_calls} | {row.token_estimate} | {row.latency_sec:.1f} | {so_mode} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[EvalRow]) -> dict:
    total = len(rows)
    passed = sum(row.passed for row in rows)
    ghost_quotes = sum(row.ghost_quotes for row in rows)
    unsupported_numbers = sum(row.unsupported_numbers for row in rows)
    avg_tokens = round(sum(row.token_estimate for row in rows) / total, 1) if total else 0.0
    total_cost = round(sum(row.cost_usd_estimate for row in rows), 6)
    so_modes: dict[str, int] = {}
    for row in rows:
        if row.so_mode:
            so_modes[row.so_mode] = so_modes.get(row.so_mode, 0) + 1
    by_type = {}
    for row in rows:
        bucket = by_type.setdefault(row.type, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(row.passed)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "ghost_quotes": ghost_quotes,
        "unsupported_numbers": unsupported_numbers,
        "avg_steps": round(sum(row.steps for row in rows) / total, 2) if total else 0.0,
        "avg_token_estimate": avg_tokens,
        "total_cost_usd_estimate": total_cost,
        "stackoverflow_modes": so_modes,
        "by_type": by_type,
        "corpus": stats(),
    }


def run_ghost_tests() -> dict:
    worker = WorkerResult(
        subquestion_id=1,
        tool="local_rag",
        answer="grounded",
        evidence=[
            Evidence(
                source="synthetic",
                quote="RAG uses retrieved context to ground the answer.",
                url=None,
            )
        ],
        raw={"hits": [{"text": "RAG uses retrieved context to ground the answer."}]},
    )
    valid = AssistantAnswer(
        status="answered",
        answer="RAG uses retrieved context to ground the answer.",
        route=["local_rag"],
        evidence=worker.evidence,
        confidence=0.9,
    )
    ghost = AssistantAnswer(
        status="answered",
        answer="RAG uses retrieved context to ground the answer.",
        route=["local_rag"],
        evidence=[Evidence(source="synthetic", quote="This quote is absent from context.", url=None)],
        confidence=0.9,
    )
    fake_number = AssistantAnswer(
        status="answered",
        answer="RAG improves accuracy by 999%.",
        route=["local_rag"],
        evidence=worker.evidence,
        confidence=0.9,
    )
    valid_report = verify_grounding(valid, [worker])
    ghost_report = verify_grounding(ghost, [worker])
    number_report = verify_grounding(fake_number, [worker])
    return {
        "valid_passed": valid_report.ghost_quotes == 0 and valid_report.unsupported_numbers == 0,
        "ghost_quote_caught": ghost_report.ghost_quotes == 1,
        "unsupported_number_caught": number_report.unsupported_numbers == 1,
        "reports": {
            "valid": valid_report.model_dump(),
            "ghost": ghost_report.model_dump(),
            "unsupported_number": number_report.model_dump(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Use OpenAI structured refinement when OPENAI_API_KEY is set")
    parser.add_argument("--llm-judge", action="store_true", help="Use OpenAI as LLM-as-judge when OPENAI_API_KEY is set")
    parser.add_argument("--ghost-tests", action="store_true", help="Run synthetic hallucination gate tests")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.ghost_tests:
        report = run_ghost_tests()
        (OUTPUT_DIR / "ghost_tests.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    build_index()
    cases = [EvalCase(**item) for item in json.loads(GOLD_PATH.read_text(encoding="utf-8"))]

    rows: list[EvalRow] = []
    traces = []
    for case in cases:
        row, result = evaluate_case(case, use_llm_answer=args.llm, use_llm_judge=args.llm_judge)
        rows.append(row)
        traces.append({"id": case.id, **result})
        print(
            f"{case.id}: passed={row.passed} route={','.join(row.actual_route)} "
            f"status={row.status} judge={row.judge_verdict} ghost={row.ghost_quotes}"
        )

    summary = summarize(rows)
    (OUTPUT_DIR / "eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": [row.model_dump() for row in rows]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "hallucination_report.json").write_text(
        json.dumps(
            {
                "ghost_quotes": summary["ghost_quotes"],
                "unsupported_numbers": summary["unsupported_numbers"],
                "synthetic_gate_tests": run_ghost_tests(),
                "unsupported_cases": [
                    row.model_dump()
                    for row in rows
                    if row.judge_verdict == "unsupported" or row.ghost_quotes or row.unsupported_numbers
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "trace.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, OUTPUT_DIR / "eval_table.csv")
    write_markdown(rows, OUTPUT_DIR / "eval_table.md")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
