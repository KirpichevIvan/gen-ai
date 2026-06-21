"""Eval 6x3 для одиночного агента, PWC без валидатора и PWC с валидатором."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import VALID_TOOLS, run_pwc


CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "expected_tools": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["USD"],
        "needs_calculate": True,
    },
    {
        "id": "Q2",
        "query": "Какая сейчас реальная ключевая ставка, если инфляцию брать по последнему доступному месяцу, а не по году?",
        "expected_tools": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
        "needs_calculate": True,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "expected_tools": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "needs_calculate": True,
        "allow_empty_plan": True,
    },
    {
        "id": "Q4_validator",
        "query": "Какова накопленная инфляция за 2024 год? Не используй выдуманные агрегированные инструменты.",
        "expected_tools": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "needs_calculate": True,
    },
    {
        "id": "Q5_parallel",
        "query": (
            "Сравни курсы USD, EUR и CNY на 1 января 2022 и сегодня: "
            "для каждой валюты скажи, во сколько раз изменился курс."
        ),
        "expected_tools": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["USD", "EUR", "CNY"],
        "needs_calculate": True,
    },
    {
        "id": "Q6_real",
        "query": "Сколько лет потребуется вкладу удвоиться при текущей ключевой ставке, если проценты капитализируются ежегодно?",
        "expected_tools": {"get_key_rate", "calculate"},
        "must_have_keywords": ["лет"],
        "needs_calculate": True,
    },
]


CONFIGS = [
    {"id": "single", "label": "одиночный агент"},
    {"id": "pwc_no_validator", "label": "pwc без валидатора"},
    {"id": "pwc_validator", "label": "pwc + валидатор"},
]


def _single_used(result: dict) -> set[str]:
    return {e["call"] for e in result.get("trace", []) if "call" in e}


def _pwc_used(result: dict) -> set[str]:
    used = set()
    for event in result.get("trace", []):
        if event.get("kind") == "worker":
            used.update(event.get("used_tools") or [])
    return used


def _plan_tools(result: dict) -> set[str]:
    plan = result.get("plan")
    if plan is None:
        return set()
    tools = set()
    for sq in plan.subquestions:
        tools.update(sq.expected_tools)
    return tools


def _check(case: dict, result: dict, *, config_id: str) -> dict:
    answer = result.get("answer") or ""
    answer_lower = answer.lower()
    used = _single_used(result) if config_id == "single" else _pwc_used(result)
    plan_tools = set() if config_id == "single" else _plan_tools(result)
    hallucinated = (used | plan_tools) - VALID_TOOLS
    impossible_ok = (
        config_id != "single"
        and case.get("allow_empty_plan")
        and not plan_tools
        and any(marker in answer_lower for marker in ["невозмож", "не может", "нереш"])
    )
    if impossible_ok:
        return {
            "ok": True,
            "used_tools": sorted(used),
            "plan_tools": sorted(plan_tools),
            "hallucinated": sorted(hallucinated),
            "must_have_ok": True,
            "calculate_ok": True,
            "expected_ok": True,
            "iterations": result.get("iterations"),
            "answer_preview": answer[:240],
            "error": result.get("error"),
        }

    must_have_ok = all(kw.lower() in answer_lower for kw in case["must_have_keywords"])
    calculate_ok = (not case["needs_calculate"]) or ("calculate" in used)
    expected_ok = case["expected_tools"].issubset(used | plan_tools)
    ok = bool(answer) and not hallucinated and must_have_ok and calculate_ok and expected_ok

    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must_have_ok,
        "calculate_ok": calculate_ok,
        "expected_ok": expected_ok,
        "iterations": result.get("iterations"),
        "answer_preview": answer[:240],
        "error": result.get("error"),
    }


def _run_once(case: dict, config_id: str) -> dict:
    started = perf_counter()
    try:
        if config_id == "single":
            raw = run_agent(case["query"], max_iter=8, verbose=False)
        elif config_id == "pwc_no_validator":
            raw = run_pwc(case["query"], max_iter=3, verbose=False, validate_schema=False, parallel=True)
        elif config_id == "pwc_validator":
            raw = run_pwc(case["query"], max_iter=3, verbose=False, validate_schema=True, parallel=True)
        else:
            raise ValueError(f"unknown config: {config_id}")
    except Exception as e:
        raw = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": [], "plan": None}
    elapsed = perf_counter() - started
    checked = _check(case, raw, config_id=config_id)
    checked["elapsed_sec"] = round(elapsed, 3)
    return checked


def run_case(case: dict, *, n: int) -> dict:
    configs = {}
    for cfg in CONFIGS:
        runs = [_run_once(case, cfg["id"]) for _ in range(n)]
        configs[cfg["id"]] = {
            "label": cfg["label"],
            "pass": sum(int(r["ok"]) for r in runs),
            "runs": runs,
        }
    return {"id": case["id"], "query": case["query"], "n": n, "configs": configs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="Один прогон на кейс")
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()
    n = 1 if args.single else args.n

    results = []
    print(f"Eval S6: {len(CASES)} вопросов x 3 конфигурации x {n} прогонов\n")
    for case in CASES:
        result = run_case(case, n=n)
        results.append(result)
        print(f"{case['id']}: {case['query'][:80]}...")
        for cfg in CONFIGS:
            c = result["configs"][cfg["id"]]
            print(f"  {cfg['id']}: {c['pass']}/{n}")
        print()

    print("| id | single | pwc без валидатора | pwc + валидатор |")
    print("|---|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['id']} | "
            f"{result['configs']['single']['pass']}/{n} | "
            f"{result['configs']['pwc_no_validator']['pass']}/{n} | "
            f"{result['configs']['pwc_validator']['pass']}/{n} |"
        )

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
