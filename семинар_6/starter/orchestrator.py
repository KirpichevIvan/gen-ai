"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker


VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана: пустой список означает, что план корректен."""
    errors: list[str] = []

    seen_ids: set[int] = set()
    for sq in plan.subquestions:
        if sq.id in seen_ids:
            errors.append(f"дублируется id подвопроса: {sq.id}")
        seen_ids.add(sq.id)

        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(f"подвопрос {sq.id}: неизвестный инструмент {tool!r}")

    known_ids = {sq.id for sq in plan.subquestions}
    for sq in plan.subquestions:
        for dep_id in sq.depends_on:
            if dep_id not in known_ids:
                errors.append(f"подвопрос {sq.id}: depends_on ссылается на неизвестный id {dep_id}")
            if dep_id == sq.id:
                errors.append(f"подвопрос {sq.id}: depends_on ссылается на самого себя")

    return errors


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Разбить подвопросы на уровни зависимостей."""
    by_id = {s.id: s for s in subqs}
    visiting: set[int] = set()
    depth_cache: dict[int, int] = {}

    def depth(node_id: int, path: list[int]) -> int:
        if node_id in path or node_id in visiting:
            raise ValueError(f"цикл в depends_on: {path + [node_id]}")
        if node_id not in by_id:
            return -1
        if node_id in depth_cache:
            return depth_cache[node_id]

        visiting.add(node_id)
        dep_depths = [
            depth(dep_id, path + [node_id])
            for dep_id in by_id[node_id].depends_on
            if dep_id in by_id
        ]
        visiting.remove(node_id)
        depth_cache[node_id] = (max(dep_depths) + 1) if dep_depths else 0
        return depth_cache[node_id]

    for sq in subqs:
        depth(sq.id, [])

    levels: list[list[SubQuestion]] = []
    for sq in subqs:
        level_index = depth_cache[sq.id]
        while len(levels) <= level_index:
            levels.append([])
        levels[level_index].append(sq)
    return levels


def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы одного уровня параллельно."""
    if not level:
        return {}
    if len(level) == 1:
        sq = level[0]
        return {sq.id: worker(sq, prev_answers=prev_answers)}

    answers: dict[int, WorkerAnswer] = {}
    with ThreadPoolExecutor(max_workers=len(level)) as pool:
        futures = {pool.submit(worker, sq, prev_answers): sq for sq in level}
        for future in as_completed(futures):
            sq = futures[future]
            answers[sq.id] = future.result()
    return answers


def _validate_or_replan(
    question: str,
    plan: Plan,
    trace: list[dict[str, Any]],
    *,
    validate_schema: bool,
    verbose: bool,
) -> tuple[Plan, list[str]]:
    if not validate_schema:
        return plan, []

    errors = validate_plan(plan)
    if not errors:
        return plan, []

    if verbose:
        print(f"\n[plan validator] {errors}")
    trace.append({"iter": 0, "kind": "plan_validation", "errors": errors})

    plan = planner(question, feedback=f"Инструменты не существуют: {errors}")
    errors = validate_plan(plan)
    if errors:
        trace.append({"iter": 0, "kind": "plan_validation", "errors": errors})
    return plan, errors


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ из ответов исполнителей."""
    parts = [answers[i].answer for i in sorted(answers)]
    return " · ".join(parts)


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    validate_schema: bool = True,
    parallel: bool = True,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    plan, plan_errors = _validate_or_replan(
        question,
        plan,
        trace,
        validate_schema=validate_schema,
        verbose=verbose,
    )
    if plan_errors:
        return {
            "answer": None,
            "error": f"план не прошёл валидацию после перепланирования: {plan_errors}",
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
        }

    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            deps = f" depends_on={sq.depends_on}" if sq.depends_on else ""
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}]{deps} {sq.question}")

    if not plan.subquestions:
        return {
            "answer": plan.reasoning,
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
        }

    answers: dict[int, WorkerAnswer] = {}
    for iter_num in range(1, max_iter + 1):
        answers = {}
        levels = _topological_levels(plan.subquestions)

        for level_num, level in enumerate(levels, start=1):
            if parallel:
                level_answers = execute_level(level, answers)
            else:
                level_answers = {sq.id: worker(sq, prev_answers=answers) for sq in level}

            for sq in level:
                ans = level_answers[sq.id]
                answers[sq.id] = ans
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "level": level_num,
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [L{level_num} / {sq.id}] -> {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "OK" if verdict.ok else "FAIL"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            return {
                "answer": _synthesize(question, plan, answers),
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        if verdict.action == "rework":
            feedback = f"{verdict.reason}. Переделай подвопросы: {verdict.rework_ids}."
        else:
            feedback = verdict.reason

        plan = planner(question, feedback=feedback)
        plan, plan_errors = _validate_or_replan(
            question,
            plan,
            trace,
            validate_schema=validate_schema,
            verbose=verbose,
        )
        if plan_errors:
            return {
                "answer": None,
                "error": f"план не прошёл валидацию после перепланирования: {plan_errors}",
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        trace.append(
            {
                "iter": iter_num,
                "kind": "plan",
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            }
        )

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-validator", action="store_true")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--trace", type=Path, default=None, help="Куда сохранить JSON-лог")
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(
        q,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        validate_schema=not args.no_validator,
        parallel=not args.sequential,
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
