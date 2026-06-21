"""Замер ложных принятий Критика на заведомо сломанных ответах."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer


FAKE_BROKEN = [
    {
        "id": "arithmetic_without_calculate",
        "title": "арифметика без calculate",
        "question": "На сколько курс EUR выше курса USD?",
        "plan": Plan(
            reasoning="Нужно получить два курса и сравнить их.",
            subquestions=[
                SubQuestion(id=1, question="Какой курс USD?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Какой курс EUR?", expected_tools=["get_fx_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Какой курс USD?",
                answer="USD=82.5, EUR=89, разница=6.5 руб.",
                used_tools=["get_fx_rate"],
            )
        },
    },
    {
        "id": "invented_number",
        "title": "выдуманное число",
        "question": "Какая ключевая ставка сейчас?",
        "plan": Plan(
            reasoning="Достаточно запросить ключевую ставку.",
            subquestions=[
                SubQuestion(id=1, question="Какая ключевая ставка сейчас?", expected_tools=["get_key_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Какая ключевая ставка сейчас?",
                answer="Ключевая ставка составляет 13.37% годовых.",
                used_tools=[],
            )
        },
    },
    {
        "id": "inconsistent_answers",
        "title": "несогласованные данные",
        "question": "Во сколько раз USD вырос с 2022 года?",
        "plan": Plan(
            reasoning="Нужно получить стартовый и конечный курс, затем посчитать отношение.",
            subquestions=[
                SubQuestion(id=1, question="Курс USD на 2022-01-01", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс USD сегодня", expected_tools=["get_fx_rate"]),
                SubQuestion(id=3, question="Отношение текущего курса к начальному", expected_tools=["calculate"], depends_on=[1, 2]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="Курс USD на 2022-01-01", answer="USD=74.29 руб.", used_tools=["get_fx_rate"]),
            2: WorkerAnswer(subquestion_id=2, question_snippet="Курс USD сегодня", answer="USD=82.50 руб.", used_tools=["get_fx_rate"]),
            3: WorkerAnswer(subquestion_id=3, question_snippet="Отношение", answer="USD вырос в 2.8 раза.", used_tools=["calculate"]),
        },
    },
    {
        "id": "missing_subquestion",
        "title": "пропущенный подвопрос",
        "question": "Какая реальная ключевая ставка с учётом последней инфляции?",
        "plan": Plan(
            reasoning="Нужно получить ставку, инфляцию и вычесть.",
            subquestions=[
                SubQuestion(id=1, question="Текущая ключевая ставка", expected_tools=["get_key_rate"]),
                SubQuestion(id=2, question="Последняя инфляция", expected_tools=["get_inflation"]),
                SubQuestion(id=3, question="Реальная ставка", expected_tools=["calculate"], depends_on=[1, 2]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="Текущая ключевая ставка", answer="Ключевая ставка 16%.", used_tools=["get_key_rate"]),
            3: WorkerAnswer(subquestion_id=3, question_snippet="Реальная ставка", answer="Реальная ставка около 8%.", used_tools=["calculate"]),
        },
    },
    {
        "id": "worker_error_accepted",
        "title": "ответ с ошибкой исполнителя",
        "question": "Какой CPI в марте 2026?",
        "plan": Plan(
            reasoning="Нужно запросить инфляцию за март 2026.",
            subquestions=[
                SubQuestion(id=1, question="ИПЦ за март 2026", expected_tools=["get_inflation"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="ИПЦ за март 2026",
                answer="(ошибка: нет данных ИПЦ на 2026-03)",
                used_tools=["get_inflation"],
            )
        },
    },
]


def run_case(case: dict, *, temperature: float, n: int) -> dict:
    false_accepts = 0
    runs = []
    for _ in range(n):
        verdict = critic(
            case["question"],
            case["plan"],
            case["answers"],
            temperature=temperature,
        )
        false_accepts += int(verdict.ok)
        runs.append(verdict.model_dump())
    return {"false_accepts": false_accepts, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    results = []
    print("| Битый кейс | T=0.0 | T=0.7 |")
    print("|---|---:|---:|")
    for case in FAKE_BROKEN:
        r0 = run_case(case, temperature=0.0, n=args.n)
        r7 = run_case(case, temperature=0.7, n=args.n)
        results.append(
            {
                "id": case["id"],
                "title": case["title"],
                "n": args.n,
                "temperature_0_0": r0,
                "temperature_0_7": r7,
            }
        )
        print(f"| {case['title']} | {r0['false_accepts']}/{args.n} | {r7['false_accepts']}/{args.n} |")

    out = Path(__file__).parent / "critic_experiment_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
