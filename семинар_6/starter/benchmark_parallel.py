"""Замер ускорения PWC при параллельном выполнении независимых подвопросов."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator import run_pwc


CASES = [
    {
        "id": "Q1_usd_growth",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
    },
    {
        "id": "bonus_3_currencies",
        "query": (
            "Сравни курсы USD, EUR и CNY на 1 января 2022 и сегодня: "
            "для каждой валюты скажи, во сколько раз изменился курс."
        ),
    },
]


def timed_run(query: str, *, parallel: bool) -> dict:
    started = perf_counter()
    result = run_pwc(query, max_iter=3, verbose=False, validate_schema=True, parallel=parallel)
    elapsed = perf_counter() - started
    return {
        "elapsed_sec": round(elapsed, 3),
        "ok": bool(result.get("answer")),
        "answer_preview": (result.get("answer") or result.get("error") or "")[:240],
        "iterations": result.get("iterations"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1)
    args = ap.parse_args()

    results = []
    print("| case | sequential, sec | parallel, sec | speedup |")
    print("|---|---:|---:|---:|")
    for case in CASES:
        seq_runs = [timed_run(case["query"], parallel=False) for _ in range(args.n)]
        par_runs = [timed_run(case["query"], parallel=True) for _ in range(args.n)]
        seq_avg = sum(r["elapsed_sec"] for r in seq_runs) / args.n
        par_avg = sum(r["elapsed_sec"] for r in par_runs) / args.n
        speedup = seq_avg / par_avg if par_avg else 0.0
        results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "n": args.n,
                "sequential": seq_runs,
                "parallel": par_runs,
                "sequential_avg_sec": round(seq_avg, 3),
                "parallel_avg_sec": round(par_avg, 3),
                "speedup": round(speedup, 2),
            }
        )
        print(f"| {case['id']} | {seq_avg:.3f} | {par_avg:.3f} | {speedup:.2f}x |")

    out = Path(__file__).parent / "benchmark_parallel_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
