"""Evaluate chunking strategies on the homework gold set.

Metric: hit-rate@k at source-document level.

For each question we retrieve top-k chunks. A gold source is counted as found
if at least one retrieved chunk belongs to that source document. For multi-hop
questions with several gold sources the score is the fraction of required
documents found in top-k.

Commands:
    python eval.py
    python eval.py --strategies fixed recursive --k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import VALID_STRATEGIES, get_collection, ingest, print_safe, result_rows


BASE_DIR = Path(__file__).resolve().parent
GOLD_PATH = BASE_DIR / "data" / "gold.json"
RESULTS_PATH = BASE_DIR / "eval_results.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def source_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("__", 1)[0]


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    retrieved_sources = {source_from_chunk_id(chunk_id) for chunk_id in retrieved_ids}
    found = [source for source in gold_sources if source in retrieved_sources]
    return len(found) / len(gold_sources)


def run_strategy(strategy: str, k: int, auto_ingest: bool) -> dict:
    collection = get_collection(strategy)
    if collection.count() == 0:
        if not auto_ingest:
            raise RuntimeError(
                f"Collection '{strategy}' is empty. Run: python pipeline.py ingest --strategy {strategy}"
            )
        print_safe(f"Collection '{strategy}' is empty, ingesting first...")
        ingest(strategy=strategy)
        collection = get_collection(strategy)

    total = 0.0
    rows = []
    gold = load_gold()

    print_safe(f"\n=== {strategy.upper()} chunking | hit-rate@{k} ===")
    for item in gold:
        hits = collection.query(query_texts=[item["question"]], n_results=k)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [source_from_chunk_id(chunk_id) for chunk_id in retrieved_ids]
        unique_sources = list(dict.fromkeys(retrieved_sources))
        score = hit_rate(retrieved_ids, item["gold_sources"])
        total += score

        mark = "OK" if score == 1.0 else ("PART" if score > 0 else "MISS")
        print_safe(
            f"[{item['id']:02d}] {item['type']:<14} hit@{k}={score:.2f} {mark} | "
            f"gold={item['gold_sources']} | retrieved={unique_sources[:5]}"
        )

        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "gold_sources": item["gold_sources"],
                "retrieved_ids": retrieved_ids,
                "retrieved_sources": unique_sources,
                "score": score,
                "top_chunks": [
                    {
                        "id": row["id"],
                        "source": row["source"],
                        "distance": row["distance"],
                        "preview": row["text"][:300],
                    }
                    for row in result_rows(hits)
                ],
            }
        )

    mean = total / len(gold)
    print_safe(f"TOTAL {strategy}: hit-rate@{k} = {mean:.3f} ({total:.2f} / {len(gold)})")
    return {"strategy": strategy, "k": k, "mean": mean, "results": rows}


def compare(results: list[dict]) -> dict:
    if len(results) < 2:
        return {}
    by_strategy = {item["strategy"]: item for item in results}
    fixed = by_strategy.get("fixed")
    recursive = by_strategy.get("recursive")
    if not fixed or not recursive:
        return {}

    per_question = []
    fixed_rows = {row["id"]: row for row in fixed["results"]}
    recursive_rows = {row["id"]: row for row in recursive["results"]}
    for item_id in sorted(fixed_rows):
        diff = recursive_rows[item_id]["score"] - fixed_rows[item_id]["score"]
        per_question.append(
            {
                "id": item_id,
                "type": fixed_rows[item_id]["type"],
                "fixed": fixed_rows[item_id]["score"],
                "recursive": recursive_rows[item_id]["score"],
                "delta_recursive_minus_fixed": diff,
            }
        )
    return {
        "fixed_mean": fixed["mean"],
        "recursive_mean": recursive["mean"],
        "delta_recursive_minus_fixed": recursive["mean"] - fixed["mean"],
        "per_question": per_question,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", choices=VALID_STRATEGIES, default=["fixed", "recursive"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--auto-ingest", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    results = [run_strategy(strategy, k=args.k, auto_ingest=args.auto_ingest) for strategy in args.strategies]
    comparison = compare(results)
    payload = {"k": args.k, "strategies": results, "comparison": comparison}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if comparison:
        print_safe("\n=== COMPARISON ===")
        print_safe(f"fixed:     {comparison['fixed_mean']:.3f}")
        print_safe(f"recursive: {comparison['recursive_mean']:.3f}")
        print_safe(f"delta:     {comparison['delta_recursive_minus_fixed']:+.3f}")
        winner = "recursive" if comparison["delta_recursive_minus_fixed"] > 0 else "fixed"
        if comparison["delta_recursive_minus_fixed"] == 0:
            winner = "tie"
        print_safe(f"winner:    {winner}")

    print_safe(f"\nSaved detailed results to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
