from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from prompts import (
    ASPECTS_SYSTEM,
    DISCOVERY_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    MAP_SYSTEM,
    REDUCE_SYSTEM,
    REDUCE_SYSTEM_STRICT,
)
from schema import (
    AspectDiscoveryReport,
    JudgeReport,
    MapChunkSummary,
    RawReview,
    ReviewAspects,
    ReviewExtraction,
    ReviewSummary,
)


HOMEWORK_DIR = Path(__file__).resolve().parent
SEMINAR_DIR = HOMEWORK_DIR.parent
STARTER_DIR = SEMINAR_DIR / "starter"
ASPECT_ORDER = ["performance", "design", "support", "price", "ads", "reliability"]


@dataclass
class QuoteRecord:
    review_id: str
    quote: str
    stage: str


@dataclass
class UsageAggregate:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


@dataclass
class CostReport:
    baseline_input_cost_usd: float
    actual_input_cost_usd: float
    output_cost_usd: float
    total_actual_cost_usd: float
    savings_usd: float
    cache_hit_rate: float


def configure_llm_environment() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(SEMINAR_DIR / ".env")
        load_dotenv(HOMEWORK_DIR / ".env")
    except ImportError:
        pass

    if os.getenv("OPENROUTER_BASE_URL") and not os.getenv("LLM_BASE_URL"):
        os.environ["LLM_BASE_URL"] = os.environ["OPENROUTER_BASE_URL"]
    if os.getenv("OPENROUTER_MODEL") and not os.getenv("LLM_MODEL"):
        os.environ["LLM_MODEL"] = os.environ["OPENROUTER_MODEL"]
    if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]


def get_llm_client() -> tuple[Any, str]:
    configure_llm_environment()
    if str(STARTER_DIR) not in sys.path:
        sys.path.insert(0, str(STARTER_DIR))
    from llm_client import get_model, make_client

    return make_client(), get_model()


def usage_to_dict(usage: Any) -> dict[str, int]:
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cache_hit_tokens = int(
        getattr(usage, "prompt_cache_hit_tokens", None)
        or getattr(usage, "cached_tokens", 0)
        or 0
    )
    cache_miss_tokens = int(
        getattr(usage, "prompt_cache_miss_tokens", prompt_tokens - cache_hit_tokens) or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
    }


def accumulate_usage(total: UsageAggregate, usage: Any) -> None:
    stats = usage_to_dict(usage)
    total.prompt_tokens += stats["prompt_tokens"]
    total.completion_tokens += stats["completion_tokens"]
    total.cache_hit_tokens += stats["cache_hit_tokens"]
    total.cache_miss_tokens += stats["cache_miss_tokens"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc
    return rows


def load_reviews(path: Path) -> tuple[list[RawReview], list[dict[str, Any]]]:
    valid: list[RawReview] = []
    invalid: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        try:
            valid.append(RawReview.model_validate(row))
        except ValidationError as exc:
            invalid.append({"review_id": row.get("review_id"), "error": str(exc)})
    return valid, invalid


def chunked(items: Iterable[Any], size: int) -> list[list[Any]]:
    values = list(items)
    return [values[index : index + size] for index in range(0, len(values), size)]


def make_completion(
    client: Any,
    *,
    model: str,
    response_model: Any,
    messages: list[dict[str, str]],
) -> tuple[Any, Any]:
    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=messages,
    )


def extract_reviews(
    client: Any,
    model: str,
    reviews: list[RawReview],
    usage: UsageAggregate,
) -> list[ReviewExtraction]:
    result: list[ReviewExtraction] = []
    for review in reviews:
        payload = (
            f"review_id: {review.review_id}\n"
            f"app_name: {review.app_name}\n"
            f"source: {review.source}\n"
            f"rating: {review.rating}\n"
            f"review_date: {review.review_date}\n\n"
            f"Текст отзыва:\n{review.text}"
        )
        extracted, completion = make_completion(
            client,
            model=model,
            response_model=ReviewExtraction,
            messages=[
                {"role": "system", "content": IE_SYSTEM},
                {"role": "user", "content": payload},
            ],
        )
        if extracted.review_id != review.review_id:
            extracted.review_id = review.review_id
        result.append(extracted)
        accumulate_usage(usage, completion.usage)
    return result


def extract_aspects(
    client: Any,
    model: str,
    reviews: list[RawReview],
    usage: UsageAggregate,
) -> list[ReviewAspects]:
    result: list[ReviewAspects] = []
    for review in reviews:
        aspects, completion = make_completion(
            client,
            model=model,
            response_model=ReviewAspects,
            messages=[
                {"role": "system", "content": ASPECTS_SYSTEM},
                {"role": "user", "content": f"review_id: {review.review_id}\n\n{review.text}"},
            ],
        )
        if aspects.review_id != review.review_id:
            aspects.review_id = review.review_id
        result.append(aspects)
        accumulate_usage(usage, completion.usage)
    return result


def discover_aspects(
    client: Any,
    model: str,
    reviews: list[RawReview],
    usage: UsageAggregate,
) -> AspectDiscoveryReport:
    packet = [
        {"review_id": review.review_id, "rating": review.rating, "text": review.text}
        for review in reviews
    ]
    discovery, completion = make_completion(
        client,
        model=model,
        response_model=AspectDiscoveryReport,
        messages=[
            {"role": "system", "content": DISCOVERY_SYSTEM},
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ],
    )
    accumulate_usage(usage, completion.usage)
    return discovery


def map_reduce_summary(
    client: Any,
    model: str,
    extractions: list[ReviewExtraction],
    aspects: list[ReviewAspects],
    usage: UsageAggregate,
    strict: bool = False,
) -> tuple[ReviewSummary, list[MapChunkSummary]]:
    aspects_by_id = {item.review_id: item for item in aspects}
    map_summaries: list[MapChunkSummary] = []
    for index, chunk in enumerate(chunked(extractions, 8), 1):
        packet = []
        for item in chunk:
            packet.append(
                {
                    "review_id": item.review_id,
                    "sentiment": item.sentiment,
                    "issues": [issue.model_dump() for issue in item.issues],
                    "aspects": [
                        aspect.model_dump()
                        for aspect in aspects_by_id.get(
                            item.review_id, ReviewAspects(review_id=item.review_id)
                        ).aspects
                    ],
                }
            )
        mapped, completion = make_completion(
            client,
            model=model,
            response_model=MapChunkSummary,
            messages=[
                {"role": "system", "content": MAP_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps({"chunk_id": f"chunk_{index}", "items": packet}, ensure_ascii=False),
                },
            ],
        )
        if not mapped.chunk_id:
            mapped.chunk_id = f"chunk_{index}"
        map_summaries.append(mapped)
        accumulate_usage(usage, completion.usage)

    summary, completion = make_completion(
        client,
        model=model,
        response_model=ReviewSummary,
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM_STRICT if strict else REDUCE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {"chunks": [item.model_dump() for item in map_summaries]},
                    ensure_ascii=False,
                ),
            },
        ],
    )
    accumulate_usage(usage, completion.usage)
    return summary, map_summaries


def run_judge(
    client: Any,
    model: str,
    extractions: list[ReviewExtraction],
    aspects: list[ReviewAspects],
    summary: ReviewSummary,
    usage: UsageAggregate,
) -> JudgeReport:
    evidence = []
    for extraction in extractions:
        for issue in extraction.issues:
            evidence.append(
                {
                    "review_id": extraction.review_id,
                    "category": issue.category,
                    "severity": issue.severity,
                    "quote": issue.quote,
                }
            )
    for review_aspects in aspects:
        for aspect in review_aspects.aspects:
            evidence.append(
                {
                    "review_id": review_aspects.review_id,
                    "category": aspect.aspect,
                    "sentiment": aspect.sentiment,
                    "quote": aspect.quote,
                }
            )
    report, completion = make_completion(
        client,
        model=model,
        response_model=JudgeReport,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {"action_items": summary.action_items, "evidence": evidence},
                    ensure_ascii=False,
                ),
            },
        ],
    )
    accumulate_usage(usage, completion.usage)
    return report


def collect_quotes(
    extractions: list[ReviewExtraction],
    aspects: list[ReviewAspects],
    summary: ReviewSummary,
) -> list[QuoteRecord]:
    quotes: list[QuoteRecord] = []
    for extraction in extractions:
        for issue in extraction.issues:
            quotes.append(QuoteRecord(extraction.review_id, issue.quote, "ie"))
    for review_aspects in aspects:
        for aspect in review_aspects.aspects:
            quotes.append(QuoteRecord(review_aspects.review_id, aspect.quote, "aspects"))
    for quote in summary.evidence_quotes:
        quotes.append(QuoteRecord("__summary__", quote, "summary"))
    return quotes


def check_quotes_equivalent(
    quotes: list[QuoteRecord],
    source_texts: dict[str, str],
) -> list[QuoteRecord]:
    full_corpus = "\n".join(source_texts.values()).lower()
    ghosts: list[QuoteRecord] = []
    for quote in quotes:
        probe = quote.quote.strip().lower()[:30]
        if not probe:
            continue
        if quote.review_id == "__summary__":
            if probe not in full_corpus:
                ghosts.append(quote)
            continue
        if probe not in source_texts.get(quote.review_id, "").lower():
            ghosts.append(quote)
    return ghosts


def build_heatmap(aspects: list[ReviewAspects], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sent_to_num = {"positive": 1, "neutral": 0, "negative": -1}
    names = [item.review_id for item in aspects]
    matrix = np.full((len(names), len(ASPECT_ORDER)), np.nan)
    for row_index, review in enumerate(aspects):
        for item in review.aspects:
            if item.aspect in ASPECT_ORDER:
                matrix[row_index, ASPECT_ORDER.index(item.aspect)] = sent_to_num[item.sentiment]
    plt.figure(figsize=(10, max(5, len(names) * 0.25)))
    sns.heatmap(
        matrix,
        xticklabels=ASPECT_ORDER,
        yticklabels=names,
        center=0,
        cmap="coolwarm",
        cbar_kws={"label": "Sentiment"},
    )
    plt.title("Aspect sentiment by review")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def compute_cost_report(usage: UsageAggregate) -> CostReport:
    prompt_price = float(os.getenv("PRICE_PROMPT_PER_1M", "0.27"))
    cache_hit_price = float(os.getenv("PRICE_CACHE_HIT_PER_1M", "0.07"))
    completion_price = float(os.getenv("PRICE_COMPLETION_PER_1M", "1.10"))
    baseline_input_cost = usage.prompt_tokens / 1_000_000 * prompt_price
    actual_input_cost = (
        usage.cache_miss_tokens / 1_000_000 * prompt_price
        + usage.cache_hit_tokens / 1_000_000 * cache_hit_price
    )
    output_cost = usage.completion_tokens / 1_000_000 * completion_price
    return CostReport(
        baseline_input_cost_usd=baseline_input_cost,
        actual_input_cost_usd=actual_input_cost,
        output_cost_usd=output_cost,
        total_actual_cost_usd=actual_input_cost + output_cost,
        savings_usd=baseline_input_cost - actual_input_cost,
        cache_hit_rate=usage.cache_hit_tokens / usage.prompt_tokens if usage.prompt_tokens else 0.0,
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def analyze(input_path: str, out_dir: str = "output") -> dict[str, Any]:
    started = time.time()
    input_file = Path(input_path)
    if not input_file.is_absolute():
        input_file = HOMEWORK_DIR / input_file
    out = Path(out_dir)
    if not out.is_absolute():
        out = HOMEWORK_DIR / out
    out.mkdir(parents=True, exist_ok=True)

    reviews, invalid_rows = load_reviews(input_file)
    if not reviews:
        raise ValueError("No valid reviews loaded from input.")

    client, model = get_llm_client()
    usage = UsageAggregate()

    extractions = extract_reviews(client, model, reviews, usage)
    aspects = extract_aspects(client, model, reviews, usage)
    discovery = discover_aspects(client, model, reviews, usage)
    summary, map_summaries = map_reduce_summary(client, model, extractions, aspects, usage)
    judge_report = run_judge(client, model, extractions, aspects, summary, usage)
    reran_reduce = False
    if judge_report.overall_score < 0.7:
        summary, map_summaries = map_reduce_summary(
            client, model, extractions, aspects, usage, strict=True
        )
        judge_report = run_judge(client, model, extractions, aspects, summary, usage)
        reran_reduce = True

    source_texts = {review.review_id: review.text for review in reviews}
    quote_records = collect_quotes(extractions, aspects, summary)
    ghosts = check_quotes_equivalent(quote_records, source_texts)
    ghost_rate = len(ghosts) / len(quote_records) if quote_records else 0.0
    build_heatmap(aspects, out / "heatmap.png")
    cost = compute_cost_report(usage)

    write_json(out / "reviews.json", [item.model_dump() for item in extractions])
    write_json(out / "aspects.json", [item.model_dump() for item in aspects])
    write_json(out / "aspect_discovery.json", discovery.model_dump())
    write_json(out / "map_summaries.json", [item.model_dump() for item in map_summaries])
    write_json(out / "summary.json", summary.model_dump())
    write_json(out / "judge_report.json", judge_report.model_dump())
    write_json(out / "ghost_quotes.json", [asdict(item) for item in ghosts])

    report = {
        "model": model,
        "input_count": len(reviews) + len(invalid_rows),
        "valid_count": len(reviews),
        "validation_errors": invalid_rows,
        "validation_error_count": len(invalid_rows),
        "quote_count": len(quote_records),
        "ghost_quotes_count": len(ghosts),
        "ghost_quote_rate": ghost_rate,
        "judge_overall_score": judge_report.overall_score,
        "reran_reduce_after_low_judge_score": reran_reduce,
        "usage": asdict(usage),
        "cost_report": asdict(cost),
        "elapsed_sec": time.time() - started,
    }
    write_json(out / "run_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seminar 3 homework pipeline")
    parser.add_argument("input_path", nargs="?", default="input/reviews.jsonl")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    result = analyze(args.input_path, out_dir=args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
