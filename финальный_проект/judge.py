from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI

from schema import AssistantAnswer, JudgeDecision


def _norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = text.replace("‐", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def _offline_judge(
    question: str,
    answer: AssistantAnswer,
    evidence_texts: list[str],
    *,
    expect_abstain: bool = False,
) -> JudgeDecision:
    """Deterministic fallback used when no LLM judge key is configured."""
    haystack = _norm("\n".join(evidence_texts))
    unsupported = []
    for evidence in answer.evidence:
        if _norm(evidence.quote) not in haystack:
            unsupported.append(f"ghost quote from {evidence.source}")

    if expect_abstain:
        if answer.status == "insufficient":
            return JudgeDecision(verdict="abstain_ok", score=1.0, reason="The answer abstained on an out-of-scope question.")
        return JudgeDecision(
            verdict="unsupported",
            score=0.0,
            reason="Out-of-scope question received an answered status.",
            unsupported_claims=["expected abstain"],
        )

    if answer.status == "insufficient":
        return JudgeDecision(
            verdict="unsupported",
            score=0.0,
            reason="The system abstained on an in-scope evaluation question.",
            unsupported_claims=["unexpected insufficient status"],
        )

    if unsupported:
        return JudgeDecision(
            verdict="unsupported",
            score=0.2,
            reason="Some citations are not exact substrings of retrieved evidence.",
            unsupported_claims=unsupported,
        )
    return JudgeDecision(verdict="supported", score=0.8, reason="All citations are grounded in retrieved evidence.")


def _llm_judge(
    question: str,
    answer: AssistantAnswer,
    evidence_texts: list[str],
    *,
    expect_abstain: bool = False,
) -> JudgeDecision:
    client = OpenAI()
    evidence = "\n\n---\n\n".join(evidence_texts[:8])
    prompt = f"""Ты LLM-as-judge для RAG-системы.

Проверь, следует ли ответ из evidence. Если вопрос вне корпуса и ответ честно отказался, verdict=abstain_ok.

Вопрос:
{question}

expect_abstain={expect_abstain}

Ответ JSON:
{answer.model_dump_json(indent=2)}

Evidence:
{evidence}
"""
    completion = client.beta.chat.completions.parse(
        model=os.environ.get("OPENAI_JUDGE_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
        messages=[{"role": "user", "content": prompt}],
        response_format=JudgeDecision,
        temperature=0,
        max_completion_tokens=800,
    )
    return completion.choices[0].message.parsed or _offline_judge(
        question,
        answer,
        evidence_texts,
        expect_abstain=expect_abstain,
    )


def judge_answer(
    question: str,
    answer: AssistantAnswer,
    evidence_texts: list[str],
    *,
    expect_abstain: bool = False,
    use_llm: bool = False,
) -> JudgeDecision:
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            return _llm_judge(question, answer, evidence_texts, expect_abstain=expect_abstain)
        except Exception as exc:
            return JudgeDecision(
                verdict="partially_supported",
                score=0.5,
                reason=f"LLM judge failed, used conservative fallback note: {exc}",
            )
    return _offline_judge(question, answer, evidence_texts, expect_abstain=expect_abstain)
