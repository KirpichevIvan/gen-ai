from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ToolName = Literal["local_rag", "stackoverflow_search"]
QuestionType = Literal["local_only", "stackoverflow", "mixed", "out_of_scope"]
AnswerStatus = Literal["answered", "insufficient"]
JudgeVerdict = Literal["supported", "partially_supported", "unsupported", "abstain_ok"]


class Evidence(BaseModel):
    source: str
    quote: str = Field(..., min_length=8)
    url: str | None = None

    @field_validator("quote")
    @classmethod
    def quote_is_short(cls, value: str) -> str:
        words = value.split()
        if len(words) > 80:
            raise ValueError("quote must stay short enough to verify manually")
        return value


class AssistantAnswer(BaseModel):
    status: AnswerStatus = "answered"
    answer: str = Field(..., min_length=20)
    route: list[ToolName]
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("route")
    @classmethod
    def route_is_not_empty(cls, value: list[ToolName]) -> list[ToolName]:
        if not value:
            raise ValueError("route must contain at least one tool")
        return value

    @model_validator(mode="after")
    def status_confidence_invariant(self) -> "AssistantAnswer":
        if self.status == "answered":
            if self.confidence < 0.45:
                raise ValueError("answered status requires confidence >= 0.45")
            if not self.evidence:
                raise ValueError("answered status requires verifiable evidence")
        if self.status == "insufficient" and self.confidence > 0.35:
            raise ValueError("insufficient status requires confidence <= 0.35")
        return self


class JudgeDecision(BaseModel):
    verdict: JudgeVerdict
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    unsupported_claims: list[str] = Field(default_factory=list)


class SubQuestion(BaseModel):
    id: int = Field(..., ge=1)
    question: str
    tool: ToolName
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    reasoning: str
    subquestions: list[SubQuestion]


class WorkerResult(BaseModel):
    subquestion_id: int
    tool: ToolName
    answer: str
    evidence: list[Evidence]
    raw: dict[str, Any] = Field(default_factory=dict)


class CriticReport(BaseModel):
    ok: bool
    ghost_quotes: int = Field(..., ge=0)
    unsupported_numbers: int = Field(default=0, ge=0)
    missing_expected_tools: list[ToolName] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    id: str
    type: QuestionType
    question: str
    expected_route: list[ToolName]
    gold_sources: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    expect_abstain: bool = False


class EvalRow(BaseModel):
    id: str
    type: QuestionType
    question: str
    expected_route: list[ToolName]
    actual_route: list[ToolName]
    route_ok: bool
    keyword_hits: int
    keyword_total: int
    source_hits: int
    source_total: int
    passed: bool
    steps: int
    ghost_quotes: int
    unsupported_numbers: int
    judge_verdict: JudgeVerdict
    status: AnswerStatus
    token_estimate: int
    cost_usd_estimate: float
    hit_at_5: float
    search_calls: int
    latency_sec: float
    so_mode: str | None = None
    answer: str
