"""Evaluator registry and built-in evaluators for the evaluation pipeline.

Each evaluator is a callable that receives ``(inputs, outputs, reference_outputs)``
and returns one or more LangSmith feedback dicts.  Evaluators are registered by
name and retrieved via :func:`get_evaluators`.
"""

import inspect
import random
from functools import partial
from typing import Any, Callable, Optional, cast

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from openevals.llm import create_llm_as_judge
from openevals.prompts import ANSWER_RELEVANCE_PROMPT
from src.evaluation.json_evaluators import json_llm_evaluator, json_structure_evaluator
from src.evaluation.prompts import (
    ANSWER_CORRECTNESS_PROMPT,
    PAIRWISE_RESPONSE_QUALITY_PROMPT,
)

DEFAULT_JUDGE_MODEL = "gpt-5-mini"

EvaluatorFunc = Callable[..., Any]

_REGISTRY: dict[str, EvaluatorFunc] = {}


class ResponseQualityResponse(BaseModel):
    response_a_relevance_reasoning: str = Field(
        description="Reasoning for Response A relevance"
    )
    response_a_relevance: float = Field(description="Response A relevance score (0-1)")
    response_a_completeness_reasoning: str = Field(
        description="Reasoning for Response A completeness"
    )
    response_a_completeness: float = Field(
        description="Response A completeness score (0-1)"
    )
    response_a_clarity_reasoning: str = Field(
        description="Reasoning for Response A clarity"
    )
    response_a_clarity: float = Field(description="Response A clarity score (0-1)")
    response_b_relevance_reasoning: str = Field(
        description="Reasoning for Response B relevance"
    )
    response_b_relevance: float = Field(description="Response B relevance score (0-1)")
    response_b_completeness_reasoning: str = Field(
        description="Reasoning for Response B completeness"
    )
    response_b_completeness: float = Field(
        description="Response B completeness score (0-1)"
    )
    response_b_clarity_reasoning: str = Field(
        description="Reasoning for Response B clarity"
    )
    response_b_clarity: float = Field(description="Response B clarity score (0-1)")


def register(name: str) -> Callable[[EvaluatorFunc], EvaluatorFunc]:
    def decorator(fn: EvaluatorFunc) -> EvaluatorFunc:
        _REGISTRY[name] = fn
        return fn

    return decorator


register("json_structure_evaluator")(json_structure_evaluator)
register("json_llm_match")(json_llm_evaluator)


def get_evaluators(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    evaluator_names: list[str] | None = None,
    judge: Optional[BaseChatModel] = None,
) -> list[EvaluatorFunc]:
    if evaluator_names is None:
        evaluator_names = list(_REGISTRY.keys())

    result: list[EvaluatorFunc] = []
    for name in evaluator_names:
        fn = _REGISTRY.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown evaluator '{name}'. Available: {list(_REGISTRY.keys())}"
            )
        sig = inspect.signature(fn).parameters
        kwargs: dict[str, Any] = {}
        if "judge_model" in sig:
            kwargs["judge_model"] = judge_model
        if "judge" in sig:
            kwargs["judge"] = judge
        if kwargs:
            result.append(partial(fn, **kwargs))
        else:
            result.append(fn)
    return result


@register("relevance")
def answer_relevance_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge: Optional[BaseChatModel] = None,
):
    evaluator = create_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT,
        judge=judge,
        model=judge_model if judge is None else None,
        feedback_key="relevance",
    )
    return evaluator(
        inputs=inputs,
        outputs=outputs.get("answer", ""),
        reference_outputs=reference_outputs,
    )


@register("correctness")
def answer_correctness_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge: Optional[BaseChatModel] = None,
):
    evaluator = create_llm_as_judge(
        prompt=ANSWER_CORRECTNESS_PROMPT,
        judge=judge,
        model=judge_model if judge is None else None,
        feedback_key="correctness",
    )

    return evaluator(
        inputs=inputs,
        outputs=outputs.get("answer", ""),
        reference_outputs=reference_outputs,
    )


@register("agent_match")
def agent_name_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """
    Evaluator to check if the predicted agent name matches the expected agent name.
    Expects the ground-truth agent in reference_outputs["agent"]; dataset sync
    (src/dataset/sync.py) places the CSV "agent" column in example outputs so it
    is available here when LangSmith passes example.outputs as reference_outputs.
    """
    # 1. Get predicted agent name (from model output)
    predicted_agent = outputs.get("agent_name", "").strip().lower()

    # 2. Get expected agent name (from reference_outputs, populated by sync from CSV "agent" column)
    expected_agent = reference_outputs.get("agent", "").strip().lower()

    # Logic:
    # - Case insensitive match
    # - If expected starts with "none" (e.g. none_sut_agent), predicted can be "orchestrator" or "none..."

    score = 0
    reason = f"Expected: '{expected_agent}', Got: '{predicted_agent}'"

    if expected_agent == predicted_agent:
        score = 1
    elif expected_agent.startswith("none"):
        # Special logic: if expected is none_sut_agent, accept "orchestrator" or "none..."
        if predicted_agent == "orchestrator" or predicted_agent.startswith("none"):
            score = 1
            reason += " (Matched via None/Orchestrator logic)"

    return {"key": "agent_match", "score": score, "comment": reason}


@register("report_agent_data")
def report_agent_data_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """
    Evaluator to check if the predicted report agent data matches the expected report agent data.
    """
    # 1. Get predicted report agent data with defensive type checks
    predicted_data = []

    report_agent = outputs.get("report_agent")
    if isinstance(report_agent, dict):
        data_result = report_agent.get("data_result")
        if isinstance(data_result, dict):
            result = data_result.get("result")
            if isinstance(result, dict):
                data = result.get("data", [])
                if isinstance(data, list):
                    # Only keep items that are dicts to avoid attribute errors below
                    predicted_data = [item for item in data if isinstance(item, dict)]
    # 2. Get expected report agent data
    reference_data = str(reference_outputs.get("answer", ""))

    # 3. Compare the predicted and expected report agent data
    match = next(
        (
            (key, value)
            for data in predicted_data
            for key, value in data.items()
            if reference_data == str(value)
        ),
        None,
    )

    if match:
        key, value = match
        score = 1
        reason = (
            f"Reference data {reference_data} found in predicted data {key}: {value}"
        )
    else:
        score = 0
        reason = f"Reference data {reference_data} not found in predicted data"

    return {"key": "report_agent_data_match", "score": score, "comment": reason}


@register("comparison")
def comparison_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge: Optional[BaseChatModel] = None,
):
    sut_answer = outputs.get("answer", "")

    # Short-circuit: don't penalize SUT for infrastructure errors (e.g. timeout)
    if isinstance(sut_answer, str) and sut_answer.startswith("Error: Timeout"):
        return [
            {
                "key": "sut_overall",
                "score": 0.5,
                "comment": f"Skipped — SUT error: {sut_answer}",
            },
            {
                "key": "baseline_overall",
                "score": 0.5,
                "comment": "N/A (SUT returned an error)",
            },
            {"key": "comparison_winner", "score": 0.5},
            {
                "key": "comparison_reasoning",
                "score": None,
                "comment": f"Tie (not scored) — SUT returned an error: {sut_answer}",
            },
        ]

    baseline_out_list = reference_outputs.get("baseline_output")
    baseline_answer = (
        baseline_out_list[0]
        if (isinstance(baseline_out_list, list) and baseline_out_list)
        else "No response"
    )

    question_raw = reference_outputs.get("question", inputs.get("question", ""))
    question = (
        question_raw[0]
        if isinstance(question_raw, list) and question_raw
        else question_raw
    )

    # Randomize position to eliminate order bias
    sut_is_a = random.random() < 0.5
    if sut_is_a:
        response_a, response_b = sut_answer, baseline_answer
    else:
        response_a, response_b = baseline_answer, sut_answer

    evaluator_runnable = create_llm_as_judge(
        prompt=PAIRWISE_RESPONSE_QUALITY_PROMPT,
        judge=judge,
        model=judge_model if judge is None else None,
        output_schema=ResponseQualityResponse,
        feedback_key="response_quality",
    )

    raw_result = evaluator_runnable(
        inputs=question,
        outputs=sut_answer,
        response_a=response_a,
        response_b=response_b,
    )
    # create_llm_as_judge returns the Pydantic model directly when output_schema is set
    raw_response: dict = (
        raw_result.model_dump()
        if isinstance(raw_result, ResponseQualityResponse)
        else cast(dict, raw_result)
    )

    # Map A/B scores back to sut/baseline
    if sut_is_a:
        sut_prefix, baseline_prefix = "response_a", "response_b"
    else:
        sut_prefix, baseline_prefix = "response_b", "response_a"

    sut_rel = raw_response[f"{sut_prefix}_relevance"]
    sut_comp = raw_response[f"{sut_prefix}_completeness"]
    sut_clar = raw_response[f"{sut_prefix}_clarity"]
    baseline_rel = raw_response[f"{baseline_prefix}_relevance"]
    baseline_comp = raw_response[f"{baseline_prefix}_completeness"]
    baseline_clar = raw_response[f"{baseline_prefix}_clarity"]

    # Extract per-criteria reasoning
    sut_rel_reason = raw_response[f"{sut_prefix}_relevance_reasoning"]
    sut_comp_reason = raw_response[f"{sut_prefix}_completeness_reasoning"]
    sut_clar_reason = raw_response[f"{sut_prefix}_clarity_reasoning"]
    baseline_rel_reason = raw_response[f"{baseline_prefix}_relevance_reasoning"]
    baseline_comp_reason = raw_response[f"{baseline_prefix}_completeness_reasoning"]
    baseline_clar_reason = raw_response[f"{baseline_prefix}_clarity_reasoning"]

    # Weighted overall: 50% relevance, 30% completeness, 20% clarity
    sut_overall = 0.5 * sut_rel + 0.3 * sut_comp + 0.2 * sut_clar
    baseline_overall = 0.5 * baseline_rel + 0.3 * baseline_comp + 0.2 * baseline_clar

    # Derive winner: 1.0=SUT wins, 0.5=tie, 0.0=Baseline wins
    if sut_overall > baseline_overall:
        winner = 1.0
    elif sut_overall < baseline_overall:
        winner = 0.0
    else:
        winner = 0.5

    winner_label = "SUT" if winner == 1.0 else ("Baseline" if winner == 0.0 else "Tie")

    return [
        {
            "key": "sut_overall",
            "score": round(sut_overall, 4),
            "comment": f"Relevance: {sut_rel_reason}\nCompleteness: {sut_comp_reason}\nClarity: {sut_clar_reason}",
        },
        {
            "key": "baseline_overall",
            "score": round(baseline_overall, 4),
            "comment": f"Relevance: {baseline_rel_reason}\nCompleteness: {baseline_comp_reason}\nClarity: {baseline_clar_reason}",
        },
        {"key": "comparison_winner", "score": winner},
        {
            "key": "comparison_reasoning",
            "score": None,
            "comment": (
                f"Winner: {winner_label} (SUT overall: {round(sut_overall, 4)} vs "
                f"Baseline overall: {round(baseline_overall, 4)})\n\n"
                f"* Relevance -- SUT {sut_rel} vs Baseline {baseline_rel}\n"
                f"  SUT: {sut_rel_reason}\n"
                f"  Baseline: {baseline_rel_reason}\n\n"
                f"* Completeness -- SUT {sut_comp} vs Baseline {baseline_comp}\n"
                f"  SUT: {sut_comp_reason}\n"
                f"  Baseline: {baseline_comp_reason}\n\n"
                f"* Clarity -- SUT {sut_clar} vs Baseline {baseline_clar}\n"
                f"  SUT: {sut_clar_reason}\n"
                f"  Baseline: {baseline_clar_reason}"
            ),
        },
    ]
