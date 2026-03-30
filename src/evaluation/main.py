"""Entry point for the evaluation pipeline.

Supports two modes:
- **Standard evaluation**: sends each dataset example to the SUT once and
  evaluates the response.
- **AI provider evaluation**: uses an LLM query generator to simulate a
  multi-turn conversation with a fresh SUT per example before evaluating.

Configuration is loaded from a TOML file (see ``src.common.config``).
"""

import argparse
import random
from typing import cast

from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree, traceable

from src.common.config import EvaluationConfig, load_config
from src.common.logger import setup_logging
from src.evaluation.client import (
    BaseSUTClient,
    OpenSourceClient,
    ScriptSUTClient,
    ScriptSUTMCPClient,
)
from src.evaluation.evaluators import get_evaluators
from src.evaluation.query_generator import QueryGenerator, UserStyle

logger = setup_logging("runner")

load_dotenv(override=True)


def _build_client(eval_cfg: EvaluationConfig) -> BaseSUTClient:
    client_cfg = eval_cfg.client
    if client_cfg.type == "opensource":
        return OpenSourceClient(model_name=client_cfg.opensource_model)
    elif client_cfg.type == "script":
        return ScriptSUTClient(config=client_cfg)
    elif client_cfg.type == "mcp":
        return ScriptSUTMCPClient(config=client_cfg)
    else:
        raise ValueError(f"Unknown client type: {client_cfg.type}")


def _fetch_examples(langsmith_client: Client, dataset_name: str) -> list:
    examples = list(langsmith_client.list_examples(dataset_name=dataset_name))

    ids: list[int] = []
    has_invalid_ids = False
    for example in examples:
        metadata = example.metadata or {}
        value = metadata.get("id")
        if value is None:
            has_invalid_ids = True
            break
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            has_invalid_ids = True
            break

    if has_invalid_ids:
        logger.warning(
            "Dataset '%s' contains examples with missing or non-integer 'id' metadata; "
            "falling back to unsorted example order.",
            dataset_name,
        )
        return examples
    examples.sort(key=lambda e: int((e.metadata or {})["id"]))
    return examples


def _run_evaluation(sut_client: BaseSUTClient, eval_cfg: EvaluationConfig) -> None:
    langsmith_client = Client()

    @traceable(name="eval_target")
    def target(inputs: dict) -> dict:
        # Pass current run context so script client can attach SUT's traces (and token usage) to this run
        run_tree = get_current_run_tree()
        parent_headers = run_tree.to_headers() if run_tree else None
        if parent_headers is None:
            logger.debug(
                "get_current_run_tree() is None; SUT subprocess will not attach to this run (usage may not appear in experiment)."
            )
        return sut_client.predict(inputs, langsmith_parent_headers=parent_headers)  # type: ignore

    examples = _fetch_examples(langsmith_client, eval_cfg.dataset_name)

    # multi_round + max_concurrency > 1 can cause nondeterministic conversation order; enforce 1
    effective_concurrency = eval_cfg.max_concurrency
    if eval_cfg.client.mode == "multi_round" and eval_cfg.max_concurrency > 1:
        logger.warning(
            "multi_round mode: forcing max_concurrency=1 for stable evaluation (config had %s)",
            eval_cfg.max_concurrency,
        )
        effective_concurrency = 1

    logger.info(
        f"Starting evaluation on dataset: {eval_cfg.dataset_name} "
        f"({len(examples)} examples, mode={eval_cfg.client.mode})"
    )

    # Pass None when evaluators list is empty so get_evaluators uses all registered evaluators
    evaluator_names = eval_cfg.evaluators or None

    judge = None
    if eval_cfg.judge_provider == "azure_openai":
        from src.common.azure_openai import create_azure_chat_model

        judge = create_azure_chat_model(eval_cfg.judge_model)

    experiment_results = langsmith_client.evaluate(
        target,
        data=examples,
        evaluators=get_evaluators(
            eval_cfg.judge_model,
            evaluator_names,
            judge=judge,
        ),
        experiment_prefix=eval_cfg.experiment_prefix,
        max_concurrency=effective_concurrency,
    )

    logger.info("Evaluation completed!")
    logger.info(experiment_results)


def _run_ai_provider_evaluation(eval_cfg: EvaluationConfig) -> None:
    """Run AI provider evaluation.

    For each dataset example:
    1. Create a fresh SUT client (isolated conversation).
    2. Use LLM to generate realistic user queries from the scenario.
    3. Send each query to SUT, collecting the conversation.
    4. After all turns, evaluate the final output against reference.
    5. Log everything as one experiment run in LangSmith.
    """
    langsmith_client = Client()

    qg_model = eval_cfg.query_generator_model or eval_cfg.judge_model
    qg_provider = eval_cfg.query_generator_provider or eval_cfg.judge_provider
    query_gen = QueryGenerator(
        model=qg_model,
        temperature=eval_cfg.query_generator_temperature,
        provider=qg_provider,
    )
    max_turns = eval_cfg.max_turns

    # Empty-response template (must match client.py _EMPTY_RESPONSE shape)
    empty_response: dict = {
        "answer": "",
        "agent_name": "",
        "thinking": "",
        "report_agent": {},
    }

    def _resolve_user_style() -> UserStyle:
        raw = (eval_cfg.query_generator_user_style or "normal").strip().lower()
        if raw == "random":
            return cast(UserStyle, random.choice(["brief", "normal", "verbose"]))
        if raw in ("brief", "normal", "verbose"):
            return cast(UserStyle, raw)
        return "normal"

    @traceable(name="ai_provider_target")
    def target(inputs: dict) -> dict:
        # Reference output was injected into inputs for query generation
        reference = inputs.get("_reference_for_generation", {})
        scenario = inputs.get("question", "")
        user_style = _resolve_user_style()

        # Create a fresh client for this conversation (isolated from other examples)
        turn_client = _build_client(eval_cfg)

        # For ScriptSUTClient multi_round, start the persistent process
        if (
            isinstance(turn_client, ScriptSUTClient)
            and eval_cfg.client.mode == "multi_round"
        ):
            turn_client.start()

        run_tree = get_current_run_tree()
        parent_headers = run_tree.to_headers() if run_tree else None

        conversation_history: list[dict[str, str]] = []
        final_response: dict = dict(empty_response)

        try:
            for turn in range(1, max_turns + 1):
                gen_result = query_gen.generate_next_query(
                    scenario=scenario,
                    reference_output=reference,
                    conversation_history=conversation_history,
                    turn_number=turn,
                    max_turns=max_turns,
                    user_style=user_style,
                )

                if gen_result.is_done:
                    logger.debug(
                        "Turn %d: generator signalled done — %s",
                        turn,
                        gen_result.reasoning,
                    )
                    break

                if not gen_result.query:
                    logger.warning("Turn %d: empty query, stopping.", turn)
                    break

                if gen_result.reasoning:
                    logger.debug(
                        "Turn %d/%d reasoning: %s",
                        turn,
                        max_turns,
                        gen_result.reasoning,
                    )
                else:
                    logger.debug("Turn %d/%d: no reasoning", turn, max_turns)

                logger.debug(
                    "Turn %d/%d query: %s",
                    turn,
                    max_turns,
                    gen_result.query[:120],
                )

                response = turn_client.predict(
                    {"question": gen_result.query},
                    langsmith_parent_headers=parent_headers,
                )

                if response.get("answer", ""):
                    logger.debug(
                        "Turn %d/%d response: %s",
                        turn,
                        max_turns,
                        response.get("answer", ""),
                    )
                else:
                    logger.warning("Turn %d/%d response: <empty>", turn, max_turns)

                conversation_history.append(
                    {"role": "user", "content": gen_result.query}
                )
                conversation_history.append(
                    {"role": "assistant", "content": response.get("answer", "")}
                )

                final_response = response

            # Attach conversation metadata to the output
            final_response["_conversation_turns"] = len(conversation_history) // 2
            final_response["_conversation_history"] = conversation_history

        finally:
            if (
                isinstance(turn_client, ScriptSUTClient)
                and eval_cfg.client.mode == "multi_round"
            ):
                turn_client.stop()

        return final_response

    examples = _fetch_examples(langsmith_client, eval_cfg.dataset_name)

    # Inject reference outputs into inputs so the target can guide query generation.
    # Evaluators receive reference_outputs separately via example.outputs, so
    # the extra key does not affect them.
    for example in examples:
        if example.outputs:
            example.inputs["_reference_for_generation"] = example.outputs

    evaluator_names = eval_cfg.evaluators or None

    judge = None
    if eval_cfg.judge_provider == "azure_openai":
        from src.common.azure_openai import create_azure_chat_model

        judge = create_azure_chat_model(eval_cfg.judge_model)

    logger.info(
        "Starting AI provider evaluation on dataset: %s "
        "(%d examples, max_turns=%d, query_model=%s)",
        eval_cfg.dataset_name,
        len(examples),
        max_turns,
        qg_model,
    )

    experiment_results = langsmith_client.evaluate(
        target,
        data=examples,
        evaluators=get_evaluators(
            eval_cfg.judge_model,
            evaluator_names,
            judge=judge,
        ),
        experiment_prefix=eval_cfg.experiment_prefix,
        max_concurrency=1,  # Always sequential for AI provider conversations
    )

    logger.info("AI provider evaluation completed!")
    logger.info(experiment_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (default: config.toml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.evaluation

    if eval_cfg.ai_provider:
        # AI provider: the runner creates fresh clients per example internally
        _run_ai_provider_evaluation(eval_cfg)
    else:
        # Original single-turn evaluation
        sut_client = _build_client(eval_cfg)
        if eval_cfg.client.mode == "multi_round" and isinstance(
            sut_client, ScriptSUTClient
        ):
            with sut_client:
                _run_evaluation(sut_client, eval_cfg)
        else:
            _run_evaluation(sut_client, eval_cfg)


if __name__ == "__main__":
    main()
