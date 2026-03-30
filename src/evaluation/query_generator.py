"""
LLM-based query generator for AI provider evaluation.

Takes a scenario description and expected output from a LangSmith dataset
example, then generates realistic user queries that a real business owner
would type to accomplish the scenario through the SUT AI assistant.

Supports different user styles: brief (few words), normal, or verbose (a lot).
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI
from dotenv import load_dotenv

from src.common.logger import setup_logging
from src.evaluation.prompts.query_generation import (
    QUERY_GENERATION_SYSTEM_PROMPT,
    QUERY_GENERATION_USER_PROMPT,
    USER_STYLE_INSTRUCTIONS,
)

# Supported styles for simulating different types of users (few words vs a lot).
UserStyle = Literal["brief", "normal", "verbose"]
DEFAULT_USER_STYLE: UserStyle = "brief"

load_dotenv(override=True)

logger = setup_logging("query_generator")


@dataclass
class GeneratedQuery:
    """Result from the query generator."""

    query: str
    is_done: bool
    reasoning: str


class QueryGenerator:
    """Generates realistic user queries from scenario descriptions using LLM.

    For each turn in an AI provider evaluation, the generator:
    1. Reads the scenario (natural-language description of what the user wants)
    2. Reads the reference output (expected final result)
    3. Reviews the conversation history so far
    4. Produces the next natural user message (or signals completion)
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        provider: str = "openai",
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        if provider == "azure_openai":
            from src.common.azure_openai import create_azure_openai_client

            self.client, self.model = create_azure_openai_client(model)
        else:
            self.client = OpenAI()
            self.model = model

    @staticmethod
    def _format_history(conversation_history: list[dict[str, str]]) -> str:
        """Format conversation history into a readable string."""
        if not conversation_history:
            return "(No messages yet — this is the first turn.)"

        lines: list[str] = []
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            label = "User" if role == "user" else "Assistant"
            lines.append(f"[{label}]: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_reference(reference_output: dict[str, Any] | str) -> str:
        """Format the reference output for the prompt.

        Handles both string (CSV answer) and dict/JSON (JSON dataset) formats.
        """
        if isinstance(reference_output, str):
            return reference_output

        # For dict outputs, prefer the full answer, fall back to whole dict
        answer = reference_output.get("answer", "")
        if answer:
            # Try to parse stringified JSON for readability
            if isinstance(answer, str):
                try:
                    parsed = json.loads(answer)
                    return json.dumps(parsed, indent=2)
                except (json.JSONDecodeError, TypeError):
                    return answer
            return json.dumps(answer, indent=2)
        return json.dumps(reference_output, indent=2)

    def generate_next_query(
        self,
        scenario: str,
        reference_output: dict[str, Any],
        conversation_history: list[dict[str, str]],
        turn_number: int,
        max_turns: int,
        user_style: UserStyle = DEFAULT_USER_STYLE,
    ) -> GeneratedQuery:
        """Generate the next user query for an AI provider conversation.

        Args:
            scenario: Natural-language description of what the user wants
                (the ``input_query`` / ``question`` from the dataset).
            reference_output: Expected final output from the dataset example's
                ``outputs`` dict (used to guide what info to communicate).
            conversation_history: List of ``{"role": ..., "content": ...}``
                dicts representing the conversation so far.
            turn_number: Current turn (1-indexed).
            max_turns: Maximum allowed turns.
            user_style: Simulated user verbosity — "brief" (few words), "normal",
                or "verbose" (a lot). Affects message length and detail.

        Returns:
            A ``GeneratedQuery`` with the next user message, a done flag, and
            reasoning.
        """
        style_instruction = USER_STYLE_INSTRUCTIONS.get(
            user_style, USER_STYLE_INSTRUCTIONS["brief"]
        )
        system_content = QUERY_GENERATION_SYSTEM_PROMPT.format()
        user_prompt = QUERY_GENERATION_USER_PROMPT.format(
            scenario=scenario,
            reference_output=self._format_reference(reference_output),
            conversation_history=self._format_history(conversation_history),
            user_style_instruction=style_instruction,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_prompt},
                ],
                # TODO: Uncomment this when the model supports temperature
                # temperature=self.temperature,
                # max_completion_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            result = json.loads(content)

            return GeneratedQuery(
                query=result.get("query", ""),
                is_done=bool(result.get("is_done", False)),
                reasoning=result.get("reasoning", ""),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse query generator response: %s", e)
            # If we can't parse, signal done to avoid infinite loop
            return GeneratedQuery(
                query="",
                is_done=True,
                reasoning=f"Parse error — stopping: {e}",
            )
        except Exception as e:
            logger.error("Query generation failed: %s", e)
            return GeneratedQuery(
                query="",
                is_done=True,
                reasoning=f"Generation error — stopping: {e}",
            )
