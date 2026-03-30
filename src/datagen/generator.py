"""
LLM-based benchmark test-case generator.

Uses Claude (via Azure Foundry or direct Anthropic API) to produce
evaluation dataset entries in the same JSON format as existing datasets.
"""

import json
import os
import re
from typing import Any, cast

from anthropic import Anthropic, AnthropicFoundry
from dotenv import load_dotenv

from src.common.logger import get_logger
from src.datagen.prompts import SYSTEM_PROMPT, GENERATION_PROMPT_TEMPLATE

load_dotenv(override=True)

logger = get_logger(__name__)


def _build_client(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Anthropic:
    """Build an Anthropic client, preferring Azure Foundry when configured."""
    foundry_key = api_key or os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
    foundry_url = base_url or os.environ.get("ANTHROPIC_FOUNDRY_BASE_URL")
    foundry_resource = os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE")

    if foundry_key and (foundry_url or foundry_resource):
        logger.info("Using AnthropicFoundry client (Azure)")
        kwargs: dict[str, Any] = {"api_key": foundry_key}
        if foundry_url:
            # Strip trailing path segments the SDK adds automatically
            # (users sometimes paste the full endpoint including /v1/messages)
            clean_url = re.sub(r"/v1(/messages)?/?$", "", foundry_url.rstrip("/"))
            if clean_url != foundry_url.rstrip("/"):
                logger.warning(
                    "Stripped '/v1/messages' from ANTHROPIC_FOUNDRY_BASE_URL "
                    "(SDK adds it automatically). Using: %s",
                    clean_url,
                )
            kwargs["base_url"] = clean_url
        elif foundry_resource:
            kwargs["resource"] = foundry_resource
        return AnthropicFoundry(**kwargs)  # type: ignore[return-value]

    direct_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if direct_key:
        logger.info("Using direct Anthropic client")
        return Anthropic(api_key=direct_key)

    raise RuntimeError(
        "No Anthropic credentials found. Set ANTHROPIC_FOUNDRY_API_KEY + "
        "ANTHROPIC_FOUNDRY_BASE_URL (Azure), or ANTHROPIC_API_KEY (direct)."
    )


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from LLM output, tolerating markdown fences."""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in LLM response")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return cast(list[dict[str, Any]], json.loads(candidate))

    return cast(list[dict[str, Any]], json.loads(text[start:]))


class BenchmarkGenerator:
    """Generates benchmark test cases using Claude."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        temperature: float = 0.8,
        max_tokens: int = 16384,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = _build_client(model, api_key=api_key, base_url=base_url)

    def generate_batch(
        self,
        *,
        schema: dict,
        business_data: dict[str, Any],
        reference_samples: list[dict],
        count: int,
        object_type: str,
        id_prefix: str,
        start_id: int,
    ) -> list[dict]:
        """Generate a single batch of test cases via one LLM call."""

        refs_for_prompt = reference_samples[:3]

        prompt = GENERATION_PROMPT_TEMPLATE.format(
            count=count,
            object_type=object_type,
            schema=json.dumps(schema, indent=2),
            business_data=json.dumps(business_data, indent=2),
            reference_examples=json.dumps(refs_for_prompt, indent=2),
            id_prefix=id_prefix,
            start_id=start_id,
        )

        logger.info(
            "Requesting %d samples (batch start_id=%03d, model=%s)",
            count,
            start_id,
            self.model,
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text  # type: ignore[union-attr]
        logger.debug("Raw LLM response length: %d chars", len(raw_text))

        samples = _extract_json_array(raw_text)
        logger.info("Parsed %d samples from LLM response", len(samples))
        return samples

    def generate(
        self,
        *,
        schema: dict,
        business_data: dict[str, Any],
        reference_samples: list[dict],
        total_count: int,
        batch_size: int,
        object_type: str,
        id_prefix: str,
        existing_count: int = 0,
    ) -> list[dict]:
        """Generate total_count samples in batches, assigning sequential IDs."""

        all_samples: list[dict] = []
        generated = 0
        current_id = existing_count + 1

        while generated < total_count:
            remaining = total_count - generated
            batch_count = min(batch_size, remaining)

            try:
                batch = self.generate_batch(
                    schema=schema,
                    business_data=business_data,
                    reference_samples=reference_samples,
                    count=batch_count,
                    object_type=object_type,
                    id_prefix=id_prefix,
                    start_id=current_id,
                )
            except Exception:
                logger.exception("Batch generation failed at id=%d", current_id)
                raise

            if not batch:
                raise ValueError(
                    "LLM returned an empty batch (no test cases). "
                    "This can happen when the model returns '[]'. Try again or adjust the prompt."
                )

            for sample in batch:
                sample["id"] = f"{id_prefix}_{current_id:03d}"
                current_id += 1

            all_samples.extend(batch)
            generated += len(batch)
            logger.info("Progress: %d / %d samples generated", generated, total_count)

        return all_samples[:total_count]
