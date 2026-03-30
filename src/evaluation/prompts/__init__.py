"""Prompt templates used by evaluators and the query generator."""

__all__ = [
    "ANSWER_CORRECTNESS_PROMPT",
    "JSON_CORRECTNESS_PROMPT",
    "PAIRWISE_RESPONSE_QUALITY_PROMPT",
    "QUERY_GENERATION_SYSTEM_PROMPT",
    "QUERY_GENERATION_USER_PROMPT",
]

from .answer_correctness import ANSWER_CORRECTNESS_PROMPT
from .json_correctness import JSON_CORRECTNESS_PROMPT
from .query_generation import (
    QUERY_GENERATION_SYSTEM_PROMPT,
    QUERY_GENERATION_USER_PROMPT,
)
from .response_quality import PAIRWISE_RESPONSE_QUALITY_PROMPT
