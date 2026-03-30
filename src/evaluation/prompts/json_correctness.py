"""Prompt template for LLM-based semantic evaluation of JSON field correctness."""

JSON_CORRECTNESS_PROMPT = """\
Evaluate the correctness of individual fields in a JSON object produced by an AI agent compared to a given ground truth (fields to compare).

**Your Task:**
For each field specified in the "Expected JSON", determine if the "Actual JSON" contains a value that is semantically and technically correct according to the "User Query" and "Expected JSON".

**Scoring Rules:**
1. Binary per field: Provide a score of 1 (correct) or 0 (incorrect) for each field in the Expected JSON.
2. Semantic / soft fields (names, descriptions): Score 1 if the agent's value is a reasonable and professional interpretation of the request, even if not an exact string match.
3. Strict / technical fields (IDs, dates, numbers, booleans): Score 1 only if the value matches exactly or is functionally identical (e.g. "Percent" vs "percentage", 20 vs 20.0).
4. For list fields, compare items; score 1 only if the overall intent of the list matches (same IDs, types, amounts).

User Query: {inputs}
Expected JSON (Ground Truth): {reference_outputs}
Actual JSON (Agent Output): {outputs}
"""
