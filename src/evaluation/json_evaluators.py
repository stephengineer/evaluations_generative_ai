"""JSON-focused evaluators for comparing structured agent output to a reference.

Provides two evaluators:
- ``json_structure_evaluator`` -- deterministic, recursive field-by-field comparison.
- ``json_llm_evaluator`` -- uses an LLM judge for semantic field-level matching.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from openevals.llm import create_llm_as_judge
from pydantic import BaseModel, Field

from src.common.logger import get_logger
from src.evaluation.prompts import JSON_CORRECTNESS_PROMPT

logger = get_logger("json_evaluators")


class FieldEvaluation(BaseModel):
    field_name: str = Field(description="The name of the field being evaluated.")
    score: int = Field(
        description="Binary score for the field: 1 if correct, 0 if incorrect."
    )
    reasoning: str = Field(description="Brief explanation for the score.")


class JsonEvaluationResponse(BaseModel):
    evaluations: List[FieldEvaluation] = Field(
        description="Evaluation of each field from the expected JSON."
    )
    overall_comment: str = Field(description="A summary of the semantic match.")


def extract_json_from_string(text: str) -> Union[Dict, List, None]:
    """
    Extracts the first valid JSON object or array from a string.
    Handles markdown code blocks and raw text.
    """
    if not isinstance(text, str):
        return None

    # Try to find JSON within code blocks first
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, text)
    if match:
        try:
            return cast(Union[Dict, List], json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass

    # Try to find the first outer-most JSON object or array
    # This is a simple heuristic: look for first '{' or '['
    stack: list[str] = []
    start_index = -1

    for i, char in enumerate(text):
        if char == "{" or char == "[":
            if not stack:
                start_index = i
            stack.append(char)
        elif char == "}" or char == "]":
            if not stack:
                continue

            last = stack[-1]
            if (char == "}" and last == "{") or (char == "]" and last == "["):
                stack.pop()
                if not stack:
                    # Found a potential complete JSON block
                    try:
                        potential_json = text[start_index : i + 1]
                        return cast(Union[Dict, List], json.loads(potential_json))
                    except json.JSONDecodeError:
                        # Keep trying if this one failed
                        pass
            else:
                pass

    # Fallback: try parsing the whole string if it looks like JSON
    try:
        return cast(Union[Dict, List], json.loads(text))
    except json.JSONDecodeError:
        return None


def _is_number(param: Any) -> bool:
    return isinstance(param, (int, float))


def _count_leaves(data: Any) -> float:
    """
    Counts the total number of checkable items (leaves) in the expected data structure.
    Used as the denominator for scoring.
    """
    if isinstance(data, dict):
        if not data:
            return 1.0
        return sum(_count_leaves(v) for v in data.values())
    elif isinstance(data, list):
        if not data:
            return 1.0
        return sum(_count_leaves(item) for item in data)
    else:
        # Primitives count as 1 point
        return 1.0


def _compare_lists_order_independent(
    actual: List, expected: List, path: str
) -> Tuple[float, List[str]]:
    """
    Compares two lists order-independently.
    Returns (matched_points, error_messages).
    """
    mismatches = []
    total_matched = 0.0

    # Create copies to avoid modifying originals
    actual_copy = list(actual)
    expected_copy = list(expected)

    # Try to find a match for each expected item
    for exp_item in expected_copy:
        found_match = False
        for i, act_item in enumerate(actual_copy):
            # Check if this pair matches deep
            # We only care if it's a "perfect match" for this specific expected item to count it
            sub_matches, sub_errors = _compare_values(act_item, exp_item, path)
            if not sub_errors:
                # specific match found
                total_matched += sub_matches  # Add the points from the child
                actual_copy.pop(i)
                found_match = True
                break

        if not found_match:
            # If we didn't find a match, we don't add points.
            # We should probably report it as missing.
            mismatches.append(f"{path}: Missing expected item: {exp_item}")

    if actual_copy:
        mismatches.append(f"{path}: Unexpected items found in list: {actual_copy}")

    return total_matched, mismatches


def _compare_values(
    actual: Any, expected: Any, path: str = ""
) -> Tuple[float, List[str]]:
    """
    Recursive comparison function.
    Returns (matched_points, list of error strings).
    """
    errors = []

    # Handle null/missing equivalence logic
    if expected is None:
        if actual is not None:
            errors.append(
                f"{path}: Expected None/null, got {type(actual).__name__}: {actual}"
            )
            return 0.0, errors
        else:
            return 1.0, []

    # Check for type mismatch, allowing int/float interoperability
    if _is_number(expected) and _is_number(actual):
        if abs(expected - actual) > 1e-9:  # Default float tolerance
            errors.append(f"{path}: Value mismatch. Expected {expected}, got {actual}")
            return 0.0, errors
        else:
            return 1.0, []

    if actual is None:
        errors.append(f"{path}: Expected {type(expected).__name__}, got None")
        return 0.0, errors

    if not isinstance(actual, type(expected)):
        errors.append(
            f"{path}: Type mismatch. Expected {type(expected).__name__}, got {type(actual).__name__}"
        )
        return 0.0, errors

    # Deep compare Dicts
    if isinstance(expected, dict):
        current_matches = 0.0
        # If expected dict is empty, we just expect actual to be a dict (checked above)
        if not expected:
            return 1.0, []

        for key, exp_val in expected.items():
            new_path = f"{path}.{key}" if path else key

            # Null handling: "null means we can expect it to be missing as well"
            if exp_val is None:
                if key in actual and actual[key] is not None:
                    sub_m, sub_err = _compare_values(actual[key], exp_val, new_path)
                    current_matches += sub_m
                    errors.extend(sub_err)
                else:
                    # implicitly matched (missing or None)
                    current_matches += 1.0
            else:
                if key not in actual:
                    errors.append(f"{new_path}: Missing required key")
                    # No points added
                else:
                    sub_m, sub_err = _compare_values(actual[key], exp_val, new_path)
                    current_matches += sub_m
                    errors.extend(sub_err)

        return current_matches, errors

    # Deep compare Lists (Order Independent)
    if isinstance(expected, list):
        if not expected:
            # Expect empty list
            if actual:
                errors.append(f"{path}: Expected empty list, got {actual}")
                return 0.0, errors
            else:
                return 1.0, []
        return _compare_lists_order_independent(actual, expected, path)

    # Primitive equality
    if actual != expected:
        errors.append(f"{path}: Value mismatch. Expected '{expected}', got '{actual}'")
        return 0.0, errors

    return 1.0, []


def json_structure_evaluator(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> dict:
    """
    Evaluator that checks if the agent's output contains a JSON object matching the reference output.
    Returns a score between 0.0 and 1.0 based on percentage of matches.
    """
    # 1. Get Actual JSON from Agent Output
    # 'answer' or 'output' or whatever key the chain returns.
    # Usually standard standard conversational agents return a string in 'output' or 'messages'.
    # Adjust based on your specific chain output format.
    agent_output_text = outputs.get("answer", outputs.get("output", ""))
    actual_json: Optional[Union[Dict, List]] = None
    if not isinstance(agent_output_text, str):
        # Try to stringify if it's not a string, or maybe it's already a dict?
        if isinstance(agent_output_text, dict):
            actual_json = agent_output_text
        else:
            agent_output_text = str(agent_output_text)
            actual_json = extract_json_from_string(agent_output_text)
    else:
        actual_json = extract_json_from_string(agent_output_text)

    if actual_json is None:
        return {
            "key": "json_structure_match",
            "score": 0.0,
            "comment": "Failed to extract valid JSON from agent output.",
        }

    # 2. Get Expected JSON
    # reference_outputs usually contains the specific dataset output columns
    # We assume 'expected_output' or similar key maps to the dict we want.
    # If the dataset just has keys in root, we might compare the whole reference_outputs?
    # Based on packageDataset.json, expected_output is a nested object.

    # Prefer 'fields_to_compare' as the ground truth (a curated subset of fields per test case).
    # Fall back to 'expected_output', then to the raw reference_outputs dict.
    if "fields_to_compare" in reference_outputs:
        expected_json = reference_outputs["fields_to_compare"]
    elif "expected_output" in reference_outputs:
        expected_json = reference_outputs["expected_output"]
    else:
        expected_json = reference_outputs

    # Un-nest 'answer' or 'output' if it's the only key in expected_json and matches agent output type
    # This handles cases where dataset expected output is wrapped in {"answer": ...} but agent returns direct object
    if isinstance(expected_json, dict) and len(expected_json) == 1:
        single_key = next(iter(expected_json))
        # If the single key is remarkably specific like "answer" or "output"
        if single_key in ["answer", "output", "result"] and isinstance(
            expected_json[single_key], (dict, list)
        ):
            # If actual_json does NOT have this key at root, we assume expected_json is wrapped and actual isn't
            if isinstance(actual_json, dict) and single_key not in actual_json:
                expected_json = expected_json[single_key]

    # 3. Calculate Score
    total_possible = _count_leaves(expected_json)
    if total_possible == 0:
        return {
            "key": "json_structure_match",
            "score": 1.0,
            "comment": "No expectation to match against.",
        }

    matched_points, errors = _compare_values(actual_json, expected_json, path="root")

    score = matched_points / total_possible
    score = max(0.0, min(1.0, score))

    if not errors:
        return {
            "key": "json_structure_match",
            "score": score,
            "comment": "Perfect JSON match",
        }
    else:
        # Show specific mismatches
        error_msg = f"Score: {score:.2f}. Mismatches: {'; '.join(errors[:5])}" + (
            "..." if len(errors) > 5 else ""
        )
        return {"key": "json_structure_match", "score": score, "comment": error_msg}


def json_llm_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    judge_model: str = "gpt-5-mini",
    judge: Optional[Any] = None,
) -> dict:
    """
    Evaluator that uses an LLM as a judge to evaluate the semantic correctness
    of individual fields in the JSON output.
    """
    logger.debug("Running json_llm_evaluator with judge_model: %s", judge_model)
    # 1. Get Actual JSON from Agent Output
    agent_output_text = outputs.get("answer", outputs.get("output", ""))
    actual_json: Optional[Union[Dict, List]] = None
    if not isinstance(agent_output_text, str):
        if isinstance(agent_output_text, dict):
            actual_json = agent_output_text
        else:
            agent_output_text = str(agent_output_text)
            actual_json = extract_json_from_string(agent_output_text)
    else:
        actual_json = extract_json_from_string(agent_output_text)

    if actual_json is None:
        return {
            "key": "json_llm_match",
            "score": 0.0,
            "comment": "Failed to extract valid JSON from agent output.",
        }

    # 2. Get Ground Truth (fields_to_compare)
    if "fields_to_compare" in reference_outputs:
        expected_json = reference_outputs["fields_to_compare"]
    elif "expected_output" in reference_outputs:
        expected_json = reference_outputs["expected_output"]
    else:
        expected_json = reference_outputs

    logger.debug("Expected (ground truth): %s", json.dumps(expected_json, indent=2))
    logger.debug("Actual (agent output): %s", json.dumps(actual_json, indent=2))

    # 3. Call LLM as a Judge with structured output
    evaluator = create_llm_as_judge(
        prompt=JSON_CORRECTNESS_PROMPT,
        judge=judge,
        model=judge_model if judge is None else None,
        output_schema=JsonEvaluationResponse,
        feedback_key="json_llm_match",
    )

    try:
        # raw_result will be an instance of JsonEvaluationResponse due to response_model
        raw_result = evaluator(
            inputs=str(inputs),
            outputs=json.dumps(actual_json),
            reference_outputs=json.dumps(expected_json),
        )

        # Handling case where openevals might return a dict instead of the model instance
        if isinstance(raw_result, dict):
            res_data = JsonEvaluationResponse(**raw_result)
        else:
            res_data = cast(JsonEvaluationResponse, raw_result)

        evaluations = res_data.evaluations
        for eval_data in evaluations:
            logger.debug(
                "Field %s: score=%d, reasoning=%s",
                eval_data.field_name,
                eval_data.score,
                eval_data.reasoning,
            )

        if not evaluations:
            return {
                "key": "json_llm_match",
                "score": 1.0,
                "comment": "No fields to evaluate.",
            }

        # Calculate score as average of binary field scores
        total_score = sum(e.score for e in evaluations)
        total_fields = len(evaluations)
        avg_score = total_score / total_fields

        mismatches = [e.field_name for e in evaluations if e.score == 0]
        reasons = [
            f"{e.field_name}: {e.reasoning}" for e in evaluations if e.score == 0
        ]

        logger.debug("Computed average score: %.2f", avg_score)

        comment = f"Avg Score: {avg_score:.2f}. "
        if mismatches:
            comment += f"Mismatches in: {', '.join(mismatches)}. "
            comment += " | ".join(reasons[:3])
        else:
            comment += "All fields matched semantically."

        return {
            "key": "json_llm_match",
            "score": float(avg_score),
            "comment": comment,
        }

    except Exception as e:
        logger.error("LLM evaluation error: %s", e)
        return {
            "key": "json_llm_match",
            "score": 0.0,
            "comment": f"Error during LLM evaluation: {str(e)}",
        }
