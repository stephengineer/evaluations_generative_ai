from src.evaluation.json_evaluators import json_structure_evaluator


def test_json_structure_score_perfect():
    outputs = {"output": '{"a": 1, "b": 2}'}
    reference = {"expected_output": {"a": 1, "b": 2}}
    # Should be 1.0 (100%)
    result = json_structure_evaluator({}, outputs, reference)
    assert result["score"] == 1.0


def test_json_structure_score_zero():
    outputs = {"output": '{"a": 9, "b": 8}'}
    reference = {"expected_output": {"a": 1, "b": 2}}
    # Should be 0.0
    result = json_structure_evaluator({}, outputs, reference)
    assert result["score"] == 0.0


def test_json_structure_score_partial_dict_basic():
    # 5 keys, 3 match perfectly, 2 mismatch
    # Data:
    # Expected: {a:1, b:2, c:3, d:4, e:5}
    # Actual:   {a:1, b:2, c:3, d:99, e:99}

    reference = {"expected_output": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}}
    outputs = {"output": '{"a": 1, "b": 2, "c": 3, "d": 99, "e": 99}'}

    result = json_structure_evaluator({}, outputs, reference)
    expected = 3 / 5  # 0.6

    assert abs(result["score"] - expected) < 1e-6


def test_json_structure_score_list_simple():
    # 5 items, 3 match
    reference = {"expected_output": [1, 2, 3, 4, 5]}
    outputs = {"output": "[1, 2, 3, 99, 88]"}

    result = json_structure_evaluator({}, outputs, reference)
    expected = 3 / 5  # 0.6
    assert abs(result["score"] - expected) < 1e-6


def test_json_structure_score_nested_mixed():
    # Expected:
    # {
    #   "meta": "data",  (1 point)
    #   "items": [
    #      {"id": 1},    (1 point inside dict inside list)
    #      {"id": 2}     (1 point inside dict inside list)
    #   ]
    # }
    # Total points = 1 (meta) + 1 (id:1) + 1 (id:2) = 3

    # Actual:
    # {
    #   "meta": "data",  (Match)
    #   "items": [
    #      {"id": 1},    (Match)
    #      {"id": 99}    (Mismatch)
    #   ]
    # }
    # Matches = 1 (meta) + 1 (id:1) = 2
    # Score = 2/3 ≈ 0.666...

    reference = {"expected_output": {"meta": "data", "items": [{"id": 1}, {"id": 2}]}}
    outputs = {"output": '{"meta": "data", "items": [{"id": 1}, {"id": 99}]}'}

    result = json_structure_evaluator({}, outputs, reference)
    expected_score = 2 / 3
    assert abs(result["score"] - expected_score) < 1e-6
