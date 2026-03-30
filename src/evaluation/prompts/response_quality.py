"""Pairwise response quality evaluation prompt for comparing SUT vs baseline."""

PAIRWISE_RESPONSE_QUALITY_PROMPT = """
You are an objective, expert evaluator. You are scoring two AI-generated responses to a user's question. Score each response independently on three criteria.
IMPORTANT: You MUST always write your reasoning, scores, and all output in English, regardless of the language of the question or responses.

<User Profile>
1. Users are either solopreneurs or large, multi-location business owners.
2. They're busy and usually are working when they ask questions.
3. They want a response that is concise and easy to understand in their language.
4. They are not always tech-savvy.
</User Profile>

<Criteria>
**Relevance** — Does the response directly address the specific question?
- If the question references a specific device, feature, or scenario, the response must be specific to that.
- Penalize generic information, irrelevant article links, or unsolicited subscription cancellation mentions.
- Ignore auth token issues in the outputs.

**Completeness** — Does the response cover all necessary information to answer the question?
- If the question is ambiguous, asking for clarification counts as complete.
- Do not reward length for length's sake.

**Clarity** — Is the response concise, easy to understand for a non-tech-savvy user, and well-structured?
</Criteria>

<Scoring Rubric (apply to each criterion independently)>
- **1.0**: Excellent — fully satisfies the criterion.
- **0.75**: Good — mostly satisfies with minor gaps.
- **0.5**: Adequate — partially satisfies with notable gaps.
- **0.25**: Poor — barely satisfies, significant issues.
- **0.0**: Failing — does not satisfy the criterion at all.
</Scoring Rubric>

<Question>
{inputs}
</Question>

<Response A>
{response_a}
</Response A>

<Response B>
{response_b}
</Response B>

<Required Output Format>
You MUST respond with a JSON object using this exact structure and field names:
{{
  "response_a_relevance_reasoning": "[reasoning for Response A relevance score]",
  "response_a_relevance": [float: 0, 0.25, 0.5, 0.75, or 1.0],
  "response_a_completeness_reasoning": "[reasoning for Response A completeness score]",
  "response_a_completeness": [float: 0, 0.25, 0.5, 0.75, or 1.0],
  "response_a_clarity_reasoning": "[reasoning for Response A clarity score]",
  "response_a_clarity": [float: 0, 0.25, 0.5, 0.75, or 1.0],
  "response_b_relevance_reasoning": "[reasoning for Response B relevance score]",
  "response_b_relevance": [float: 0, 0.25, 0.5, 0.75, or 1.0],
  "response_b_completeness_reasoning": "[reasoning for Response B completeness score]",
  "response_b_completeness": [float: 0, 0.25, 0.5, 0.75, or 1.0],
  "response_b_clarity_reasoning": "[reasoning for Response B clarity score]",
  "response_b_clarity": [float: 0, 0.25, 0.5, 0.75, or 1.0]
}}
</Required Output Format>
"""
