"""Tests for the A3.2 prompt-class router.

Verifies that the heuristic classifier correctly distinguishes code, agentic,
and prose prompts from representative samples. CPU-only; no GPU or model load.

Run: cd sglang && pytest test/srt/speculative/test_prompt_class_router.py -v
"""

import pytest

from sglang.srt.speculative.prompt_class_router import (
    classify_for_routing,
    classify_prompt,
)


# Code-class samples — should be detected as "code"
CODE_SAMPLES = [
    "Write a Python function `def fibonacci(n)` that returns the n-th Fibonacci number.\n```python\ndef fib(n):\n    pass\n```",
    "Debug this Rust:\n```rust\nfn main() {\n    let x: i32 = 42;\n    println!(\"{}\", x);\n}\n```",
    "Run `npm install` then `npm run build` and tell me the output.",
    "Explain this SQL: SELECT * FROM users WHERE id = 1;",
    "I have an issue with import torch; the error is at https://github.com/pytorch/pytorch/issues/123",
]


# Agentic-class samples — should be detected as "agentic"
AGENTIC_SAMPLES = [
    "You are an agent that helps schedule meetings. Step 1: parse the user's request. Step 2: call the calendar tool. Step 3: confirm.",
    "Use the tool_call function with this JSON schema: {\"properties\": {\"name\": {\"type\": \"string\"}}, \"required\": [\"name\"]}",
    "Plan the steps to deploy a Kubernetes cluster. First, provision nodes. Second, install kubectl. Third, apply manifests.",
    "Your task is to invoke the search api endpoint and summarize results. The instruction is: query 'crucible eagle3'.",
    "<tool>web_search</tool>\n<thinking>I need to find recent papers on this.</thinking>\nGoal: 5 references.",
]


# Prose-class samples — should fall back to "prose"
PROSE_SAMPLES = [
    "Tell me a story about a dragon who learns to dance.",
    "What's the capital of France?",
    "I love walking my dog in the park on sunny mornings.",
    "Can you explain quantum entanglement in simple terms?",
    "Write a haiku about autumn leaves.",
]


@pytest.mark.parametrize("prompt", CODE_SAMPLES)
def test_classify_code(prompt):
    assert classify_prompt(prompt) == "code", f"expected code, got {classify_prompt(prompt)} for: {prompt[:60]}"


@pytest.mark.parametrize("prompt", AGENTIC_SAMPLES)
def test_classify_agentic(prompt):
    assert classify_prompt(prompt) == "agentic", f"expected agentic, got {classify_prompt(prompt)} for: {prompt[:60]}"


@pytest.mark.parametrize("prompt", PROSE_SAMPLES)
def test_classify_prose(prompt):
    assert classify_prompt(prompt) == "prose", f"expected prose, got {classify_prompt(prompt)} for: {prompt[:60]}"


def test_empty_prompt_is_prose():
    assert classify_prompt("") == "prose"
    assert classify_prompt(None or "") == "prose"


def test_classify_for_routing_returns_scores():
    cls, scores = classify_for_routing("def hello(): return 1")
    assert "code" in scores
    assert "agentic" in scores
    assert isinstance(scores["code"], int)


def test_truncation_does_not_break_classification():
    # 100K char prompt should still classify in <1 ms
    prompt = "Write a function in Python: " + "abc " * 25000
    cls = classify_prompt(prompt)
    assert cls in ("code", "agentic", "prose")
