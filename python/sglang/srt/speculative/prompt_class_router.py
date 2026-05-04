"""A3.2 — per-prompt-class tree config router (Crucible squeeze plan).

Lightweight prompt-class classifier intended to drive runtime tree-config
selection. Detects three classes:
  * "code" — programming/dev prompts (function defs, code fences, file paths,
    technical CS terms)
  * "agentic" — tool-use / multi-step reasoning prompts (function calls,
    JSON schemas, agent instructions)
  * "prose" — everything else (free-form writing, QA, dialogue)

The classifier itself is a fast n-gram + keyword heuristic suitable for
per-request classification on the request manager's hot path (sub-millisecond
per request). For production routing the operator can either:

  (a) Log the class label and use it for offline analysis / metrics
  (b) Run multiple sglang servers with different tree configs (one per class)
      behind a thin HTTP router that dispatches by class
  (c) Once a graphless fast path lands in eagle_worker, dispatch within the
      same server by class (out of scope for this patch — see A3.1 monitor
      for the same per-request-mutation issue with cuda graphs)

This module ships (a). (b) and (c) are operator concerns / future work.

Per docs/plans/dazzling-gathering-leaf.md §A3.2 + Crucible squeeze pipeline runbook.
"""

from __future__ import annotations

import re
from typing import Literal

PromptClass = Literal["code", "agentic", "prose"]


# Code-class indicators
_CODE_KEYWORDS = frozenset({
    "def ", "class ", "import ", "function ", "return ", "lambda ",
    "var ", "let ", "const ", "void ", "int ", "char ", "str ", "bool",
    "public ", "private ", "static ", "async ", "await ", "yield ",
    "for ", "while ", "if (", "else if", "elif ", "else:",
    "stdin", "stdout", "stderr", "argv", "argc",
    "github.com", "gitlab.com", "bitbucket.org",
    ".py:", ".rs:", ".go:", ".ts:", ".js:",
    "#include", "package main", "use std::", "fn main",
    "println!", "console.log", "fmt.Print",
    "select ", "insert ", "update ", "delete from", "create table", "drop table",
    "from ", " where ", "join ", "group by",
    "npm ", "pip ", "cargo ", " brew ",
    "debug this", "explain this", "fix this", "refactor",
    "stack trace", "traceback", "exception", "stack overflow",
    "compile", "runtime error", "syntax error",
    "```",  # markdown code fence
})

_CODE_PATTERNS = [
    re.compile(r"```\w*\n"),  # markdown code fence with language
    re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+\w", re.IGNORECASE),  # SQL
    re.compile(r"\b(npm|pip|cargo|go|brew|apt|yum)\s+(install|run|build|test)"),
    re.compile(r"^[\s]*[a-zA-Z_]\w*\([^)]*\)\s*[:{=]", re.MULTILINE),  # function-call/def line
]


# Agentic-class indicators (tool use, JSON schemas, multi-step reasoning)
_AGENTIC_KEYWORDS = frozenset({
    "tool_call", "function_call", "tool_use",
    "<tool>", "</tool>", "<function>", "</function>",
    "json schema", '"properties":', '"required":',
    "<thinking>", "</thinking>", "<reasoning>", "</reasoning>",
    "step 1:", "step 2:", "first,", "second,", "third,", "finally,",
    "agent:", " user:", " assistant:", "system:",
    "instruction:", " task:", "goal:", "subtask:",
    "you are an", "you should", "your task is", "your goal is",
    "your job is", "you must", "you are required",
    "invoke", "endpoint", "api call", " api ",
    "search api", "tool api", "function api",
    "query 'crucible", "query \"",
    "summarize results", "summarize the",
    "plan the steps", "plan the approach", "the steps to", "first, ", " second, ", " third, ",
})

_AGENTIC_PATTERNS = [
    re.compile(r"\{[^{}]*\"[a-z_]+\"\s*:\s*\"[^\"]+\"[^{}]*\}", re.MULTILINE),  # JSON-ish
    re.compile(r"\b(call|invoke|use|trigger)\s+(the\s+)?(tool|function|api|endpoint)"),
    re.compile(r"\bplan\s+(the\s+)?(steps?|approach|strategy)"),
]


def _count_keyword_hits(text: str, keywords: frozenset[str]) -> int:
    """Case-insensitive substring match count."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _count_pattern_hits(text: str, patterns: list) -> int:
    return sum(1 for p in patterns if p.search(text))


def classify_prompt(text: str, max_chars: int = 4096) -> PromptClass:
    """Classify a prompt into one of {code, agentic, prose}.

    Heuristic-based: counts keyword + regex pattern hits per class. Truncates
    to first `max_chars` to bound runtime cost (sub-ms per request).

    Args:
        text: The user prompt text. Should be the raw user message string,
            not the full chat-templated input.
        max_chars: Truncate input beyond this many chars before classifying.
            Default 4096; sufficient for 90+ percentile prompts to be classified
            from leading content.

    Returns:
        One of "code", "agentic", "prose". When no class dominates, falls
        back to "prose".
    """
    if not text:
        return "prose"
    if len(text) > max_chars:
        text = text[:max_chars]

    code_score = _count_keyword_hits(text, _CODE_KEYWORDS) + 2 * _count_pattern_hits(text, _CODE_PATTERNS)
    agentic_score = _count_keyword_hits(text, _AGENTIC_KEYWORDS) + 2 * _count_pattern_hits(text, _AGENTIC_PATTERNS)

    # Strong signals get class assignment; ties or low scores → prose
    if code_score >= 2 and code_score >= agentic_score:
        return "code"
    if agentic_score >= 2 and agentic_score > code_score:
        return "agentic"
    return "prose"


def classify_for_routing(text: str) -> tuple[PromptClass, dict[str, int]]:
    """Same as classify_prompt but also returns the score breakdown for logging.

    Returns:
        (class_name, {"code": int, "agentic": int})
    """
    if not text:
        return "prose", {"code": 0, "agentic": 0}
    if len(text) > 4096:
        text = text[:4096]
    code_score = _count_keyword_hits(text, _CODE_KEYWORDS) + 2 * _count_pattern_hits(text, _CODE_PATTERNS)
    agentic_score = _count_keyword_hits(text, _AGENTIC_KEYWORDS) + 2 * _count_pattern_hits(text, _AGENTIC_PATTERNS)

    if code_score >= 2 and code_score >= agentic_score:
        return "code", {"code": code_score, "agentic": agentic_score}
    if agentic_score >= 2 and agentic_score > code_score:
        return "agentic", {"code": code_score, "agentic": agentic_score}
    return "prose", {"code": code_score, "agentic": agentic_score}
