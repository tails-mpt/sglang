"""
Crucible Squeeze Track-A A3.2 — per-prompt-class tree-config routing (scaffold).

This module provides a rule-based prompt classifier and a per-class config
loader. Used when `server_args.speculative_routing_config` is set.

The actual hook into eagle_worker.forward_batch_speculative_generation is
deferred — this module currently provides the data layer (config schema,
classifier, lookup) but is NOT YET wired into request handling.

TODO: in eagle_worker_v2.py, before tree construction, call
`classify_prompt(req.prompt)` to get the class, then look up
the per-class config and override `num_steps / eagle_topk / num_draft_tokens`
for that request. Need to also handle the batched case where different
requests in the same batch may have different classes — either fall back
to the global config for batched requests, or per-request tree shape (much
more invasive).

A3.2 spec at:
crucible:experiments/MiniMax-M2.5/squeeze/lossless/A3-internal-extensions/
A3-internal-extensions.md
"""
import json
import re
from typing import Dict, Optional


# Rule-based classifier keyed on simple prompt features.
# Tuned for the four squeeze-pipeline benchmark datasets.
_PATTERNS = [
    # HumanEval: code completion prompts typically include "def " or
    # "function " or have triple-quoted docstrings as first lines.
    ("humaneval", re.compile(r"\b(def\s+\w+|class\s+\w+|function\s+\w+|\"\"\"[\s\S]+\"\"\")", re.MULTILINE)),
    # SWE-Bench: "fix the issue" / "patch" / commit-message style.
    ("swebench_verified", re.compile(r"\b(fix\s+(this\s+)?(bug|issue)|patch|repository|commit|pull\s+request)\b", re.IGNORECASE)),
    # Terminal-Bench: long-horizon agentic prompts; bash-flavored.
    ("terminal_bench", re.compile(r"\$\s+\w+|`[a-z]+\s+(--?\w+)?[^`]*`|terminal|bash\s+command", re.IGNORECASE)),
    # MT-Bench: open-ended chat / reasoning. Default fall-through.
]


def classify_prompt(prompt: str) -> str:
    """Classify a prompt into one of {humaneval, swebench_verified, terminal_bench, mt_bench}.

    Rule-based — no ML cost. The four classes match the four squeeze
    benchmark datasets so the classifier integrates cleanly with the
    bench harness for A/B comparison.
    """
    if prompt is None:
        return "mt_bench"
    for cls, pat in _PATTERNS:
        if pat.search(prompt):
            return cls
    return "mt_bench"  # default


def load_routing_config(path: Optional[str]) -> Dict[str, Dict[str, int]]:
    """Load per-class routing config JSON.

    Schema:
        {
          "humaneval": {"num_steps": 5, "eagle_topk": 1, "num_draft_tokens": 6},
          "mt_bench":  {"num_steps": 3, "eagle_topk": 4, "num_draft_tokens": 8},
          ...
        }

    Returns empty dict if path is None or file is missing. Caller is
    expected to fall back to the global config for unmapped classes.
    """
    if not path:
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    # Lightly validate schema
    valid: Dict[str, Dict[str, int]] = {}
    for cls, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        try:
            valid[cls] = {
                "num_steps": int(cfg["num_steps"]),
                "eagle_topk": int(cfg["eagle_topk"]),
                "num_draft_tokens": int(cfg["num_draft_tokens"]),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return valid


def lookup_config(
    config: Dict[str, Dict[str, int]],
    prompt: str,
    fallback_num_steps: int,
    fallback_eagle_topk: int,
    fallback_num_draft_tokens: int,
) -> Dict[str, int]:
    """Look up the routing config for a prompt's class, with fallback to globals."""
    cls = classify_prompt(prompt)
    if cls in config:
        return config[cls]
    return {
        "num_steps": fallback_num_steps,
        "eagle_topk": fallback_eagle_topk,
        "num_draft_tokens": fallback_num_draft_tokens,
    }
