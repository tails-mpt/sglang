"""B4 — FLy loose verification (Crucible squeeze plan).

Reference: arXiv:2511.22972 (FLy — A Loosely Speculative Decoding Framework).
The official reference is AMD ROCm + vLLM; this is a from-paper adaptation
for sglang that runs on the existing CUDA verify-tree-greedy kernel + a
PyTorch-level relative-probability check.

Core idea: relax the strict-rejection acceptance criterion. Instead of
accepting a draft token iff `target_argmax == draft_token` (greedy) or
`p_target(draft_token) >= threshold` (typical-acceptance fixed threshold),
FLy accepts when:

  p_target(draft_token) / p_target(argmax) >= relative_threshold

This is RELATIVE-threshold acceptance: a draft token doesn't need to win
outright, it just needs to be "close enough" to the winner. With
relative_threshold=1.0 we get back to strict greedy; with 0.5 we accept
any draft that's at least half as probable as the argmax; with 0.0 we
accept everything.

Implementation in this module is a Python-level wrapper around the existing
`verify_tree_greedy` kernel:

  1. Compute target_argmax via torch.argmax (existing behavior)
  2. For each draft candidate, compute p_target(draft) / p_target(argmax)
  3. If >= threshold, OVERRIDE target_argmax with draft (so the kernel
     accepts it). Else keep target_argmax.
  4. Pass the (possibly overridden) target_predict into verify_tree_greedy.

This is approximate — a true FLy implementation would have a custom kernel
that does the relative-threshold check inside the tree-traversal logic. The
Python wrapper has slightly different semantics for chained tokens (we make
the override decision per-position independently) but produces similar
end-to-end behavior at temp=0.

Quality gate (per Crucible squeeze plan §B4): per-dataset score must stay
within 3% of strict-rejection baseline. relative_threshold values likely to
clear quality gate: 0.7-0.9.

Per docs/plans/dazzling-gathering-leaf.md §B4 + Crucible squeeze pipeline runbook.
"""

from __future__ import annotations

import torch


def relax_target_predict_fly(
    target_logits: torch.Tensor,
    candidates: torch.Tensor,
    relative_threshold: float = 0.8,
) -> torch.Tensor:
    """Compute target_predict with FLy loose acceptance.

    Args:
        target_logits: [bs * draft_token_num, vocab_size] — target's next-token
            logits per draft position.
        candidates: [bs, draft_token_num] (long) — sampled draft token IDs.
        relative_threshold: in [0, 1]. Higher = stricter (closer to standard greedy).
            A draft token is accepted iff
              softmax(target_logits)[draft_id] >= relative_threshold * softmax(target_logits)[argmax]

    Returns:
        target_predict: [bs * draft_token_num] (int64). For each position, this
        is either the argmax (when the draft fails the FLy check) or the draft
        token ID (when it passes). Downstream verify_tree_greedy then accepts
        when target_predict[i] == candidates_flat[i], so by overriding target_predict
        we make the kernel accept FLy-eligible drafts.
    """
    bs, draft_token_num = candidates.shape
    flat_candidates = candidates.reshape(-1)  # [bs * draft_token_num]

    # Always-accept path: relative_threshold <= 0 means accept everything
    if relative_threshold <= 0.0:
        return flat_candidates

    # Strict path: relative_threshold >= 1 means standard greedy
    target_argmax = torch.argmax(target_logits, dim=-1)
    if relative_threshold >= 1.0:
        return target_argmax

    # Relative path: compute softmax probs and relative ratio
    target_probs = torch.softmax(target_logits.float(), dim=-1)  # fp32 for stability
    argmax_probs = target_probs.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
    candidate_probs = target_probs.gather(-1, flat_candidates.unsqueeze(-1)).squeeze(-1)

    # Per-position relative ratio
    ratio = candidate_probs / argmax_probs.clamp_min(1e-12)
    fly_accept = ratio >= relative_threshold

    # Where FLy accepts, return the draft token (so the kernel sees a match)
    # Where FLy doesn't accept, return the argmax (standard behavior)
    target_predict = torch.where(fly_accept, flat_candidates, target_argmax)
    return target_predict


def fly_loose_verify(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_logits: torch.Tensor,
    relative_threshold: float = 0.8,
):
    """End-to-end FLy verify. Computes target_predict via relax_target_predict_fly,
    then delegates to the existing verify_tree_greedy kernel.

    All semantics match verify_tree_greedy_func; only the acceptance criterion
    is relaxed via the relative-threshold check.
    """
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func

    target_predict = relax_target_predict_fly(
        target_logits, candidates, relative_threshold=relative_threshold
    )
    target_predict = target_predict.reshape(candidates.shape)

    return verify_tree_greedy_func(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )
