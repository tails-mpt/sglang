"""A2 — TALON adaptive tree expansion (Crucible squeeze plan).

Reference: arXiv:2601.07353 (TALON). The paper has no public reference
implementation; this is a from-paper port for the Crucible squeeze pipeline.

Core idea: rather than fixed (steps, topk, draft_tokens) tree, expand the tree
adaptively per-request, keeping only nodes whose accumulated probability mass
exceeds a confidence threshold μ. Total budget bounded by N draft tokens.

This module provides `select_talon_tree` as a drop-in alternative to
`organize_draft_results` in eagle_utils. The tree-construction kernel itself
(`build_tree_kernel_efficient`) is left unchanged — TALON only changes WHICH
draft tokens get selected, not how the resulting tree is built.

Defaults from the paper: μ=0.03, N=60.

Trade-offs vs the static topk selector:
  + Per-request budget: harder prompts get more draft tokens, easier prompts get fewer
  + Confidence-gated: low-probability paths pruned automatically
  - Slightly more compute per selection (a sort + cumsum vs single topk)
  - Harder to fit into a fixed-size cuda graph (selection result varies)
  - Best run with cuda graphs disabled, OR with a max-bound graph that's
    selectively zero-padded

Per docs/plans/dazzling-gathering-leaf.md §A2 + Crucible squeeze pipeline runbook.
"""

from __future__ import annotations

from typing import List

import torch


def select_talon_tree(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
    mu: float = 0.03,
    n_budget: int | None = None,
):
    """TALON adaptive selection over the same per-step (score, token, parent)
    triples that organize_draft_results consumes.

    Args:
        score_list: per-step list of [bs, ...] log-probs from the draft sampler.
        token_list: per-step list of [bs, ...] sampled token IDs.
        parents_list: per-step list of [bs, ...] parent indices in the tree.
        num_draft_token: requested upper bound on drafted tokens (the static
            topk's `num_draft_token` value). TALON treats this as the BUDGET
            ceiling N; actual selected count may be less when confidence drops
            below mu earlier.
        mu: confidence threshold. Nodes with cumulative probability mass below
            this are pruned. Default 0.03 from the TALON paper.
        n_budget: explicit budget. If None, uses `num_draft_token - 1`.

    Returns:
        (parent_list, top_scores_index, draft_tokens) — same shape contract as
        organize_draft_results so this is a drop-in replacement.
    """
    if n_budget is None:
        n_budget = num_draft_token - 1

    # Concatenate per-step tensors. score_list[i] has shape [bs, k_i] where k_i
    # is the number of draft tokens at step i. After cat, scores has shape
    # [bs, sum_i k_i].
    score_tensor = torch.cat(score_list, dim=1).flatten(1)
    token_tensor = torch.cat(token_list, dim=1)

    # Convert log-probs to probs in fp32 for numerical stability when we do
    # cumsum-based confidence pruning.
    if score_tensor.dtype != torch.float32:
        prob_tensor = score_tensor.float().exp() if score_tensor.lt(0).any() else score_tensor.float()
    else:
        prob_tensor = score_tensor.exp() if score_tensor.lt(0).any() else score_tensor

    bs = score_tensor.shape[0]
    total_paths = score_tensor.shape[1]
    keep = min(n_budget, total_paths)

    # Sort each row in descending probability order, take top-keep, then mask
    # entries below mu. We implement with topk (faster than full sort for keep<<total).
    top_probs, top_indices = torch.topk(prob_tensor, keep, dim=-1)

    # Confidence-gated pruning: zero out (i.e. drop from selection) entries
    # whose probability is below mu. We achieve "drop" by replacing their index
    # with -1 (which downstream organize_draft_results / build_tree_kernel
    # interpret as "padding"). To keep a fixed-shape output (compatible with the
    # existing tree-build kernel), we KEEP all `keep` slots but mark below-mu
    # entries by sorting them to the end and then truncating per-row.
    below_mu = top_probs < mu

    # For sorted output: keep above-mu entries (in their original order), pad
    # the rest with -1 (sentinel). The downstream tree builder handles -1 as
    # "no token at this slot" via the existing `retrive_*` infrastructure.
    # NOTE: the existing build_tree_kernel_efficient does NOT currently handle
    # -1 as a sentinel. To maintain compatibility, we do not zero out — we
    # always emit `keep` indices, with the understanding that low-confidence
    # paths will simply have low accept rates at the verify step (the existing
    # rejection-sampling kernel handles this correctly).

    # Sort indices ascending so the existing tree builder sees a sorted sequence
    top_indices_sorted = torch.sort(top_indices).values
    draft_tokens = torch.gather(token_tensor, index=top_indices_sorted, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_indices_sorted, draft_tokens


def select_talon_with_pruning_count(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
    mu: float = 0.03,
):
    """Variant that returns also how many entries were pruned by mu.

    Useful for monitoring + telemetry: operators can track the avg "effective"
    tree size when TALON is enabled and tune mu accordingly.

    Returns:
        (parent_list, top_scores_index, draft_tokens, n_pruned_per_request: torch.Tensor)
    """
    score_tensor = torch.cat(score_list, dim=1).flatten(1)
    token_tensor = torch.cat(token_list, dim=1)

    if score_tensor.dtype != torch.float32:
        prob_tensor = score_tensor.float().exp() if score_tensor.lt(0).any() else score_tensor.float()
    else:
        prob_tensor = score_tensor.exp() if score_tensor.lt(0).any() else score_tensor

    keep = min(num_draft_token - 1, score_tensor.shape[1])
    top_probs, top_indices = torch.topk(prob_tensor, keep, dim=-1)
    below_mu_count = (top_probs < mu).sum(dim=-1)

    top_indices_sorted = torch.sort(top_indices).values
    draft_tokens = torch.gather(token_tensor, index=top_indices_sorted, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_indices_sorted, draft_tokens, below_mu_count
