"""Tests for the A2 TALON adaptive tree selector.

Verifies output-shape compatibility with organize_draft_results, plus
mu-pruning behavior. CPU-only; no GPU.

Run: cd sglang && pytest test/srt/speculative/test_talon_tree.py -v
"""

import torch

from sglang.srt.speculative.talon_tree import (
    select_talon_tree,
    select_talon_with_pruning_count,
)


def _mk_inputs(bs: int = 2, num_draft_token: int = 6, num_steps: int = 3):
    """Synthesize per-step (score, token, parent) triples for testing."""
    # Step i has shape [bs, topk^i] in real EAGLE; for tests we use a flat shape
    # where each step has [bs, num_draft_token] entries with descending scores.
    score_list = []
    token_list = []
    parents_list = []
    for step in range(num_steps):
        # Log-probs ranging from -0.5 (high prob) to -5.0 (low prob)
        log_p = torch.linspace(-0.5, -5.0, num_draft_token).expand(bs, -1).clone()
        score_list.append(log_p)
        # Token IDs incrementing per step
        toks = torch.arange(step * num_draft_token, (step + 1) * num_draft_token, dtype=torch.long).expand(bs, -1).clone()
        token_list.append(toks)
        # Parents — first step has 0, others have prev step indices
        if step == 0:
            parents = torch.zeros(bs, num_draft_token, dtype=torch.long)
        else:
            parents = torch.arange(num_draft_token, dtype=torch.long).expand(bs, -1).clone()
        parents_list.append(parents)
    return score_list, token_list, parents_list


def test_select_talon_tree_returns_correct_shapes():
    bs, ndt = 2, 6
    score_list, token_list, parents_list = _mk_inputs(bs=bs, num_draft_token=ndt)
    parent_list, top_indices, draft_tokens = select_talon_tree(
        score_list, token_list, parents_list, num_draft_token=ndt, mu=0.0
    )
    # top_indices shape [bs, num_draft_token - 1]
    assert top_indices.shape == (bs, ndt - 1)
    # draft_tokens shape matches top_indices
    assert draft_tokens.shape == top_indices.shape


def test_indices_are_sorted_ascending():
    """Match the contract of organize_draft_results — indices must be sorted."""
    score_list, token_list, parents_list = _mk_inputs()
    _, top_indices, _ = select_talon_tree(score_list, token_list, parents_list, num_draft_token=6, mu=0.0)
    for row in top_indices:
        assert torch.all(row[:-1] <= row[1:]), f"indices not sorted: {row.tolist()}"


def test_mu_zero_picks_top_k_by_probability():
    """With mu=0 (no pruning) TALON should match topk's selection set."""
    score_list, token_list, parents_list = _mk_inputs(bs=1, num_draft_token=6)
    _, talon_indices, _ = select_talon_tree(score_list, token_list, parents_list, num_draft_token=6, mu=0.0)
    # Reference: topk selection (same logic as organize_draft_results)
    score_cat = torch.cat(score_list, dim=1).flatten(1)
    ref_topk = torch.sort(torch.topk(score_cat, 5, dim=-1).indices).values
    assert torch.equal(talon_indices, ref_topk), f"TALON ({talon_indices.tolist()}) != topk ({ref_topk.tolist()})"


def test_pruning_count_increases_with_mu():
    """As mu increases, more entries get marked as pruned."""
    score_list, token_list, parents_list = _mk_inputs(bs=1, num_draft_token=6)
    _, _, _, prune_low = select_talon_with_pruning_count(score_list, token_list, parents_list, num_draft_token=6, mu=0.001)
    _, _, _, prune_high = select_talon_with_pruning_count(score_list, token_list, parents_list, num_draft_token=6, mu=0.5)
    assert prune_high.item() >= prune_low.item(), "higher mu should prune more"


def test_no_below_zero_log_probs_handled():
    """If raw scores are already probabilities (positive), TALON should still work."""
    bs, ndt = 1, 4
    # Provide positive probs (already exp'd)
    score_list = [torch.tensor([[0.5, 0.3, 0.1, 0.05]]) for _ in range(2)]
    token_list = [torch.tensor([[10, 11, 12, 13]], dtype=torch.long) for _ in range(2)]
    parents_list = [torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)]
    _, top_indices, draft_tokens = select_talon_tree(score_list, token_list, parents_list, num_draft_token=ndt, mu=0.0)
    # Should not crash; should select 3 entries
    assert top_indices.shape == (1, 3)
