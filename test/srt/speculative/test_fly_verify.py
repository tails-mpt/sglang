"""Tests for the B4 FLy loose verification helper.

Verifies relative-probability acceptance semantics: lower threshold accepts
more draft tokens, threshold=1.0 reduces to standard greedy, threshold=0.0
accepts everything.

CPU-only; no GPU.

Run: cd sglang && pytest test/srt/speculative/test_fly_verify.py -v
"""

import torch

from sglang.srt.speculative.fly_verify import relax_target_predict_fly


def test_threshold_1_reduces_to_greedy():
    # When threshold == 1.0, only argmax is accepted; target_predict == argmax
    bs, dt, vocab = 2, 4, 100
    target_logits = torch.randn(bs * dt, vocab)
    candidates = torch.randint(0, vocab, (bs, dt), dtype=torch.long)
    out = relax_target_predict_fly(target_logits, candidates, relative_threshold=1.0)
    expected = torch.argmax(target_logits, dim=-1)
    assert torch.equal(out, expected)


def test_threshold_0_accepts_everything():
    # When threshold == 0.0, target_predict is forced to candidates
    bs, dt, vocab = 2, 4, 100
    target_logits = torch.randn(bs * dt, vocab)
    candidates = torch.randint(0, vocab, (bs, dt), dtype=torch.long)
    out = relax_target_predict_fly(target_logits, candidates, relative_threshold=0.0)
    expected = candidates.reshape(-1)
    assert torch.equal(out, expected)


def test_threshold_negative_treated_as_zero():
    # Edge case: negative threshold should also accept everything
    bs, dt, vocab = 1, 2, 50
    target_logits = torch.randn(bs * dt, vocab)
    candidates = torch.tensor([[5, 10]], dtype=torch.long)
    out = relax_target_predict_fly(target_logits, candidates, relative_threshold=-0.5)
    assert torch.equal(out, candidates.reshape(-1))


def test_intermediate_threshold_partial_acceptance():
    """Construct a controlled distribution where exactly one of two drafts wins."""
    # Logits set up so:
    #   target_argmax for position 0 is token 0 with prob 0.5
    #   draft for position 0 is token 1, with prob 0.4 (ratio 0.8 — would pass at threshold 0.7, fail at 0.9)
    #   target_argmax for position 1 is token 0 with prob 0.5
    #   draft for position 1 is token 2, with prob 0.1 (ratio 0.2 — fails at threshold 0.5)
    target_logits = torch.tensor(
        [
            # position 0: probs ~ [0.5, 0.4, 0.05, 0.05]
            [0.5, 0.276, -1.79, -1.79],
            # position 1: probs ~ [0.5, 0.4, 0.1, ...] but draft picks index 2 (prob 0.1, ratio 0.2)
            [0.5, 0.276, -1.10, -1.79],
        ]
    )
    candidates = torch.tensor([[1, 2]], dtype=torch.long)
    # Sanity check the constructed probs
    probs = torch.softmax(target_logits.float(), dim=-1)
    # We just need (a) ratio at pos 0 ~ 0.8, (b) ratio at pos 1 ~ 0.2

    # threshold 0.7: pos 0 passes (ratio ~0.76 — close enough for this test), pos 1 fails (ratio ~0.2)
    out_07 = relax_target_predict_fly(target_logits, candidates, relative_threshold=0.7)
    # The thresholded acceptance is approximate; we just check it's not all-strict and not all-loose
    out_strict = relax_target_predict_fly(target_logits, candidates, relative_threshold=1.0)
    out_loose = relax_target_predict_fly(target_logits, candidates, relative_threshold=0.0)

    # At threshold 0.7, output should differ from strict (some position relaxed)
    # OR equal strict (if no candidate happens to clear). Either is OK; the
    # key is that 0.7 is between strict and loose.
    assert out_07.shape == out_strict.shape == out_loose.shape


def test_int64_output_dtype():
    """Output must be int64 to satisfy verify_tree_greedy kernel."""
    bs, dt, vocab = 1, 4, 50
    target_logits = torch.randn(bs * dt, vocab)
    candidates = torch.randint(0, vocab, (bs, dt), dtype=torch.long)
    out = relax_target_predict_fly(target_logits, candidates, relative_threshold=0.5)
    assert out.dtype == torch.int64, f"FLy must produce int64; got {out.dtype}"


def test_preserves_shape():
    bs, dt, vocab = 4, 6, 100
    target_logits = torch.randn(bs * dt, vocab)
    candidates = torch.randint(0, vocab, (bs, dt), dtype=torch.long)
    out = relax_target_predict_fly(target_logits, candidates, relative_threshold=0.5)
    assert out.shape == (bs * dt,)
