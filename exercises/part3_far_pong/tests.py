"""
Part 3: Frame-Autoregressive Pong — Tests
==========================================

Tests that validate student implementations against reference solutions.
Run with: python tests.py
"""

import sys
from pathlib import Path

exercises_dir = Path(__file__).parent.parent
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))

import torch
import torch as t
from part3_far_pong import solutions

device = solutions.device


def test_block_causal_mask(make_block_causal_mask):
    """Test block-causal mask correctness against reference."""
    mask = make_block_causal_mask(4, 9)
    ref = solutions.make_block_causal_mask(4, 9)
    assert mask.shape == ref.shape, f"Expected shape {ref.shape}, got {mask.shape}"
    assert mask.dtype == t.bool, f"Expected dtype bool, got {mask.dtype}"
    assert (mask == ref).all(), "Mask values don't match reference"
    # Spot checks: causality
    assert mask[0, 0] == False, "Frame 0 should attend to itself"
    assert mask[0, 9] == True, "Frame 0 should not attend to frame 1"
    assert mask[9, 0] == False, "Frame 1 should attend to frame 0"
    assert mask[9, 18] == True, "Frame 1 should not attend to frame 2"
    # Block structure: within a frame, all tokens attend to each other
    assert mask[0:9, 0:9].any() == False, "All tokens within frame 0 should attend to each other"
    assert mask[9:18, 9:18].any() == False, "All tokens within frame 1 should attend to each other"
    # Future frames are fully blocked
    assert mask[0:9, 9:18].all(), "Frame 0 tokens should NOT attend to any frame 1 token"
    # Different config
    mask2 = make_block_causal_mask(2, 4)
    assert mask2.shape == (8, 8), f"Expected (8, 8), got {mask2.shape}"
    assert mask2.dtype == t.bool, f"Expected dtype bool, got {mask2.dtype}"
    ref2 = solutions.make_block_causal_mask(2, 4)
    assert (mask2 == ref2).all(), "Mask values don't match reference for (2, 4)"
    print("  Exercise 1 (block-causal mask): OK")


def test_modulate(modulate_fn, gate_fn):
    """Test modulate() and gate_fn() produce correct values."""
    t.manual_seed(0)
    B, T, P, d = 2, 3, 5, 16
    S = T * P
    x = t.randn(B, S, d)
    shift = t.randn(B, T, d)
    scale = t.randn(B, T, d)

    # Compare against reference
    out = modulate_fn(x, shift, scale)
    ref = solutions.modulate(x, shift, scale)
    assert out.shape == ref.shape, f"modulate: expected {ref.shape}, got {out.shape}"
    assert t.allclose(out, ref, atol=1e-5), f"modulate values don't match reference (max diff: {(out - ref).abs().max():.2e})"

    gated = gate_fn(x, scale)
    ref_g = solutions.gate_fn(x, scale)
    assert t.allclose(gated, ref_g, atol=1e-5), f"gate_fn values don't match reference (max diff: {(gated - ref_g).abs().max():.2e})"

    # Single-frame case should match Part 2's [:, None, :] broadcast
    x1 = t.randn(B, P, d)
    shift1 = t.randn(B, 1, d)
    scale1 = t.randn(B, 1, d)
    out1 = modulate_fn(x1, shift1, scale1)
    expected = x1 * (1 + scale1) + shift1
    assert t.allclose(out1, expected, atol=1e-5), "Single-frame modulate should match Part 2 broadcasting"

    print("  Exercise 2 (modulate / gate_fn): OK")


def test_causal_dit_block_forward(CausalDiTBlock, make_block_causal_mask):
    """Test CausalDiTBlock.forward values against reference."""
    t.manual_seed(42)
    d, n_head, exp = 64, 4, 4
    n_frames, toks_per_frame = 3, 5

    block = CausalDiTBlock(d=d, n_head=n_head, exp=exp)
    ref_block = solutions.CausalDiTBlock(d=d, n_head=n_head, exp=exp)
    ref_block.load_state_dict(block.state_dict())

    x = t.randn(2, n_frames * toks_per_frame, d)
    cond = t.randn(2, n_frames, d)
    mask = make_block_causal_mask(n_frames, toks_per_frame)

    out = block(x, cond, mask=mask)
    ref = ref_block(x, cond, mask=mask)
    assert out.shape == ref.shape, f"Expected {ref.shape}, got {out.shape}"
    assert t.allclose(out, ref, atol=1e-4), f"Block output values don't match reference (max diff: {(out - ref).abs().max():.2e})"
    print("  Exercise 3 (CausalDiTBlock.forward): OK")


def test_causal_dit_forward(CausalDiT):
    """Test CausalDiT.forward values against reference."""
    t.manual_seed(42)
    kwargs = dict(h=24, w=24, n_actions=4, in_channels=3,
                  patch_size=3, n_blocks=2, d=64, n_head=4, exp=4,
                  n_registers=1, n_window=10)

    model = CausalDiT(**kwargs).to(device)
    ref_model = solutions.CausalDiT(**kwargs).to(device)
    ref_model.load_state_dict(model.state_dict())

    t.manual_seed(123)
    x = t.randn(2, 5, 3, 24, 24, device=device)
    actions = t.randint(0, 4, (2, 5), device=device)
    ts = t.rand(2, 5, device=device)

    out = model(x, actions, ts)
    ref = ref_model(x, actions, ts)
    assert out.shape == ref.shape, f"Expected {ref.shape}, got {out.shape}"
    assert t.allclose(out, ref, atol=1e-4), f"CausalDiT output values don't match reference (max diff: {(out - ref).abs().max():.2e})"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Exercise 4 (CausalDiT.forward): OK — {n_params:,} parameters")


def test_causal_dit_state_dict_keys(CausalDiT):
    """Check that Part 3 CausalDiT has the same state_dict keys as Part 4."""
    model = CausalDiT(h=24, w=24, n_actions=4, in_channels=3,
                       patch_size=3, n_blocks=2, d=64, n_head=4, exp=4,
                       n_registers=1, n_window=10)
    keys = set(model.state_dict().keys())
    required_prefixes = [
        'blocks.0.modulation.',
        'blocks.0.selfattn.',
        'blocks.0.geglu.',
        'blocks.0.norm1.w',
        'patch.init_conv_seq.',
        'unpatch.unpatch.',
        'registers',
        'time_emb_mixer.',
        'rope_seq.',
        'modulation.',
    ]
    for prefix in required_prefixes:
        matching = [k for k in keys if k.startswith(prefix)]
        assert len(matching) > 0, f"No state_dict key starting with '{prefix}'"
    print("  State dict key check: OK (compatible with Part 4)")


def test_sample_video(sample_video, CausalDiT):
    """Test video sampling values against reference."""
    t.manual_seed(42)
    kwargs = dict(h=24, w=24, n_actions=4, in_channels=3,
                  patch_size=3, n_blocks=2, d=64, n_head=4, exp=4,
                  n_registers=1, n_window=10)

    model = CausalDiT(**kwargs).to(device)
    ref_model = solutions.CausalDiT(**kwargs).to(device)
    ref_model.load_state_dict(model.state_dict())

    first_frame = t.randn(1, 1, 3, 24, 24, device=device)
    act_seq = t.randint(1, 4, (1, 4), device=device)

    t.manual_seed(99)
    generated = sample_video(model, first_frame, act_seq, n_denoise_steps=3, cfg=0)
    t.manual_seed(99)
    ref = solutions.sample_video(ref_model, first_frame, act_seq, n_denoise_steps=3, cfg=0)

    assert generated.shape == ref.shape, f"Expected {ref.shape}, got {generated.shape}"
    assert t.allclose(generated[:, 0], first_frame[:, 0]), "First frame should be preserved!"
    assert t.allclose(generated, ref, atol=1e-3), f"Sampling output values don't match reference (max diff: {(generated - ref).abs().max():.2e})"
    print("  Exercise 5 (video sampling): OK")


def run_all_tests():
    """Run all tests using reference solutions."""
    print("=" * 60)
    print("Part 3: Frame-Autoregressive Pong — Tests")
    print("=" * 60)

    test_block_causal_mask(solutions.make_block_causal_mask)
    test_modulate(solutions.modulate, solutions.gate_fn)
    test_causal_dit_block_forward(solutions.CausalDiTBlock, solutions.make_block_causal_mask)
    test_causal_dit_forward(solutions.CausalDiT)
    test_causal_dit_state_dict_keys(solutions.CausalDiT)
    test_sample_video(solutions.sample_video, solutions.CausalDiT)

    print("\nTraining: provided as starter code (requires Pong data)")

    print("\n" + "=" * 60)
    print("All Part 3 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
