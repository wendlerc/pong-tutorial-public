"""
Part 4: KV Caching for Frame-Autoregressive Inference — Tests
==============================================================

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
import time
from part4_far_kv_cache import solutions

device = solutions.device

# Small config for fast tests (architecture matches pretrained model)
TEST_CONFIG = dict(h=24, w=24, n_actions=4, in_channels=3,
                   patch_size=3, n_blocks=2, d=64, n_head=4,
                   exp=2, T=100, C=500, n_registers=1, n_window=10)


def test_video_kv_cache(VideoKVCache):
    """Test VideoKVCache append, get, and get_and_extend."""
    cache = VideoKVCache(n_layers=4)
    assert cache.cached_seq_len == 0, "Empty cache should have seq_len=0"

    k = t.randn(2, 4, 9, 16)
    v = t.randn(2, 4, 9, 16)
    cache.append(0, k, v)
    assert cache.cached_seq_len == 9, f"Expected 9, got {cache.cached_seq_len}"

    k2 = t.randn(2, 4, 9, 16)
    v2 = t.randn(2, 4, 9, 16)
    cache.append(0, k2, v2)
    assert cache.cached_seq_len == 18, f"Expected 18, got {cache.cached_seq_len}"

    # get_and_extend should NOT modify cache
    k3 = t.randn(2, 4, 9, 16)
    v3 = t.randn(2, 4, 9, 16)
    k_ext, v_ext = cache.get_and_extend(0, k3, v3)
    assert k_ext.shape == (2, 4, 27, 16), f"Expected (2,4,27,16), got {k_ext.shape}"
    assert cache.cached_seq_len == 18, "get_and_extend should not modify cache"

    # Sliding window eviction
    cache_sw = VideoKVCache(n_layers=1, max_seq_len=20)
    cache_sw.append(0, t.randn(2, 4, 15, 16), t.randn(2, 4, 15, 16))
    assert cache_sw.cached_seq_len == 15
    cache_sw.append(0, t.randn(2, 4, 9, 16), t.randn(2, 4, 9, 16))
    assert cache_sw.cached_seq_len == 20, f"Expected 20 after eviction, got {cache_sw.cached_seq_len}"

    print("  Exercise 1 (VideoKVCache): OK")


def test_cached_video_attention(CachedVideoAttention, VideoKVCache, make_block_causal_mask):
    """Test CachedVideoAttention: shapes + correctness vs non-cached."""
    tpf = TEST_CONFIG["h"] // TEST_CONFIG["patch_size"]
    tpf = tpf * tpf + TEST_CONFIG["n_registers"]
    d = TEST_CONFIG["d"]
    n_head = TEST_CONFIG["n_head"]

    # Build a non-cached reference and a cached version with shared weights
    ref_attn = solutions.CausalVideoAttention(d=d, n_head=n_head).to(device)
    cached_attn = CachedVideoAttention(d=d, n_head=n_head).to(device)
    cached_attn.load_state_dict(ref_attn.state_dict())

    # Standard mode — shape check
    n_frames = 3
    x = t.randn(1, n_frames * tpf, d, device=device)
    mask = make_block_causal_mask(n_frames, tpf, device=device)
    out_standard = cached_attn(x, mask=mask)
    assert out_standard.shape == (1, n_frames * tpf, d), f"Standard: unexpected shape {out_standard.shape}"

    # Correctness: finalize frame0 + denoise frame1 must match non-cached 2-frame forward
    x_two = t.randn(1, 2 * tpf, d, device=device)
    x_f0, x_f1 = x_two[:, :tpf], x_two[:, tpf:]
    mask2 = make_block_causal_mask(2, tpf, device=device)

    out_ref = ref_attn(x_two, mask=mask2)  # non-cached reference

    cache = VideoKVCache(n_layers=1)
    out_fin = cached_attn(x_f0, kv_cache=cache, layer_idx=0, cache_mode="finalize")
    assert cache.cached_seq_len == tpf, f"Finalize: expected {tpf} cached, got {cache.cached_seq_len}"
    out_den = cached_attn(x_f1, kv_cache=cache, layer_idx=0, cache_mode="denoise")
    assert cache.cached_seq_len == tpf, "Denoise should not modify cache"

    diff_f0 = (out_fin - out_ref[:, :tpf]).abs().max().item()
    diff_f1 = (out_den - out_ref[:, tpf:]).abs().max().item()
    assert diff_f0 < 1e-3, f"Finalize output diverges from reference: {diff_f0}"
    assert diff_f1 < 1e-3, f"Denoise output diverges from reference: {diff_f1}"

    print(f"  Exercise 2 (CachedVideoAttention): OK — max_diff f0={diff_f0:.6f}, f1={diff_f1:.6f}")


def test_cached_causal_dit(CachedCausalDiT, VideoKVCache):
    """Test CachedCausalDiT: shapes + correctness vs non-cached."""
    model_naive = solutions.CausalDiT(**TEST_CONFIG).to(device)
    model_cached = CachedCausalDiT(**TEST_CONFIG).to(device)
    model_cached.load_state_dict(model_naive.state_dict())

    # Normal (uncached) mode — shape check
    x = t.randn(1, 3, 3, 24, 24, device=device)
    actions = t.randint(0, 4, (1, 3), device=device)
    ts = t.rand(1, 3, device=device)
    out = model_cached(x, actions, ts)
    assert out.shape == (1, 3, 3, 24, 24), f"Normal: expected (1,3,3,24,24), got {out.shape}"

    # Correctness: finalize frame0 + denoise frame1 vs non-cached 2-frame forward
    x2 = t.randn(1, 2, 3, 24, 24, device=device)
    act2 = t.randint(1, 4, (1, 2), device=device)
    ts2 = t.tensor([[0.0, 0.5]], device=device)

    out_ref = model_naive(x2, act2, ts2)

    cache = VideoKVCache(model_cached.n_blocks)
    out_fin = model_cached(x2[:, :1], act2[:, :1], ts2[:, :1], kv_cache=cache, cache_mode="finalize")
    out_den = model_cached(x2[:, 1:], act2[:, 1:], ts2[:, 1:], kv_cache=cache, cache_mode="denoise")

    diff_f0 = (out_fin - out_ref[:, :1]).abs().max().item()
    diff_f1 = (out_den - out_ref[:, 1:]).abs().max().item()
    assert diff_f0 < 1e-3, f"Finalize output diverges: {diff_f0}"
    assert diff_f1 < 1e-3, f"Denoise output diverges: {diff_f1}"

    print(f"  Exercise 3 (CachedCausalDiT): OK — max_diff f0={diff_f0:.6f}, f1={diff_f1:.6f}")


def test_sample_video_cached(sample_video_cached, CachedCausalDiT):
    """Test cached sampling: shape, first frame, and correctness vs non-cached."""
    model_naive = solutions.CausalDiT(**TEST_CONFIG).to(device)
    model_cached = CachedCausalDiT(**TEST_CONFIG).to(device)
    model_cached.load_state_dict(model_naive.state_dict())

    first_frame = t.randn(1, 1, 3, 24, 24, device=device)
    act_seq = t.randint(1, 4, (1, 4), device=device)

    # Shape + first frame preserved
    t.manual_seed(42)
    generated = sample_video_cached(model_cached, first_frame, act_seq, n_denoise_steps=3, cfg=0)
    assert generated.shape == (1, 4, 3, 24, 24), f"Expected (1,4,3,24,24), got {generated.shape}"
    assert t.allclose(generated[:, 0], first_frame[:, 0]), "First frame should be preserved!"

    # Correctness vs non-cached
    t.manual_seed(42)
    video_naive = solutions.sample_video(model_naive, first_frame, act_seq, n_denoise_steps=3, cfg=0)
    max_diff = (generated - video_naive).abs().max().item()
    assert max_diff < 1e-3, f"Cached vs naive diverge: {max_diff}"

    print(f"  Exercise 4 (cached video sampling): OK — max_diff={max_diff:.6f}")


def test_correctness(CausalDiT, CachedCausalDiT, sample_video, sample_video_cached):
    """Verify cached and uncached produce matching results (end-to-end)."""
    model_naive = CausalDiT(**TEST_CONFIG).to(device)
    model_cached = CachedCausalDiT(**TEST_CONFIG).to(device)
    model_cached.load_state_dict(model_naive.state_dict())

    max_diff, _, _ = solutions.verify_correctness(model_naive, model_cached)
    assert max_diff < 1e-3, f"Outputs diverge too much: {max_diff}"
    print(f"  Exercise 5 (correctness): OK — max_diff={max_diff:.6f}")


def run_all_tests():
    """Run all tests using reference solutions."""
    print("=" * 60)
    print("Part 4: KV Caching for FAR Inference — Tests")
    print("=" * 60)

    test_video_kv_cache(solutions.VideoKVCache)
    test_cached_video_attention(solutions.CachedVideoAttention, solutions.VideoKVCache,
                                solutions.make_block_causal_mask)
    test_cached_causal_dit(solutions.CachedCausalDiT, solutions.VideoKVCache)
    test_sample_video_cached(solutions.sample_video_cached, solutions.CachedCausalDiT)
    test_correctness(solutions.CausalDiT, solutions.CachedCausalDiT,
                     solutions.sample_video, solutions.sample_video_cached)

    print("\n" + "=" * 60)
    print("All Part 4 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
