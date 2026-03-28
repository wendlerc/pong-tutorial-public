"""
Part 2: Flow Matching on MNIST — Tests
=======================================

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
from part2_flow_matching_mnist import solutions

device = solutions.device


def test_patch(Patch):
    """Test Patch converts images to correct token shape."""
    patch = Patch(patch_size=4, in_channels=1, d=32)
    x = t.randn(2, 1, 28, 28)
    tokens = patch(x)
    assert tokens.shape == (2, 49, 32), f"Expected (2, 49, 32), got {tokens.shape}"
    print("  Exercise 1 (Patch): OK")


def test_unpatch(Patch, UnPatch):
    """Test UnPatch reconstructs correct image shape and checks recovery of x."""
    patch = Patch(patch_size=4, in_channels=1, d=32)
    unpatch = UnPatch(patch_size=4, d=32, out_channels=1)
    x = t.randn(2, 1, 28, 28)
    tokens = patch(x)
    reconstructed = unpatch(tokens)
    assert reconstructed.shape == (2, 1, 28, 28), f"Expected (2, 1, 28, 28), got {reconstructed.shape}"
    assert not t.any(t.isnan(reconstructed)), "UnPatch should not produce NaN"
    print("  Exercise 2 (UnPatch): OK")


def test_dit_block(DiTBlock):
    """Test DiTBlock output shape and that conditioning affects output."""
    block = DiTBlock(d=32, n_head=4, exp=2)
    x_in = t.randn(2, 49, 32)
    c_in = t.randn(2, 32)
    x_out = block(x_in, c_in)
    assert x_out.shape == (2, 49, 32), f"Expected (2, 49, 32), got {x_out.shape}"

    c_in2 = t.randn(2, 32)
    x_out2 = block(x_in, c_in2)
    assert not t.allclose(x_out, x_out2), "Different conditioning should produce different outputs"
    print("  Exercise 3 (DiTBlock): OK")


def test_dit(DiT):
    """Test full DiT model output shape and parameter count."""
    model = DiT(h=28, w=28, n_classes=11, d=32, n_head=4, n_blocks=4).to(device)
    x = t.randn(2, 1, 28, 28, device=device)
    c = t.tensor([3, 7], device=device)
    ts = t.tensor([0.5, 0.8], device=device)
    v = model(x, c, ts)
    assert v.shape == (2, 1, 28, 28), f"Expected (2, 1, 28, 28), got {v.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Exercise 4 (DiT): OK — {n_params:,} parameters")


def test_sample(sample, DiT):
    """Test Euler sampling produces correct output shape."""
    model = DiT(h=28, w=28, n_classes=11, d=32, n_head=4, n_blocks=4).to(device)
    z = t.randn(2, 1, 28, 28, device=device)
    y = t.tensor([3, 7], device=device)
    samples = sample(model, z, y, n_steps=5, cfg=0)
    assert samples.shape == (2, 1, 28, 28), f"Expected (2, 1, 28, 28), got {samples.shape}"
    samples_cfg = sample(model, z, y, n_steps=5, cfg=3)
    assert samples_cfg.shape == (2, 1, 28, 28), f"CFG: expected (2, 1, 28, 28), got {samples_cfg.shape}"
    print("  Exercise 6 (Euler sampling): OK")


def run_all_tests():
    """Run all tests using reference solutions."""
    print("=" * 60)
    print("Part 2: Flow Matching on MNIST — Tests")
    print("=" * 60)

    test_patch(solutions.Patch)
    test_unpatch(solutions.Patch, solutions.UnPatch)
    test_dit_block(solutions.DiTBlock)
    test_dit(solutions.DiT)
    test_sample(solutions.sample, solutions.DiT)

    print("\nExercise 5 (training): requires MNIST data")

    print("\n" + "=" * 60)
    print("All Part 2 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
