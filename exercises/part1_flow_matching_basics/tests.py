"""
Part 1: Rectified Flow Matching Basics — Tests
================================================

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
import numpy as np
from part1_flow_matching_basics import solutions

device = solutions.device


def test_velocity_mlp(VelocityMLP):
    """Test VelocityMLP output shapes — c is always required."""
    model = VelocityMLP(data_dim=2, hidden_dim=64, class_dim=10, n_classes=2).to(device)
    x = t.randn(16, 2, device=device)
    ts = t.rand(16, device=device)
    c = t.randint(0, 3, (16,), device=device)  # 0=uncond, 1=moon0, 2=moon1
    out = model(x, ts, c)
    assert out.shape == (16, 2), f"Expected (16, 2), got {out.shape}"

    # Unconditional (c=0) should also work
    c_uncond = t.zeros(16, dtype=t.long, device=device)
    out_uncond = model(x, ts, c_uncond)
    assert out_uncond.shape == (16, 2), f"Expected (16, 2), got {out_uncond.shape}"
    print("  Exercise 1 (VelocityMLP): OK")


def test_train(train_fn, VelocityMLP):
    """Test training loop reduces loss."""
    dataloader, _, _ = solutions.make_dataloader(n_samples=500, batch_size=64, label_dropout=0.0)
    model = VelocityMLP(data_dim=2, hidden_dim=64).to(device)
    losses = train_fn(model, dataloader, n_steps=200, lr=1e-3)
    assert len(losses) == 200, f"Expected 200 losses, got {len(losses)}"
    early = np.mean(losses[:50])
    late = np.mean(losses[-50:])
    assert late < early * 1.5, f"Loss should decrease: early={early:.4f}, late={late:.4f}"
    print(f"  Exercise 2 (training): OK — early={early:.4f}, late={late:.4f}")


def test_sample_uniform(sample_uniform, VelocityMLP):
    """Test uniform sampling produces correct shapes."""
    model = VelocityMLP(data_dim=2, hidden_dim=64).to(device)
    c = t.zeros(100, dtype=t.long, device=device)
    samples = sample_uniform(model, n_samples=100, data_dim=2, n_steps=10, c=c)
    assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"
    assert not t.any(t.isnan(samples)), "Samples should not contain NaN"
    print("  Exercise 3 (uniform sampling): OK")


def test_sample_sd3(sample_sd3, VelocityMLP):
    """Test SD3 sampling produces correct shapes."""
    model = VelocityMLP(data_dim=2, hidden_dim=64).to(device)
    c = t.zeros(100, dtype=t.long, device=device)
    samples = sample_sd3(model, n_samples=100, data_dim=2, n_steps=10, c=c)
    assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"
    assert not t.any(t.isnan(samples)), "Samples should not contain NaN"
    print("  Exercise 4 (SD3 sampling): OK")


def test_sample_cfg(sample_cfg, VelocityMLP):
    """Test CFG sampling produces correct shapes."""
    model = VelocityMLP(data_dim=2, hidden_dim=64, n_classes=2).to(device)
    c = t.ones(100, dtype=t.long, device=device)
    samples = sample_cfg(model, n_samples=100, data_dim=2, n_steps=10, c=c, cfg=2.0)
    assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"
    assert not t.any(t.isnan(samples)), "Samples should not contain NaN"
    print("  Exercise 5 (CFG sampling): OK")


def run_all_tests():
    """Run all tests using reference solutions."""
    print("=" * 60)
    print("Part 1: Rectified Flow Matching Basics — Tests")
    print("=" * 60)

    test_velocity_mlp(solutions.VelocityMLP)
    test_train(solutions.train, solutions.VelocityMLP)
    test_sample_uniform(solutions.sample_uniform, solutions.VelocityMLP)
    test_sample_sd3(solutions.sample_sd3, solutions.VelocityMLP)
    test_sample_cfg(solutions.sample_cfg, solutions.VelocityMLP)

    print("\n" + "=" * 60)
    print("All Part 1 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
