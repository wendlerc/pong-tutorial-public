"""
Part 1: Rectified Flow Matching Basics — Solutions
===================================================

Reference implementations for all exercises.
Uses the two moons dataset to teach flow matching fundamentals
with a simple MLP velocity predictor.

Exercises:
  1. VelocityMLP — gated MLP with class embedding
  2. Training loop — flow matching loss inline, with CFG dropout
  3. Euler sampler — uniform schedule
  4. Improved sampler — SD3 schedule
  5. CFG sampler — SD3 + classifier-free guidance
"""

import torch
import torch as t
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Starter code ────────────────────────────────────────────────────────────

def make_two_moons(n_samples=2000, noise=0.04):
    """Generate two moons dataset. Returns (points, labels)."""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=noise)
    return t.tensor(X, dtype=t.float32), t.tensor(y, dtype=t.long)


def make_dataloader(n_samples=5000, batch_size=256, noise=0.04, label_dropout=0.2):
    """Create a dataloader that yields (points, labels) with label shift and dropout.

    Labels: 0 = unconditional, 1 = moon 0, 2 = moon 1.
    With label_dropout=0.2, ~20% of labels are replaced with 0.
    """
    data, labels = make_two_moons(n_samples, noise=noise)
    labels = labels + 1  # shift: 0=uncond, 1=moon0, 2=moon1

    class TwoMoonsDataset(t.utils.data.Dataset):
        def __init__(self, data, labels, label_dropout):
            self.data = data
            self.labels = labels
            self.label_dropout = label_dropout

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x = self.data[idx]
            c = self.labels[idx].clone()
            if self.label_dropout > 0 and t.rand(1).item() < self.label_dropout:
                c = t.tensor(0, dtype=t.long)  # unconditional
            return x, c

    dataset = TwoMoonsDataset(data, labels, label_dropout)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), data, labels


@t.no_grad()
def compute_trajectories(model, n_samples=200, n_steps=50, c=None):
    """Compute full denoising trajectories for visualization.

    Returns:
        (n_steps+1, n_samples, data_dim) tensor of positions at each step
    """
    dev = next(model.parameters()).device
    z = t.randn(n_samples, 2, device=dev)
    ts = t.linspace(1, 0, n_steps + 1, device=dev)
    ts = 3 * ts / (2 * ts + 1)

    trajectory = [z.cpu().clone()]

    for i in range(n_steps):
        t_batch = ts[i] * t.ones(n_samples, device=dev)
        v_pred = model(z, t_batch, c)
        dt = ts[i] - ts[i + 1]
        z = z + dt * v_pred
        trajectory.append(z.cpu().clone())

    return t.stack(trajectory)


# ── Exercise 1: MLP Velocity Predictor ──────────────────────────────────────

class VelocityMLP(nn.Module):
    """Simple MLP that predicts velocity given (x_t, t, class)."""
    def __init__(self, data_dim=2, hidden_dim=128, class_dim=10, n_classes=2):
        super().__init__()
        input_dim = data_dim + 1 + class_dim  # x_t, t, and class embedding
        self.class_emb = nn.Embedding(n_classes + 1, class_dim)  # +1 for unconditional (index 0)

        self.up = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(hidden_dim, data_dim)

    def forward(self, x_t, ts, c):
        """
        Args:
            x_t: (batch, data_dim) noisy points
            ts: (batch,) timesteps in [0, 1]
            c: (batch,) class labels (0 = unconditional)

        Returns:
            (batch, data_dim) predicted velocity
        """
        inp = t.cat([x_t, ts[:, None], self.class_emb(c)], dim=-1)
        return self.down(self.up(inp) * self.act(self.gate(inp)))


# ── Exercise 2: Training Loop ───────────────────────────────────────────────

def train(model, dataloader, n_steps=5000, lr=1e-3):
    """Train the velocity model with flow matching loss inline.

    Args:
        model: VelocityMLP
        dataloader: yields (points, labels) batches
        n_steps: optimization steps
        lr: learning rate

    Returns:
        list of losses
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    dl_iter = iter(dataloader)

    for step in range(n_steps):
        try:
            x, c = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            x, c = next(dl_iter)

        x = x.to(next(model.parameters()).device)
        c = c.to(x.device)

        # Flow matching loss (inline)
        ts = F.sigmoid(t.randn(x.shape[0], device=x.device))  # logit-normal sampling
        z = t.randn_like(x)
        v_true = x - z
        x_t = x - ts[:, None] * v_true
        v_pred = model(x_t, ts, c)
        loss = F.mse_loss(v_pred, v_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{n_steps} | Loss: {np.mean(losses[-100:]):.4f}")

    return losses


# ── Exercise 3: Euler Sampling (Uniform Schedule) ───────────────────────────

@t.no_grad()
def sample_uniform(model, n_samples, data_dim=2, n_steps=50, c=None):
    """Sample from the learned flow using uniform timestep schedule.

    Schedule: linearly spaced from t=1 (noise) to t=0 (data).
    """
    z = t.randn(n_samples, data_dim, device=next(model.parameters()).device)
    ts = t.linspace(1, 0, n_steps + 1, device=z.device)

    for i in range(n_steps):
        t_batch = ts[i] * t.ones(n_samples, device=z.device)
        v_pred = model(z, t_batch, c)
        dt = ts[i] - ts[i + 1]
        z = z + dt * v_pred

    return z


# ── Exercise 4: Euler Sampling (SD3 Schedule) ───────────────────────────────

@t.no_grad()
def sample_sd3(model, n_samples, data_dim=2, n_steps=50, c=None):
    """Sample using the SD3 logit-normal schedule.

    Schedule: ts = 3*ts / (2*ts + 1), applied to uniform linspace.
    """
    z = t.randn(n_samples, data_dim, device=next(model.parameters()).device)
    ts = t.linspace(1, 0, n_steps + 1, device=z.device)
    ts = 3 * ts / (2 * ts + 1)  # SD3 schedule

    for i in range(n_steps):
        t_batch = ts[i] * t.ones(n_samples, device=z.device)
        v_pred = model(z, t_batch, c)
        dt = ts[i] - ts[i + 1]
        z = z + dt * v_pred

    return z


# ── Exercise 5: Classifier-Free Guidance Sampling ───────────────────────────

@t.no_grad()
def sample_cfg(model, n_samples, data_dim=2, n_steps=50, c=None, cfg=2.0):
    """Sample with classifier-free guidance using SD3 schedule.

    v_guided = v_uncond + cfg * (v_cond - v_uncond)
    """
    z = t.randn(n_samples, data_dim, device=next(model.parameters()).device)
    ts = t.linspace(1, 0, n_steps + 1, device=z.device)
    ts = 3 * ts / (2 * ts + 1)  # SD3 schedule

    c_uncond = t.zeros_like(c)  # unconditional class = 0

    for i in range(n_steps):
        t_batch = ts[i] * t.ones(n_samples, device=z.device)
        v_cond = model(z, t_batch, c)
        v_uncond = model(z, t_batch, c_uncond)
        v_guided = v_uncond + cfg * (v_cond - v_uncond)
        dt = ts[i] - ts[i + 1]
        z = z + dt * v_guided

    return z
