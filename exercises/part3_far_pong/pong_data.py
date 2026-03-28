"""
Pong 1M data — same layout as toy-wm ``src/datasets/pong1m.py``.

**Raw arrays** — :func:`load_pong_data` returns ``frames`` ``(N,H,W,C)`` uint8 and
``actions`` ``(N,)`` for plotting or inspection.

**Training** — :func:`get_pong_loader` returns ``(loader, pred2frame)``:

- ``loader`` yields batches ``(frames, actions)`` with shapes ``(B, T, 3, 24, 24)``
  and ``(B, T)``, with ``T = fps * duration`` (defaults: 150).
- ``pred2frame`` maps model outputs in ``[-1, 1]`` to uint8 frames.

The underlying ``TensorDataset`` is ``loader.dataset``; index ``dataset[i]`` for a
single episode (no batch dimension). Alias: ``get_loader``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as t
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset


def fixed2frame(y, lam=1e-6):
    y = y.clamp(-1, 1) * 0.5 + 0.5
    return (y * 255.0).round().byte()


def _resolve_pong_npy_paths(data_dir=None):
    """Prefer local ``datasets/pong1M``, else HuggingFace ``chrisxx/pong``."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    here = Path(__file__).resolve().parent
    candidates.append(here / "datasets" / "pong1M")
    candidates.append(Path("datasets") / "pong1M")
    candidates.append(Path.cwd() / "datasets" / "pong1M")
    for p in candidates:
        fp, ap = p / "frames.npy", p / "actions.npy"
        if fp.is_file() and ap.is_file():
            return fp, ap
    from huggingface_hub import hf_hub_download

    repo = "chrisxx/pong"
    return (
        Path(hf_hub_download(repo, "frames.npy", repo_type="dataset")),
        Path(hf_hub_download(repo, "actions.npy", repo_type="dataset")),
    )


def load_pong_data(data_dir=None):
    """Load raw uint8 ``frames`` (N, H, W, C) and ``actions`` (N,)."""
    fp, ap = _resolve_pong_npy_paths(data_dir)
    frames = np.load(fp)
    actions = np.load(ap)
    return frames, actions


def preprocess_pong_episodes(frames_np, actions_np, fps=30, duration=5):
    """Episode packing and preprocessing (toy-wm ``get_loader`` before batching).

    Returns:
        frames: (n_episodes, T, 3, H, W), T = fps * duration
        actions: (n_episodes, T) long, values 1..3 (0 reserved for CFG)
        pred2frame: maps model outputs to uint8 frames
    """
    frames = t.from_numpy(np.ascontiguousarray(frames_np))
    actions = t.from_numpy(np.ascontiguousarray(actions_np)).long()
    height, width, channels = frames.shape[-3:]
    frames_per_example = fps * duration + 1
    n = frames.shape[0] // frames_per_example
    frames = frames[: n * frames_per_example]
    frames = frames.reshape(n, frames_per_example, height, width, channels)
    frames = frames.permute(0, 1, 4, 2, 3)
    actions = actions[: n * frames_per_example]
    actions = actions.reshape(-1, frames_per_example)
    b, dur, c, h, w = frames.shape
    z = rearrange(frames, "b dur c h w -> (b dur h w) c")
    mask = (z == t.tensor([6, 24, 24], dtype=z.dtype, device=z.device)).all(dim=1)
    z = (z.float() / 255.0 - 0.5) * 2
    z[mask] = 0
    z = rearrange(z, "(b dur h w) c -> b dur c h w", b=b, dur=dur, c=c, h=h, w=w)
    frames = z
    pred2frame = fixed2frame

    actions = actions + 1
    frames = frames[:, 1:]
    actions = actions[:, :-1]

    return frames, actions, pred2frame


def get_pong_loader(
    batch_size=64,
    fps=30,
    duration=5,
    shuffle=True,
    debug=False,
    drop_duration=False,
    data_dir=None,
    num_workers=0,
    **dataloader_kwargs,
):
    """Build ``TensorDataset(frames, actions)`` and ``DataLoader`` (toy-wm ``get_loader``).

    Each batch item is a full episode: ``frames`` (B, T, 3, H, W), ``actions`` (B, T).

    Returns:
        loader, pred2frame
    """
    frames_np, actions_np = load_pong_data(data_dir)
    frames, actions, pred2frame = preprocess_pong_episodes(
        frames_np, actions_np, fps=fps, duration=duration
    )

    firstf = frames[0]
    firsta = actions[0]
    if debug:
        frames = 0 * frames + firstf[None]
        actions = 0 * actions + firsta[None]
        frames = 0 * frames + frames[:, 0].unsqueeze(1)

    if drop_duration:
        dataset = TensorDataset(frames[:, 0], actions[:, 0] * 0)
    else:
        dataset = TensorDataset(frames, actions)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **dataloader_kwargs,
    )
    n_batches = max(frames.shape[0] // batch_size, 1)
    print(f"{n_batches} batches (dataset size {len(dataset)})")
    return loader, pred2frame


get_loader = get_pong_loader
