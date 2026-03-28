# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARENA-style tutorial on video rectified flow matching world models. Five progressive parts: flow matching basics (2D) → DiT on MNIST → frame-autoregressive Pong video generation → KV-cache for text → KV-cache for video. Each part has `exercises.ipynb` (student stubs), `solutions.py` (reference implementations), and `tests.py` (validation).

## Setup & Commands

```bash
uv sync                          # Install dependencies
uv pip install ipykernel         # Jupyter kernel support
```

### Running tests

```bash
# Run all tests for a part
cd exercises/part1_flow_matching_basics && python tests.py

# Run a specific test
cd exercises/part2_flow_matching_mnist && pytest tests.py::test_patch -xvs
```

### Training

```bash
python train_toy_lm.py           # Train byte-level LM on tutorial corpus (saves to checkpoints/)
```

## Architecture

The tutorial builds a video generation pipeline incrementally:

1. **Part 1** (`part1_flow_matching_basics`): VelocityMLP, flow matching loss, Euler sampling with schedules (uniform, SD3), classifier-free guidance on 2D two-moons data.

2. **Part 2** (`part2_flow_matching_mnist`): DiT architecture — Patch/UnPatch (strided conv tokenization), RMSNorm, multi-head Attention with QK-norm, gated SiLU MLP, DiTBlock with AdaLN (6 modulation params per block), Muon optimizer. Trains on MNIST.

3. **Part 3** (`part3_far_pong`): Extends DiT to video — VideoPatch/VideoUnPatch follow toy-wm [`patch.py`](https://github.com/wendlerc/toy-wm/blob/main/src/nn/patch.py) (init convs → patchify → linear embedder; inverse linear + `einsum` unpatchify); block-causal attention masks (frame t can't attend to t+1); CausalDiTBlock with action conditioning; diffusion forcing loss. Data: `pong_data.py` (`load_pong_data`, `get_pong_loader` → `TensorDataset` + `DataLoader`), same preprocessing as toy-wm `pong1m.py`; uses local `datasets/pong1M/` if present else HuggingFace `chrisxx/pong`.

4. **Part 4** (`part4_far_kv_cache`): VideoKVCache with finalize/denoise modes and sliding-window eviction, CachedVideoAttention with RoPE recomputation, CachedCausalDiT for efficient frame-autoregressive video inference. Loads pretrained model from `chrisxx/pong`.

### Key patterns across all parts

- **Conditioning**: AdaLN modulation (scale, shift, gate) driven by timestep/class/action embeddings
- **Flow matching**: Loss = `MSE(v_pred, x - z)`, logit-normal timestep sampling
- **Sampling**: Euler integration, classifier-free guidance via unconditional token dropout during training
- **Video causality**: Block-causal masks ensure temporal ordering; each frame is a block of patch tokens

## External Data

- `chrisxx/pong` (HuggingFace) — Pong video frames + actions (used by `part3_far_pong/pong_data.py` when local `datasets/pong1M/` is absent)
- `chrisxx/toy-lm` (HuggingFace) — Pretrained byte-level transformer
- MNIST via torchvision

## Important Notes

- Never commit `*.pt`, `*.ckpt`, or other large files — already in `.gitignore`
- `solutions.py` files are the ground truth; `tests.py` validates student code against them
- Exercise notebooks use `# YOUR CODE HERE` stubs for students to fill in
- No CI/CD pipeline; testing is manual via `tests.py`
