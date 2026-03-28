# Pong Tutorial

ARENA-style tutorial on video rectified flow matching based world model training.

## Setup

```bash
git clone git@github.com:wendlerc/pong-tutorial-public.git pong-tutorial
cd pong-tutorial
uv sync
uv pip install ipykernel
.venv/bin/python -m ipykernel install --user --name pong-tutorial --display-name "pong-tutorial (.venv)"
```

Then open any exercise notebook and select the **"pong-tutorial (.venv)"** kernel. In VS Code you may need to reload the window (`Ctrl+Shift+P` → "Reload Window") for the kernel to appear.

## Exercises

| # | Exercise | Topic |
|---|----------|-------|
| 1 | [Rectified Flow Matching Basics](exercises/part1_flow_matching_basics/exercises.ipynb) | Flow matching on two moons: velocity prediction, training, sampling, schedules, CFG |
| 2 | [Flow Matching on MNIST](exercises/part2_flow_matching_mnist/exercises.ipynb) | Build a DiT from scratch: patchify, flow matching, training, sampling, CFG |
| 3 | [Frame-Autoregressive Pong](exercises/part3_far_pong/exercises.ipynb) | Extend to video with causal attention, action conditioning, diffusion forcing |
| 4 | [KV Caching for FAR Inference](exercises/part4_far_kv_cache/exercises.ipynb) | KV caching for efficient video generation |

Each exercise directory contains:
- `exercises.ipynb` — notebook with stub functions (`# YOUR CODE HERE`)
- `solutions.py` — reference implementations
- `tests.py` — test suite validating against solutions

Part 3 also includes `pong_data.py` — Pong data loading (`load_pong_data`, `get_pong_loader`) matching toy-wm `pong1m.py`.

## Structure

```
instructions/                          # HTML overview & reading list
exercises/
  part1_flow_matching_basics/          # Two moons + MLP (7 exercises)
  part2_flow_matching_mnist/           # DiT on MNIST (12 exercises)
  part3_far_pong/                      # FAR video model (9 exercises) + pong_data.py
  part4_far_kv_cache/                   # KV-cache for video (5 exercises)
```
