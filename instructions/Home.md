# Introduction to frame-autoregressive video models

A reading list and collection of resources for building up to frame-autoregressive video generation and world models. The progression goes: flow matching basics on 2D data, then flow matching for images with a DiT, then flow matching for video with causal attention, and finally KV caching for efficient frame-autoregressive generation.

🟢 **read** — essential reading · 🟡 **skim** — worth skimming · 🟣 **bonus** — for deeper exploration · 🔵 **code** — implementation

## 1. Rectified flow matching basics

Learn the fundamentals of flow matching on a simple 2D dataset (two moons): velocity prediction, training, Euler sampling, noise schedules (uniform, SD3), and classifier-free guidance.

* 📓 **[Part 1: Rectified Flow Matching Basics](../exercises/part1_flow_matching_basics/exercises.ipynb)**
* 🟢 read: [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) -- the SD3 paper that popularized rectified flow matching
* 🟣 bonus: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) -- the original DDPM paper
* 🟣 bonus: [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) -- deterministic sampling
* 🟣 bonus: [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) -- run diffusion in a learned latent space

## 2. Flow matching on MNIST

Build a Diffusion Transformer (DiT) from scratch and train it on MNIST with flow matching, classifier-free guidance, and the Muon optimizer.

* 📓 **[Part 2: Flow Matching on MNIST](../exercises/part2_flow_matching_mnist/exercises.ipynb)**
* 🟢 read: [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) -- the transformer architecture and conditioning mechanism used throughout this tutorial
* 🔵 code: [Rectified flow matching on MNIST](https://github.com/wendlerc/mnist) -- minimal implementation on MNIST
* 🔵 code: [minRF: Rectified flow matching on CIFAR](https://github.com/cloneofsimo/minRF) -- minimal implementation on CIFAR

## 3. Frame-autoregressive Pong

Extend the DiT to video: block-causal attention masks, per-frame conditioning via modulate/gate, action conditioning, and diffusion forcing training.

* 📓 **[Part 3: Frame-Autoregressive Pong](../exercises/part3_far_pong/exercises.ipynb)**
* 🟡 skim: [Diffusion Forcing](https://arxiv.org/abs/2407.01392) -- bridges autoregressive and diffusion models for sequence generation
* 🟡 skim: [WAN: Scalable Bidirectional Text-to-Video Generation](https://arxiv.org/abs/2503.20314) -- bidirectional text-to-video generation
* 🟣 bonus: [CausVid](https://arxiv.org/abs/2412.07772v1) -- causal video generation (advanced)
* 🟣 bonus: [Self-Forcing](https://arxiv.org/abs/2506.08009) -- training autoregressive video models without teacher forcing (advanced)
* 🔵 code: [Minimal diffusion forcing implementation](https://github.com/wendlerc/toy-wm/) -- the reference implementation this tutorial follows

## 4. KV caching for frame-autoregressive inference

Add KV caching to the video DiT for efficient autoregressive generation: VideoKVCache with finalize/denoise modes, CachedVideoAttention with RoPE recomputation, and sliding-window eviction.

* 📓 **[Part 4: KV Caching for FAR Inference](../exercises/part4_far_kv_cache/exercises.ipynb)**
* 🟢 read: [Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- Sebastian Raschka's hands-on walkthrough of KV caching
* 🔵 code: [minGPT / nanoGPT](https://github.com/karpathy/minGPT) -- Karpathy's clean, minimal GPT implementations
* [ARENA Chapter 1.1: Transformer from Scratch](https://arena-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch) -- step-by-step implementation of a transformer
