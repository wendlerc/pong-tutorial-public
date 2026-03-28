"""
Part 4: KV Caching for Frame-Autoregressive Inference — Solutions
=================================================================

Reference implementations for all 5 exercises.
Architecture matches the pretrained CausalDiT from chrisxx/pong on HuggingFace.
"""

import math
import torch
import torch as t
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Shared components (provided to students) ────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(t.ones(d))

    def forward(self, x):
        return x / ((x ** 2).mean(dim=-1, keepdim=True) + 1e-6).sqrt() * self.w


class GEGLU(nn.Module):
    def __init__(self, d_in, d_mid, d_out):
        super().__init__()
        self.up_proj = nn.Linear(d_in, d_mid, bias=True)
        self.up_gate = nn.Linear(d_in, d_mid, bias=True)
        self.down = nn.Linear(d_mid, d_out, bias=True)
        self.nonlin = nn.SiLU()

    def forward(self, x):
        return self.down(self.up_proj(x) * self.nonlin(self.up_gate(x)))


class NumericEncoding(nn.Module):
    def __init__(self, C=5000, dim=64, n_max=1000):
        super().__init__()
        args = t.exp(-math.log(C) * t.arange(0, dim, 2) / dim)
        args = t.arange(n_max)[:, None] * args[None, :]
        pe = t.empty((n_max, dim))
        pe[:, ::2] = t.sin(args)
        pe[:, 1::2] = t.cos(args)
        self.register_buffer("pe", pe)

    def forward(self, num):
        return self.pe[num]


class RoPE(nn.Module):
    def __init__(self, d_head, n_ctx, C=10000):
        super().__init__()
        thetas = t.exp(-math.log(C) * t.arange(0, d_head, 2) / d_head)
        thetas = thetas.repeat(2, 1).T.flatten()
        positions = t.arange(n_ctx)
        all_thetas = positions.unsqueeze(1) * thetas.unsqueeze(0)
        sins = t.sin(all_thetas)
        coss = t.cos(all_thetas)
        self.register_buffer('sins', sins.unsqueeze(0).unsqueeze(2))
        self.register_buffer('coss', coss.unsqueeze(0).unsqueeze(2))

    def forward(self, x, offset=0):
        """x: (batch, seq, n_head, d_head)"""
        x_perm = t.empty_like(x)
        even = t.arange(0, x.shape[-1], 2, device=x.device)
        odd = t.arange(1, x.shape[-1], 2, device=x.device)
        x_perm[..., even] = -x[..., odd]
        x_perm[..., odd] = x[..., even]
        return self.coss[:, offset:offset + x.shape[1]] * x + self.sins[:, offset:offset + x.shape[1]] * x_perm


class Patch(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        dim = out_channels
        if dim % 32 == 0 and dim > 32:
            self.init_conv_seq = nn.Sequential(
                nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
                nn.GroupNorm(32, dim // 2),
                nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
                nn.GroupNorm(32, dim // 2),
            )
        else:
            self.init_conv_seq = nn.Sequential(
                nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
                nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
            )
        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)

    def patchify(self, x):
        B, C, H, W = x.size()
        ps = self.patch_size
        x = x.view(B, C, H // ps, ps, W // ps, ps)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x):
        batch, dur, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        x = self.init_conv_seq(x)
        x = self.patchify(x)
        x = self.x_embedder(x)
        x = x.reshape(batch, dur, -1, self.out_channels)
        return x


class UnPatch(nn.Module):
    def __init__(self, height, width, in_channels=64, out_channels=3, patch_size=2):
        super().__init__()
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.unpatch = nn.Linear(in_channels, out_channels * patch_size ** 2)

    def forward(self, x):
        batch, dur, seq, d = x.shape
        x = self.unpatch(x)
        x = x.reshape(-1, seq, x.shape[-1])
        c, p = self.out_channels, self.patch_size
        h, w = self.height // p, self.width // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = t.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(x.shape[0], c, h * p, w * p)
        x = x.reshape(batch, dur, c, h * p, w * p)
        return x


def modulate(x, shift, scale):
    """Apply per-frame modulation: cond is (B, T_frames, d), x is (B, S, d)."""
    b, s, d = x.shape
    toks_per_frame = s // shift.shape[1]
    x = x.reshape(b, -1, toks_per_frame, d)
    x = x * (1 + scale[:, :, None, :]) + shift[:, :, None, :]
    x = x.reshape(b, s, d)
    return x


def gate_fn(x, g):
    """Apply per-frame gating: g is (B, T_frames, d), x is (B, S, d)."""
    b, s, d = x.shape
    toks_per_frame = s // g.shape[1]
    x = x.reshape(b, -1, toks_per_frame, d)
    x = x * g[:, :, None, :]
    x = x.reshape(b, s, d)
    return x


def make_block_causal_mask(n_frames, toks_per_frame, device="cpu"):
    """Block-causal mask: token i can attend to j iff frame(i) >= frame(j)."""
    total = n_frames * toks_per_frame
    frame_idx = t.arange(total, device=device) // toks_per_frame
    mask = frame_idx[None, :] > frame_idx[:, None]  # True where BLOCKED
    return mask


# ── Non-cached CausalDiT (for comparison) ─────────────────────────────────

class CausalVideoAttention(nn.Module):
    def __init__(self, d=64, n_head=4, rope=None):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.d_head = d // n_head
        self.QKV = nn.Linear(d, 3 * d)
        self.O = nn.Linear(d, d)
        self.lnq = RMSNorm(self.d_head)
        self.lnk = RMSNorm(self.d_head)
        self.rope = rope

    def forward(self, x, mask=None):
        b, s, d = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q = q.reshape(b, s, self.n_head, self.d_head)
        k = k.reshape(b, s, self.n_head, self.d_head)
        v = v.reshape(b, s, self.n_head, self.d_head)
        q = self.lnq(q)
        k = self.lnk(k)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(mask[None, None, :, :], float('-inf'))
        attn = attn.softmax(dim=-1)
        z = attn @ v
        z = z.permute(0, 2, 1, 3).reshape(b, s, self.d)
        return self.O(z)


class CausalDiTBlock(nn.Module):
    def __init__(self, d=64, n_head=4, exp=4, rope=None):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.selfattn = CausalVideoAttention(d, n_head, rope=rope)
        self.norm2 = RMSNorm(d)
        self.geglu = GEGLU(d, exp * d, d)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d, bias=True))

    def forward(self, x, cond, mask=None):
        mu1, sigma1, c1, mu2, sigma2, c2 = self.modulation(cond).chunk(6, dim=-1)
        residual = x
        x = modulate(self.norm1(x), mu1, sigma1)
        x = self.selfattn(x, mask=mask)
        x = residual + gate_fn(x, c1)
        residual = x
        x = modulate(self.norm2(x), mu2, sigma2)
        x = self.geglu(x)
        x = residual + gate_fn(x, c2)
        return x


class CausalDiT(nn.Module):
    """Non-cached version for comparison."""
    def __init__(self, h=24, w=24, n_actions=4, in_channels=3,
                 patch_size=3, n_blocks=8, d=320, n_head=20, exp=4,
                 T=1000, C=5000, n_registers=1, n_window=30):
        super().__init__()
        self.T = T
        self.n_blocks = n_blocks
        self.n_registers = n_registers
        self.n_window = n_window
        self.toks_per_frame = (h // patch_size) * (w // patch_size) + n_registers
        self.patches_per_frame = self.toks_per_frame
        d_head = d // n_head
        rope_ctx = n_window * self.toks_per_frame
        self.rope_seq = RoPE(d_head, rope_ctx, C=C)
        self.blocks = nn.ModuleList(
            [CausalDiTBlock(d, n_head, exp, rope=self.rope_seq) for _ in range(n_blocks)]
        )
        self.patch = Patch(in_channels, d, patch_size)
        self.norm = RMSNorm(d)
        self.unpatch = UnPatch(h, w, d, in_channels, patch_size)
        self.action_emb = nn.Embedding(n_actions, d)
        self.registers = nn.Parameter(t.randn(n_registers, d) * d ** -0.5)
        self.time_emb = NumericEncoding(C=C, dim=d, n_max=T)
        self.time_emb_mixer = nn.Linear(d, d)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d, bias=True))

    def forward(self, x, actions, ts):
        B, T_frames, C, H, W = x.shape
        a = self.action_emb(actions)
        ts_scaled = (ts.float() * (self.T - 1)).long()
        cond = self.time_emb_mixer(self.time_emb(ts_scaled)) + a
        z = self.patch(x)  # (B, T_frames, n_patches, d)
        # Append registers to each frame
        regs = self.registers[None, None].expand(B, T_frames, -1, -1)
        zr = t.cat([z, regs], dim=2)  # (B, T_frames, toks_per_frame, d)
        batch, dur, seq, d = zr.shape
        zr = zr.reshape(batch, dur * seq, d)
        mask = make_block_causal_mask(T_frames, self.toks_per_frame, device=x.device)
        for block in self.blocks:
            zr = block(zr, cond, mask=mask)
        mu, sigma = self.modulation(cond).chunk(2, dim=-1)
        zr = modulate(self.norm(zr), mu, sigma)
        zr = zr.reshape(batch, dur, seq, d)
        out = self.unpatch(zr[:, :, :-self.n_registers])
        return out

    @property
    def device(self):
        return self.registers.device


@t.no_grad()
def sample_video(model, first_frame, actions, n_denoise_steps=30, cfg=0):
    """Non-cached sampling for comparison."""
    B = first_frame.shape[0]
    total_frames = actions.shape[1]
    C, H, W = first_frame.shape[2:]
    video = first_frame.clone()
    for frame_idx in range(1, total_frames):
        z = t.randn(B, 1, C, H, W, device=first_frame.device, dtype=first_frame.dtype)
        denoise_ts = t.linspace(1, 0, n_denoise_steps + 1, device=first_frame.device)
        denoise_ts = 3 * denoise_ts / (2 * denoise_ts + 1)
        for step_idx in range(n_denoise_steps):
            current_video = t.cat([video, z], dim=1)
            n_ctx = current_video.shape[1]
            ts = t.zeros(B, n_ctx, device=first_frame.device)
            ts[:, -1] = denoise_ts[step_idx]
            act = actions[:, :n_ctx]
            v_pred = model(current_video, act, ts)
            if cfg > 0:
                v_uncond = model(current_video, act * 0, ts)
                v_pred = v_uncond + cfg * (v_pred - v_uncond)
            dt = denoise_ts[step_idx] - denoise_ts[step_idx + 1]
            z = z + dt * v_pred[:, -1:]
        video = t.cat([video, z], dim=1)
    return video


# ── Exercise 1: VideoKVCache ────────────────────────────────────────────────

class VideoKVCache:
    """KV Cache for frame-autoregressive video generation.

    Stores raw (pre-norm, pre-RoPE) keys and values so that RoPE can be
    recomputed from position 0 each time — essential for sliding-window
    attention where absolute positions shift as old frames are evicted.

    Args:
        n_layers: Number of transformer layers.
        max_seq_len: Maximum cached sequence length. When exceeded, the
            oldest tokens are evicted (sliding window).
    """
    def __init__(self, n_layers, max_seq_len=None):
        self.cache = [None] * n_layers
        self.max_seq_len = max_seq_len

    def append(self, layer_idx, keys, values):
        """Permanently add keys/values to cache for this layer."""
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (keys, values)
        else:
            prev_k, prev_v = self.cache[layer_idx]
            self.cache[layer_idx] = (
                t.cat([prev_k, keys], dim=2),
                t.cat([prev_v, values], dim=2),
            )
        # Sliding window: evict oldest tokens if over capacity
        if self.max_seq_len is not None:
            k, v = self.cache[layer_idx]
            if k.shape[2] > self.max_seq_len:
                excess = k.shape[2] - self.max_seq_len
                self.cache[layer_idx] = (k[:, :, excess:], v[:, :, excess:])

    def get_cached_kv(self, layer_idx):
        """Return (cached_keys, cached_values) or (None, None)."""
        if self.cache[layer_idx] is None:
            return None, None
        return self.cache[layer_idx]

    def get_and_extend(self, layer_idx, new_keys, new_values):
        """Return (cached + new keys, cached + new values) WITHOUT modifying cache."""
        cached_k, cached_v = self.get_cached_kv(layer_idx)
        if cached_k is None:
            return new_keys, new_values
        return (
            t.cat([cached_k, new_keys], dim=2),
            t.cat([cached_v, new_values], dim=2),
        )

    @property
    def cached_seq_len(self):
        """Return the cached sequence length (from layer 0)."""
        if self.cache[0] is None:
            return 0
        return self.cache[0][0].shape[2]


# ── Exercise 2: CachedVideoAttention ────────────────────────────────────────

class CachedVideoAttention(nn.Module):
    def __init__(self, d=64, n_head=4, rope=None):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.d_head = d // n_head
        self.QKV = nn.Linear(d, 3 * d)
        self.O = nn.Linear(d, d)
        self.lnq = RMSNorm(self.d_head)
        self.lnk = RMSNorm(self.d_head)
        self.rope = rope

    def forward(self, x, mask=None, kv_cache=None, layer_idx=None, cache_mode=None):
        b, s, d = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q = q.reshape(b, s, self.n_head, self.d_head)
        k = k.reshape(b, s, self.n_head, self.d_head)
        v = v.reshape(b, s, self.n_head, self.d_head)

        if kv_cache is not None and cache_mode is not None:
            # Cache stores RAW (pre-norm, pre-RoPE) k,v so we can
            # recompute RoPE from position 0 each time — needed for
            # sliding-window attention where old frames are evicted.
            k_raw = k.permute(0, 2, 1, 3)   # (b, n_head, s, d_head)
            v_raw = v.permute(0, 2, 1, 3)

            if cache_mode == "finalize":
                kv_cache.append(layer_idx, k_raw, v_raw)
                k_all_raw, v_all = kv_cache.get_cached_kv(layer_idx)
            elif cache_mode == "denoise":
                k_all_raw, v_all = kv_cache.get_and_extend(layer_idx, k_raw, v_raw)

            # Recompute norm + RoPE on full k from position 0
            k_all = k_all_raw.permute(0, 2, 1, 3)  # (b, s_full, n_head, d_head)
            q = self.lnq(q)
            k_all = self.lnk(k_all)
            if self.rope is not None:
                offset = k_all.shape[1] - s  # q sits at the end
                q = self.rope(q, offset=offset)
                k_all = self.rope(k_all)      # from position 0
            q = q.permute(0, 2, 1, 3)
            k_all = k_all.permute(0, 2, 1, 3)
            out = (q @ k_all.transpose(-2, -1)).softmax(dim=-1) @ v_all

        else:
            # Standard (uncached) mode
            q = self.lnq(q)
            k = self.lnk(k)
            if self.rope is not None:
                q = self.rope(q)
                k = self.rope(k)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask[None, None, :, :], float('-inf'))
            out = attn.softmax(dim=-1) @ v

        out = out.permute(0, 2, 1, 3).reshape(b, s, self.d)
        return self.O(out)


# ── Exercise 3: CachedCausalDiT ─────────────────────────────────────────────

class CachedCausalDiTBlock(nn.Module):
    def __init__(self, d=64, n_head=4, exp=4, rope=None):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.selfattn = CachedVideoAttention(d, n_head, rope=rope)
        self.norm2 = RMSNorm(d)
        self.geglu = GEGLU(d, exp * d, d)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d, bias=True))

    def forward(self, x, cond, mask=None, kv_cache=None, layer_idx=None, cache_mode=None):
        mu1, sigma1, c1, mu2, sigma2, c2 = self.modulation(cond).chunk(6, dim=-1)
        residual = x
        x = modulate(self.norm1(x), mu1, sigma1)
        x = self.selfattn(x, mask=mask, kv_cache=kv_cache,
                          layer_idx=layer_idx, cache_mode=cache_mode)
        x = residual + gate_fn(x, c1)
        residual = x
        x = modulate(self.norm2(x), mu2, sigma2)
        x = self.geglu(x)
        x = residual + gate_fn(x, c2)
        return x


class CachedCausalDiT(nn.Module):
    def __init__(self, h=24, w=24, n_actions=4, in_channels=3,
                 patch_size=3, n_blocks=8, d=320, n_head=20, exp=4,
                 T=1000, C=5000, n_registers=1, n_window=30):
        super().__init__()
        self.T = T
        self.n_blocks = n_blocks
        self.n_registers = n_registers
        self.n_window = n_window
        self.toks_per_frame = (h // patch_size) * (w // patch_size) + n_registers
        self.patches_per_frame = self.toks_per_frame
        d_head = d // n_head
        rope_ctx = n_window * self.toks_per_frame
        self.rope_seq = RoPE(d_head, rope_ctx, C=C)
        self.blocks = nn.ModuleList(
            [CachedCausalDiTBlock(d, n_head, exp, rope=self.rope_seq) for _ in range(n_blocks)]
        )
        self.patch = Patch(in_channels, d, patch_size)
        self.norm = RMSNorm(d)
        self.unpatch = UnPatch(h, w, d, in_channels, patch_size)
        self.action_emb = nn.Embedding(n_actions, d)
        self.registers = nn.Parameter(t.randn(n_registers, d) * d ** -0.5)
        self.time_emb = NumericEncoding(C=C, dim=d, n_max=T)
        self.time_emb_mixer = nn.Linear(d, d)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d, bias=True))

    @property
    def max_cache_len(self):
        """Max tokens that can be cached (leave room for one denoise frame)."""
        return (self.n_window - 1) * self.toks_per_frame

    def forward(self, x, actions, ts, kv_cache=None, cache_mode=None):
        B, T_frames, C, H, W = x.shape
        a = self.action_emb(actions)
        ts_scaled = (ts.float() * (self.T - 1)).long()
        cond = self.time_emb_mixer(self.time_emb(ts_scaled)) + a

        z = self.patch(x)  # (B, T_frames, n_patches, d)
        regs = self.registers[None, None].expand(B, T_frames, -1, -1)
        zr = t.cat([z, regs], dim=2)
        batch, dur, seq, d = zr.shape
        zr = zr.reshape(batch, dur * seq, d)

        mask = None
        if cache_mode is None and T_frames > 1:
            mask = make_block_causal_mask(T_frames, self.toks_per_frame, device=x.device)

        for i, block in enumerate(self.blocks):
            zr = block(zr, cond, mask=mask, kv_cache=kv_cache,
                       layer_idx=i, cache_mode=cache_mode)

        mu, sigma = self.modulation(cond).chunk(2, dim=-1)
        zr = modulate(self.norm(zr), mu, sigma)
        zr = zr.reshape(batch, dur, seq, d)
        out = self.unpatch(zr[:, :, :-self.n_registers])
        return out

    @property
    def device(self):
        return self.registers.device


# ── Exercise 4: Cached Video Sampling ────────────────────────────────────────

@t.no_grad()
def sample_video_cached(model, first_frame, actions, n_denoise_steps=30, cfg=0):
    """Generate video with KV-cached frame-autoregressive sampling."""
    B = first_frame.shape[0]
    total_frames = actions.shape[1]
    C, H, W = first_frame.shape[2:]

    max_len = getattr(model, 'max_cache_len', None)
    cache_cond = VideoKVCache(model.n_blocks, max_seq_len=max_len)
    cache_uncond = VideoKVCache(model.n_blocks, max_seq_len=max_len) if cfg > 0 else None

    video = first_frame.clone()

    # Finalize the first frame
    ts_zero = t.zeros(B, 1, device=first_frame.device)
    act_first = actions[:, :1]
    model(first_frame, act_first, ts_zero, kv_cache=cache_cond, cache_mode="finalize")
    if cfg > 0:
        model(first_frame, act_first * 0, ts_zero, kv_cache=cache_uncond, cache_mode="finalize")

    for frame_idx in range(1, total_frames):
        z = t.randn(B, 1, C, H, W, device=first_frame.device, dtype=first_frame.dtype)

        denoise_ts = t.linspace(1, 0, n_denoise_steps + 1, device=first_frame.device)
        denoise_ts = 3 * denoise_ts / (2 * denoise_ts + 1)

        act_frame = actions[:, frame_idx:frame_idx + 1]

        for step_idx in range(n_denoise_steps):
            ts_frame = denoise_ts[step_idx] * t.ones(B, 1, device=first_frame.device)

            v_pred = model(z, act_frame, ts_frame,
                           kv_cache=cache_cond, cache_mode="denoise")

            if cfg > 0:
                v_uncond = model(z, act_frame * 0, ts_frame,
                                 kv_cache=cache_uncond, cache_mode="denoise")
                v_pred = v_uncond + cfg * (v_pred - v_uncond)

            dt = denoise_ts[step_idx] - denoise_ts[step_idx + 1]
            z = z + dt * v_pred

        # Finalize: store clean frame's K,V in cache
        model(z, act_frame, ts_zero, kv_cache=cache_cond, cache_mode="finalize")
        if cfg > 0:
            model(z, act_frame * 0, ts_zero, kv_cache=cache_uncond, cache_mode="finalize")

        video = t.cat([video, z], dim=1)

    return video


# ── Exercise 5: Correctness & Benchmark ─────────────────────────────────────

def verify_correctness(model_naive, model_cached):
    """Verify cached and uncached produce same results."""
    first_frame = t.randn(1, 1, 3, 24, 24, device=device)
    act = t.ones(1, 4, dtype=t.long, device=device)

    t.manual_seed(42)
    video_naive = sample_video(model_naive, first_frame, act, n_denoise_steps=3, cfg=0)

    t.manual_seed(42)
    video_cached = sample_video_cached(model_cached, first_frame, act, n_denoise_steps=3, cfg=0)

    max_diff = (video_naive - video_cached).abs().max().item()
    return max_diff, video_naive, video_cached


# ── Pretrained model loading ─────────────────────────────────────────────────

def pred2frame(z):
    """Convert model output [-1, 1] to uint8 frames [0, 255]."""
    z = z.clamp(-1, 1) * 0.5 + 0.5
    return (z * 255.0).round().byte()


def load_pretrained_pong(model_class=None, device=device):
    """Load the pretrained Pong CausalDiT from HuggingFace.

    Args:
        model_class: Model class to instantiate (default: CachedCausalDiT).
        device: Device to load model onto.

    Returns:
        (model, config) tuple.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("chrisxx/pong", "pong_dit.pt")
    ckpt = t.load(path, map_location=device, weights_only=True)
    config = ckpt["config"]

    if model_class is None:
        model_class = CachedCausalDiT

    model = model_class(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config
