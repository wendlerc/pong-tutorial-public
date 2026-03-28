"""
Part 3: Frame-Autoregressive Pong — Solutions
===============================================

Reference implementations for all exercises.
Data loading is in ``pong_data``; shared components and model scaffolding are
provided as starter code.

Exercises (focused on what changes from Part 2):
  1. Block-causal mask
  2. CausalDiTBlock.forward — per-frame modulation via modulate() / gate_fn()
  3. CausalDiT.forward — conditioning pipeline, registers, mask, final modulation
  4. Video sampling — autoregressive frame-by-frame denoising
"""

import math
import torch
import torch as t
from torch import nn
import torch.nn.functional as F

from .pong_data import fixed2frame, get_pong_loader, preprocess_pong_episodes

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Starter code: Shared components ──────────────────────────────────────────

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
        self.up_proj.bias.data.zero_()
        self.up_gate = nn.Linear(d_in, d_mid, bias=True)
        self.up_gate.bias.data.zero_()
        self.down = nn.Linear(d_mid, d_out, bias=True)
        self.down.bias.data.zero_()
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
        return (self.coss[:, offset:offset + x.shape[1]] * x
                + self.sins[:, offset:offset + x.shape[1]] * x_perm)


# ── Starter code: Patch / UnPatch (toy-wm layout) ───────────────────────────

class Patch(nn.Module):
    """Patchify video frames: (B, T, C, H, W) -> (B, T, n_patches, d)."""
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
        nn.init.constant_(self.x_embedder.bias, 0)

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
    """Tokens -> frames: (B, T, n_patches, d) -> (B, T, C, H, W)."""
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


# ── Starter code: Modulation helpers ─────────────────────────────────────────

def modulate(x, shift, scale):
    """Per-frame modulation: x is (B, S, d), shift/scale are (B, T_frames, d)."""
    b, s, d = x.shape
    toks_per_frame = s // shift.shape[1]
    x = x.reshape(b, -1, toks_per_frame, d)
    x = x * (1 + scale[:, :, None, :]) + shift[:, :, None, :]
    x = x.reshape(b, s, d)
    return x


def gate_fn(x, g):
    """Per-frame gating: x is (B, S, d), g is (B, T_frames, d)."""
    b, s, d = x.shape
    toks_per_frame = s // g.shape[1]
    x = x.reshape(b, -1, toks_per_frame, d)
    x = x * g[:, :, None, :]
    x = x.reshape(b, s, d)
    return x


# ── Starter code: CausalVideoAttention ──────────────────────────────────────

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


# ── Exercise 1: Block-Causal Attention Mask ──────────────────────────────────

def make_block_causal_mask(n_frames, toks_per_frame, device="cpu"):
    """Create block-causal mask. True = blocked (masked out)."""
    total = n_frames * toks_per_frame
    frame_idx = t.arange(total, device=device) // toks_per_frame
    mask = frame_idx[None, :] > frame_idx[:, None]
    return mask


# ── Starter code: CausalDiTBlock (init provided, forward is Exercise 2) ──────

class CausalDiTBlock(nn.Module):
    """DiT block with causal video attention and per-frame modulation."""
    def __init__(self, d=64, n_head=4, exp=4, rope=None):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.selfattn = CausalVideoAttention(d, n_head, rope=rope)
        self.norm2 = RMSNorm(d)
        self.geglu = GEGLU(d, exp * d, d)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d, bias=True))

    # ── Exercise 2: CausalDiTBlock.forward ───────────────────────────────────

    def forward(self, x, cond, mask=None):
        """
        Args:
            x:    (B, T*toks_per_frame, d) — flat token sequence
            cond: (B, T, d) — per-frame conditioning (NOT per-token!)
            mask: optional block-causal mask
        Returns:
            (B, T*toks_per_frame, d)
        """
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


# ── Starter code: CausalDiT (init provided, forward is Exercise 3) ──────────

class CausalDiT(nn.Module):
    def __init__(self, h=24, w=24, n_actions=4, in_channels=3,
                 patch_size=3, n_blocks=8, d=320, n_head=20, exp=4,
                 T=1000, C=5000, n_registers=1, n_window=30):
        super().__init__()
        self.T = T
        self.n_blocks = n_blocks
        self.n_registers = n_registers
        self.n_window = n_window
        self.in_channels = in_channels
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

    # Provided helpers: tokenize video → flat sequence, and reverse
    def tokenize(self, x):
        """(B, T, C, H, W) → (B, T*toks_per_frame, d) with registers appended."""
        B, T_frames = x.shape[:2]
        z = self.patch(x)  # (B, T, n_patches, d)
        regs = self.registers[None, None].expand(B, T_frames, -1, -1)
        zr = t.cat([z, regs], dim=2)  # (B, T, toks_per_frame, d)
        self._dur, self._seq = zr.shape[1], zr.shape[2]
        return zr.reshape(B, -1, zr.shape[-1])  # (B, T*toks_per_frame, d)

    def detokenize(self, zr):
        """(B, T*toks_per_frame, d) → (B, T, C, H, W) stripping registers."""
        B = zr.shape[0]
        zr = zr.reshape(B, self._dur, self._seq, -1)
        return self.unpatch(zr[:, :, :-self.n_registers])

    # ── Exercise 3: CausalDiT.forward ────────────────────────────────────────

    def forward(self, x, actions, ts):
        """
        Args:
            x:       (B, T, C, H, W) video frames
            actions: (B, T) action indices per frame
            ts:      (B, T) timestep per frame (float in [0, 1])
        Returns:
            (B, T, C, H, W) predicted velocity per frame
        """
        B, T_frames, C, H, W = x.shape

        # 1. Build per-frame conditioning (B, T, d)
        a = self.action_emb(actions)
        ts_scaled = (ts.float() * (self.T - 1)).long()
        cond = self.time_emb_mixer(self.time_emb(ts_scaled)) + a

        # 2. Tokenize (provided helper: patch + append registers + flatten)
        zr = self.tokenize(x)  # (B, T*toks_per_frame, d)

        # 3. Block-causal mask + transformer blocks
        mask = make_block_causal_mask(T_frames, self.toks_per_frame, device=x.device)
        for block in self.blocks:
            zr = block(zr, cond, mask=mask)

        # 4. Final modulation
        mu, sigma = self.modulation(cond).chunk(2, dim=-1)
        zr = modulate(self.norm(zr), mu, sigma)

        # 5. Detokenize (provided helper: strip registers + unpatch)
        return self.detokenize(zr)

    @property
    def device(self):
        return self.registers.device


# ── Starter code: Muon optimizer setup ───────────────────────────────────────

def get_muon(model, lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=1e-5):
    """Create Muon optimizer with param group splitting."""
    from muon import SingleDeviceMuonWithAuxAdam

    body_weights = list(model.blocks.parameters())
    body_ids = {id(p) for p in body_weights}
    other_weights = [p for p in model.parameters() if id(p) not in body_ids]

    hidden_weights = [p for p in body_weights if p.ndim >= 2]
    hidden_gains_biases = [p for p in body_weights if p.ndim < 2]

    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=lr1, weight_decay=weight_decay),
        dict(params=hidden_gains_biases + other_weights, use_muon=False,
             lr=lr2, betas=betas, weight_decay=weight_decay),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)


# ── Starter code: Training loop ──────────────────────────────────────────────

def train_pong_model(model, train_loader, n_steps=2500, lr1=0.02, lr2=3e-4, action_dropout=0.2):
    """Train the Pong video model with diffusion forcing."""
    optimizer = get_muon(model, lr1=lr1, lr2=lr2)

    running_loss = 0
    train_iter = iter(train_loader)

    for step in range(n_steps):
        try:
            frames, actions = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            frames, actions = next(train_iter)

        frames = frames.to(device)
        actions = actions.to(device)
        B, T_frames, C, H, W = frames.shape

        # Truncate to model's window size
        frames = frames[:, :model.n_window]
        actions = actions[:, :model.n_window]
        T_frames = frames.shape[1]

        # Diffusion forcing loss
        ts = F.sigmoid(t.randn(B, T_frames, device=frames.device))
        z = t.randn_like(frames)
        v_true = frames - z
        x_t = frames - ts[:, :, None, None, None] * v_true

        # Action dropout for CFG
        actions = actions.clone()
        drop_mask = t.rand(B, T_frames, device=frames.device) < action_dropout
        actions[drop_mask] = 0

        v_pred = model(x_t, actions, ts)
        loss = F.mse_loss(v_pred, v_true)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        running_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{n_steps} | Loss: {running_loss / 100:.4f}")
            running_loss = 0


# ── Exercise 4: Video Sampling ───────────────────────────────────────────────

@t.no_grad()
def sample_video(model, first_frame, actions, n_denoise_steps=30, cfg=0):
    """Generate video autoregressively, one frame at a time."""
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
