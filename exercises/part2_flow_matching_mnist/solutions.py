"""
Part 2: Flow Matching on MNIST — Solutions
==========================================

Reference implementations for all exercises.
Exercises 7-8 (training loops) require MNIST data.

RMSNorm, Attention, MLP, NumEmbedding are provided as starter code.
Exercises:
  1. Patch — image to token sequences
  2. UnPatch — token sequences back to images
  3. DiTBlock — AdaLN conditioning via modulation
  4. Full DiT — assembling the model
  5. Flow matching loss — adapting Part 1's loss to images
  6. Euler sampling — adapting Part 1's sampling to images
  7-8. Training loops — unconditional and class-conditional with CFG
"""

import torch
import torch as t
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Starter code (provided) ─────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d=32, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(t.ones((1, 1, d)))
        self.eps = eps

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True)).sqrt() + self.eps
        return x / rms * self.scale


class Attention(nn.Module):
    def __init__(self, d=32, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.d_head = d // n_head
        self.QKV = nn.Linear(d, 3 * d, bias=False)
        self.O = nn.Linear(d, d, bias=False)
        self.normq = RMSNorm(self.d_head)
        self.normk = RMSNorm(self.d_head)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q = q.reshape(b, s, self.n_head, self.d_head)
        k = k.reshape(b, s, self.n_head, self.d_head)
        v = v.reshape(b, s, self.n_head, self.d_head)
        q = self.normq(q)
        k = self.normk(k)
        attn = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
        attn = attn.softmax(dim=-1)
        z = attn @ v.permute(0, 2, 1, 3)
        z = z.permute(0, 2, 1, 3).reshape(b, s, self.d)
        return self.O(z)


class MLP(nn.Module):
    def __init__(self, d=32, exp=2):
        super().__init__()
        self.up = nn.Linear(d, exp * d, bias=False)
        self.gate = nn.Linear(d, exp * d, bias=False)
        self.down = nn.Linear(exp * d, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.up(x) * self.act(self.gate(x)))


class NumEmbedding(nn.Module):
    """Sinusoidal embeddings for integer values (timesteps)."""
    def __init__(self, n_max, d=32, C=500):
        super().__init__()
        thetas = C ** (-t.arange(0, d // 2) / (d // 2))
        thetas = t.arange(0, n_max)[:, None].float() @ thetas[None, :]
        sins = t.sin(thetas)
        coss = t.cos(thetas)
        self.register_buffer("E", t.cat([sins, coss], dim=1))

    def forward(self, x):
        return self.E[x]


# ── Exercise 1: Patch (Image → Tokens) ──────────────────────────────────────

class Patch(nn.Module):
    """Convert images to patch token sequences using strided convolution."""
    def __init__(self, patch_size=4, in_channels=1, d=32):
        super().__init__()
        self.d = d
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, d, kernel_size=5, padding=2, stride=patch_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv(x)                           # (b, d, h//ps, w//ps)
        x = x.permute(0, 2, 3, 1)                  # (b, h//ps, w//ps, d)
        x = x.reshape(b, -1, self.d)               # (b, n_patches, d)
        return x


# ── Exercise 2: UnPatch (Tokens → Image) ────────────────────────────────────

class UnPatch(nn.Module):
    """Convert patch token sequences back to images."""
    def __init__(self, patch_size=4, d=32, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.up = nn.Linear(d, patch_size ** 2 * out_channels)

    def forward(self, x):
        b, s, d = x.shape
        x = self.up(x)
        w = int(s ** 0.5)
        h = w
        ps = self.patch_size
        c = self.out_channels
        x = x.reshape(b, h, w, c, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(b, c, h * ps, w * ps)
        return x


# ── Exercise 3: DiT Block with AdaLN ────────────────────────────────────────

class DiTBlock(nn.Module):
    def __init__(self, d=32, n_head=4, exp=2):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_head)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, exp)
        self.modulate = nn.Linear(d, 6 * d)

    def forward(self, x, c):
        scale1, bias1, gate1, scale2, bias2, gate2 = self.modulate(c).chunk(6, dim=-1)

        residual = x
        x = self.norm1(x) * (1 + scale1[:, None, :]) + bias1[:, None, :]
        x = self.attn(x) * gate1[:, None, :]
        x = residual + x

        residual = x
        x = self.norm2(x) * (1 + scale2[:, None, :]) + bias2[:, None, :]
        x = self.mlp(x) * gate2[:, None, :]
        x = residual + x
        return x


# ── Exercise 4: Full DiT Model ──────────────────────────────────────────────

class DiT(nn.Module):
    def __init__(self, h=28, w=28, n_classes=11, in_channels=1,
                 patch_size=2, n_blocks=6, d=192, n_head=12, exp=2, T=1000):
        super().__init__()
        self.T = T
        self.patch = Patch(patch_size, in_channels, d)
        self.n_seq = (h // patch_size) * (w // patch_size)
        self.pe = nn.Parameter(t.randn(1, self.n_seq, d) * d ** -0.5)
        self.te = NumEmbedding(T, d)
        self.ce = nn.Embedding(n_classes, d)
        self.act = nn.SiLU()
        self.blocks = nn.ModuleList([DiTBlock(d, n_head, exp) for _ in range(n_blocks)])
        self.norm = RMSNorm(d)
        self.modulate = nn.Linear(d, 2 * d)
        self.unpatch = UnPatch(patch_size, d, in_channels)

    def forward(self, x, c, ts):
        ts_int = t.minimum((ts * self.T).to(t.int64), t.tensor(self.T - 1, device=ts.device))
        cond = self.act(self.te(ts_int) + self.ce(c))
        x = self.patch(x) + self.pe
        for block in self.blocks:
            x = block(x, cond)
        scale, bias = self.modulate(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None, :]) + bias[:, None, :]
        x = self.unpatch(x)
        return x

    @property
    def device(self):
        return self.pe.device

    @property
    def dtype(self):
        return self.pe.dtype


# ── Exercise 6: Euler Sampling ─────────────────────────────────────────────

@t.no_grad()
def sample(model, z, y, n_steps=30, cfg=0):
    """Generate images from noise using Euler integration."""
    ts = t.linspace(1, 0, n_steps + 1, device=z.device, dtype=z.dtype)
    ts = 3 * ts / (2 * ts + 1)  # SD3 scheduler
    for idx in range(n_steps):
        t_batch = ts[idx] * t.ones(z.shape[0], dtype=z.dtype, device=z.device)
        v_pred = model(z, y, t_batch)
        if cfg > 0:
            v_uncond = model(z, y * 0, t_batch)
            v_pred = v_uncond + cfg * (v_pred - v_uncond)
        z = z + (ts[idx] - ts[idx + 1]) * v_pred
    return z


# ── Starter code: Muon optimizer setup ──────────────────────────────────────

def get_muon(model, lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=1e-5):
    """Create Muon optimizer with param group splitting.

    Muon uses momentum-based updates for large weight matrices (ndim >= 2)
    and falls back to Adam for biases, gains, and embeddings (ndim < 2).

    Args:
        model: nn.Module with a .blocks attribute (transformer blocks)
        lr1: learning rate for hidden weights (Muon)
        lr2: learning rate for gains/biases/embeddings (Adam)
        betas: Adam betas for the auxiliary optimizer
        weight_decay: L2 regularization
    """
    from muon import SingleDeviceMuonWithAuxAdam

    # Split: block params vs everything else (embeddings, final layers)
    body_weights = list(model.blocks.parameters())
    body_ids = {id(p) for p in body_weights}
    other_weights = [p for p in model.parameters() if id(p) not in body_ids]

    # Within blocks: large weight matrices (ndim >= 2) use Muon,
    # biases and gains (ndim < 2) use Adam
    hidden_weights = [p for p in body_weights if p.ndim >= 2]
    hidden_gains_biases = [p for p in body_weights if p.ndim < 2]

    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=lr1, weight_decay=weight_decay),
        dict(params=hidden_gains_biases + other_weights, use_muon=False,
             lr=lr2, betas=betas, weight_decay=weight_decay),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)


# ── Exercise 5: Training (with CFG) ──────────────────────────────────────

def train(model, train_loader, n_epochs=10, lr1=0.02, lr2=3e-4, label_dropout=0.2):
    """Train with class labels, CFG dropout, and logit-normal timestep sampling."""
    optimizer = get_muon(model, lr1=lr1, lr2=lr2)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            c = labels + 1  # shift: 0 reserved for unconditional
            drop_mask = t.rand(c.shape[0], device=device) < label_dropout
            c[drop_mask] = 0

            # Flow matching loss (inline)
            ts = F.sigmoid(t.randn(images.shape[0], device=images.device))  # logit-normal
            z = t.randn_like(images)
            v_true = images - z
            x_t = images - ts[:, None, None, None] * v_true
            v_pred = model(x_t, c, ts)
            loss = F.mse_loss(v_pred, v_true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")
