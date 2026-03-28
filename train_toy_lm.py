"""
Train a small byte-level transformer LM on the tutorial text.
Overfits to memorize the corpus so it can generate coherent tutorial snippets.
"""

import torch
import torch as t
from torch import nn
import glob
import os
import sys

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model definition ─────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(t.ones(d))
        self.eps = eps
    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt() * self.scale

class Attention(nn.Module):
    def __init__(self, d=64, n_head=4):
        super().__init__()
        self.n_head, self.d_head = n_head, d // n_head
        self.QKV = nn.Linear(d, 3 * d, bias=False)
        self.O = nn.Linear(d, d, bias=False)
    def forward(self, x):
        b, s, d = x.shape
        q, k, v = [z.view(b, s, self.n_head, self.d_head).transpose(1, 2) for z in self.QKV(x).chunk(3, -1)]
        mask = t.triu(t.ones(s, s, device=x.device, dtype=t.bool), diagonal=1)
        attn = (q @ k.transpose(-2, -1) / self.d_head**0.5).masked_fill(mask, float('-inf')).softmax(-1)
        return self.O((attn @ v).transpose(1, 2).reshape(b, s, d))

class MLP(nn.Module):
    def __init__(self, d=64, exp=4):
        super().__init__()
        self.up = nn.Linear(d, exp * d, bias=False)
        self.gate = nn.Linear(d, exp * d, bias=False)
        self.down = nn.Linear(exp * d, d, bias=False)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.down(self.up(x) * self.act(self.gate(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d=64, n_head=4, exp=4):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d), Attention(d, n_head)
        self.norm2, self.mlp = RMSNorm(d), MLP(d, exp)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size=256, n_ctx=128, d=64, n_head=4, n_layers=4, exp=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(n_ctx, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_head, exp) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.unembed = nn.Linear(d, vocab_size, bias=False)
    def forward(self, tokens):
        b, s = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(t.arange(s, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.norm(x))
print(f"Using device: {device}")

# ── Gather training text ────────────────────────────────────────────────────

def gather_text():
    """Concatenate all tutorial text files into one byte string."""
    files = []
    files += sorted(glob.glob("instructions/**/*.html", recursive=True))
    files += sorted(glob.glob("instructions/**/*.md", recursive=True))
    files += sorted(glob.glob("exercises/**/*.py", recursive=True))

    text = ""
    for f in files:
        with open(f) as fh:
            text += f"\n\n{'='*60}\n# FILE: {f}\n{'='*60}\n\n"
            text += fh.read()
    return text


text = gather_text()
print(f"Training corpus: {len(text):,} characters")

# Byte-level encoding (vocab_size=256)
data = t.tensor(list(text.encode("utf-8")), dtype=t.long)
print(f"Token count: {len(data):,}")

# ── Model ────────────────────────────────────────────────────────────────────

model = Transformer(
    vocab_size=256,
    n_ctx=128,
    d=64,
    n_head=4,
    n_layers=4,
    exp=4,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# ── Training ─────────────────────────────────────────────────────────────────

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
n_steps = 3000
batch_size = 32
seq_len = 128

for step in range(n_steps):
    # Random subsequences
    idx = t.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = t.stack([data[i:i+seq_len] for i in idx]).to(device)
    y = t.stack([data[i+1:i+seq_len+1] for i in idx]).to(device)

    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, 256), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if (step + 1) % 200 == 0:
        print(f"Step {step+1}/{n_steps} | Loss: {loss.item():.4f}", flush=True)

# ── Verify it generates something ────────────────────────────────────────────

@t.no_grad()
def generate(model, prompt_str, max_tokens=200, temperature=0.8):
    tokens = t.tensor(list(prompt_str.encode("utf-8")), dtype=t.long, device=device).unsqueeze(0)
    for _ in range(max_tokens):
        logits = model(tokens[:, -128:])
        next_logits = logits[:, -1, :] / temperature
        next_token = t.multinomial(next_logits.softmax(dim=-1), 1)
        tokens = t.cat([tokens, next_token], dim=1)
    return bytes(tokens[0].cpu().tolist()).decode("utf-8", errors="replace")

print("\n" + "="*60)
print("Sample generation:")
print("="*60)
print(generate(model, "class Attention"))
print("="*60)
print(generate(model, "def flow_matching"))

# ── Save checkpoint ──────────────────────────────────────────────────────────

os.makedirs("checkpoints", exist_ok=True)
save_path = "checkpoints/toy_lm.pt"
t.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size": 256,
        "n_ctx": 128,
        "d": 64,
        "n_head": 4,
        "n_layers": 4,
        "exp": 4,
    },
}, save_path)
print(f"\nCheckpoint saved to {save_path}")
print(f"File size: {os.path.getsize(save_path) / 1024:.1f} KB")
