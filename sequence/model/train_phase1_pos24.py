"""pos24 copy of train_phase1_arms.py: 24 train-only position inputs (see build_pos24_maps.py). Original unchanged.

Phase 1 sequence-architecture arms on all maps (pre-registered, decision 6fb5578a). Validation only for every decision. The shared loader (tta.load_all) builds feature and label arrays for every split, test included; the trainer never uses test rows for training, batch order, validation or checkpoint selection.

Arms, three seeds each:
  gru      GRU control, lookback 20 (nn.GRU 42 -> 64)
  gru_lb1  history control: the same GRU reading only the current second (lookback 1)
  lstm     nn.LSTM 42 -> 53 (parameter-matched to the GRU within about 1 percent)
  tcn      42 -> 40 projection, 4 residual causal blocks, kernel 3, dilations 1, 2, 4, 8 (receptive field 31 s)
  tf       causal transformer, pre-norm: 42 -> 32 projection, learned positions, 2 layers, 4 heads, FFN 64 GELU,
           causal mask plus key-padding mask (the current second is always attendable), 500-step warmup

Shared and unchanged across arms: per-slot encoder (18 -> 64 -> 64, ReLU), side sums over 5 slots, team MLP (128 -> 64 -> 32),
shared state appended (10 values), head (d -> 64 -> 1), BCE with logits, Adam lr 1e-3, no weight decay, no dropout, batch 2048.
The GRU arm is the same network as train_temporal_round_model.py (TemporalRoundModel) with in_dim 18 and state dim 10.

Budget: a fixed number of optimizer steps (default 12,000) for every arm. Validation every 1,000 steps; the best checkpoint by
validation log loss is kept. Outputs per run: <arm>_seed<s>_val_preds.parquet (best validation checkpoint) and <arm>_seed<s>_final_val_preds.parquet (the model at the last step, no checkpoint selection) and <arm>_seed<s>_metrics.json (the best step and the final-step log loss are logged).
Completed runs are skipped on re-launch.

--dist appends the four straight-line distance features from the TRAIN-ONLY site centres (data/diagnostics/allmaps_geo_trainsites,
written by sitecheck_geo.py; parity passed) to the shared state, so the state is 14 values instead of 10. The existing --dist in
train_temporal_allmaps.py fits its centres on all matches, including test, so it is not used here. Dist runs need their own --out_dir.
With --dist the encoder output is 46 values and the shared state 14; the widths quoted above are the default ones.

  ...tabicl python.exe -I -X utf8 -u scripts\\model\\train_phase1_arms.py --arms gru,gru_lb1,lstm,tcn,tf --seeds 0,1,2
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import train_temporal_round_model as ttr  # noqa: E402
import train_temporal_allmaps as tta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "diagnostics" / "phase1"
SLOTS_PER_SIDE = ttr.SLOTS_PER_SIDE  # 5
N_FEAT = tta.N_FEAT  # 18 per slot: 6 state + 12 loadout
ARMS = {"gru": 20, "gru_lb1": 1, "lstm": 20, "tcn": 20, "tf": 20, "mingru": 20, "minlstm": 20, "lru": 20,
        "xlstm": 20, "rwkv7": 20, "griffin": 20, "mamba2": 20,
        "xlstm_fast": 20, "xlstm_mh": 20, "gdn": 20, "gru2": 20, "tf2": 20, "xlstm_sig": 20, "xlstm_blk": 20, "gdn2": 20}
N_DIST = 24  # pos24: the 24 train-only position inputs (pin 66bca815, amendment 7fe1d12b)
TRAIN_SITES = ROOT / "data" / "diagnostics" / "allmaps_pos24_trainsites_grid"


class Encoder(nn.Module):
    """Per-slot encoder and team pooling, identical to TemporalRoundModel (without the GRU)."""

    def __init__(self, in_dim, state_dim, latent=32, hidden=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.team = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.out_dim = latent + state_dim

    def forward(self, Xw, Sw, mask):
        alive = Xw[..., :1]
        h = self.phi(Xw) * alive
        ct = h[:, :, :SLOTS_PER_SIDE].sum(dim=2)
        t = h[:, :, SLOTS_PER_SIDE:].sum(dim=2)
        z = self.team(torch.cat([ct, t], dim=-1)) * mask.unsqueeze(-1)
        return torch.cat([z, Sw * mask.unsqueeze(-1)], dim=-1)


class GRUSeq(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x, mask):
        out, _ = self.rnn(x)
        return out[:, -1]


class LSTMSeq(nn.Module):
    def __init__(self, d_in, hidden=53):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x, mask):
        out, _ = self.rnn(x)
        return out[:, -1]


class CausalBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(c, c, kernel_size=3, dilation=dilation)

    def forward(self, x):  # x (B, C, K)
        return x + F.relu(self.conv(F.pad(x, (self.pad, 0))))


class TCNSeq(nn.Module):
    def __init__(self, d_in, ch=40):
        super().__init__()
        self.proj = nn.Linear(d_in, ch)
        self.blocks = nn.Sequential(*[CausalBlock(ch, d) for d in (1, 2, 4, 8)])
        self.out_dim = ch

    def forward(self, x, mask):
        h = self.blocks(self.proj(x).transpose(1, 2))
        return h[:, :, -1]


class TFBlock(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, allowed):  # allowed (B, 1, K, K) bool
        B, K, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, K, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=allowed)
        x = x + self.proj(a.transpose(1, 2).reshape(B, K, D))
        return x + self.ff(self.ln2(x))


class TFSeq(nn.Module):
    def __init__(self, d_in, K, d=32, layers=2, heads=4, ff=64):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TFBlock(d, heads, ff) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x, mask):
        B, K, _ = x.shape
        h = self.inp(x) + self.pos[:K]
        causal = torch.tril(torch.ones(K, K, dtype=torch.bool, device=x.device))
        valid_key = mask[:, None, :] > 0  # (B, 1, K): padded seconds are not attendable
        eye = torch.eye(K, dtype=torch.bool, device=x.device)[None]  # every row can attend to itself
        allowed = ((causal[None] & valid_key) | eye)[:, None]  # (B, 1, K, K)
        for blk in self.blocks:
            h = blk(h, allowed)
        return self.ln(h)[:, -1]


class MinGRUSeq(nn.Module):
    """minGRU (Feng et al. 2024): gates depend only on the input, so the recurrence is a simple linear update. Sized so its
    sequence parameters match the GRU control (about 25.3k at the default input width)."""
    def __init__(self, d_in, hidden=189):
        super().__init__()
        self.lin_z = nn.Linear(d_in, hidden)
        self.lin_h = nn.Linear(d_in, hidden)
        self.out_dim = hidden

    def forward(self, x, mask):
        z = torch.sigmoid(self.lin_z(x))
        hc = self.lin_h(x)
        g = torch.where(hc >= 0, hc + 0.5, torch.sigmoid(hc))
        h = torch.zeros_like(z[:, 0])
        for t in range(x.shape[1]):
            h = (1 - z[:, t]) * h + z[:, t] * g[:, t]
        return h


class MinLSTMSeq(nn.Module):
    """minLSTM (Feng et al. 2024): forget and input gates normalised to sum to one, input-only gates. Sized to the GRU control."""
    def __init__(self, d_in, hidden=126):
        super().__init__()
        self.lin_f = nn.Linear(d_in, hidden)
        self.lin_i = nn.Linear(d_in, hidden)
        self.lin_h = nn.Linear(d_in, hidden)
        self.out_dim = hidden

    def forward(self, x, mask):
        f = torch.sigmoid(self.lin_f(x))
        i = torch.sigmoid(self.lin_i(x))
        hc = self.lin_h(x)
        fp = f / (f + i + 1e-8)
        ip = i / (f + i + 1e-8)
        h = torch.zeros_like(f[:, 0])
        for t in range(x.shape[1]):
            h = fp[:, t] * h + ip[:, t] * hc[:, t]
        return h


class LRUSeq(nn.Module):
    """LRU (Orvieto et al. 2023): diagonal complex recurrence with exponential-decay parametrisation, stable by construction.
    N complex states; output is the real part of a linear read-out plus a direct term. Sized to the GRU control."""
    def __init__(self, d_in, N=80, out=64, r_min=0.9, r_max=0.999, max_phase=6.283):
        super().__init__()
        u = torch.rand(N)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u * (r_max ** 2 - r_min ** 2) + r_min ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * torch.rand(N)))
        self.B_re = nn.Parameter(torch.randn(N, d_in) / (d_in ** 0.5))
        self.B_im = nn.Parameter(torch.randn(N, d_in) / (d_in ** 0.5))
        self.C_re = nn.Parameter(torch.randn(out, N) / (N ** 0.5))
        self.C_im = nn.Parameter(torch.randn(out, N) / (N ** 0.5))
        self.D = nn.Parameter(torch.randn(out, d_in) / (d_in ** 0.5))
        self.out_dim = out

    def forward(self, x, mask):
        mag = torch.exp(-torch.exp(self.nu_log))
        lam_re = mag * torch.cos(torch.exp(self.theta_log))
        lam_im = mag * torch.sin(torch.exp(self.theta_log))
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))[None, None]
        u_re = gamma * (x @ self.B_re.t())
        u_im = gamma * (x @ self.B_im.t())
        h_re = torch.zeros_like(u_re[:, 0])
        h_im = torch.zeros_like(u_im[:, 0])
        for t in range(x.shape[1]):
            n_re = lam_re * h_re - lam_im * h_im + u_re[:, t]
            n_im = lam_re * h_im + lam_im * h_re + u_im[:, t]
            h_re, h_im = n_re, n_im
        y = h_re @ self.C_re.t() - h_im @ self.C_im.t() + x[:, -1] @ self.D.t()
        return y


class MLSTMSeq(nn.Module):
    """mLSTM (Beck et al. 2024, xLSTM): matrix memory C, normaliser n, exponential input gate and sigmoid forget gate in log
    space with the paper's stabiliser m. Single head of width d (simplification: the paper uses several heads). Sized to the GRU
    control at d=94 (about 25.3k sequence parameters)."""
    def __init__(self, d_in, d=94):
        super().__init__()
        self.d = d
        self.lin_qkv = nn.Linear(d_in, 3 * d)
        self.lin_gates = nn.Linear(d_in, 2 + d)  # input gate, forget gate, output gate (d)
        self.out_dim = d

    def forward(self, x, mask):
        B, T, _ = x.shape
        d = self.d
        qkv = self.lin_qkv(x)
        q, k, v = qkv[..., :d], qkv[..., d:2 * d] / (d ** 0.5), qkv[..., 2 * d:]
        g = self.lin_gates(x)
        i_pre, log_f, o = g[..., 0], F.logsigmoid(g[..., 1]), torch.sigmoid(g[..., 2:])
        C = x.new_zeros(B, d, d)
        n = x.new_zeros(B, d)
        m = x.new_full((B,), -1e30)
        for t in range(T):
            m_new = torch.maximum(log_f[:, t] + m, i_pre[:, t])
            fp = torch.exp(log_f[:, t] + m - m_new)
            ip = torch.exp(i_pre[:, t] - m_new)
            C = fp[:, None, None] * C + ip[:, None, None] * (v[:, t, :, None] * k[:, t, None, :])
            n = fp[:, None] * n + ip[:, None] * k[:, t]
            m = m_new
        q_T = q[:, -1]
        num = torch.einsum("bij,bj->bi", C, q_T)
        den = torch.maximum(torch.abs((n * q_T).sum(-1)), torch.exp(-m))  # stabilised normaliser
        return o[:, -1] * num / den[:, None]


class RWKV7Seq(nn.Module):
    """RWKV-7 (Peng et al. 2025) WKV core, one head of width d: generalised delta-rule state update
    S_t = S_{t-1} diag(w_t) - (S_{t-1} kh_t)(a_t * kh_t)^T + v_t k_t^T, with unit-norm kh_t, in-context learning rate a_t in (0,1)
    and decay w_t in (0,1); output S_T r_T, then LayerNorm. Simplifications: no token shift, no value residual, no bonus term.
    Sized to the GRU control at d=63."""
    def __init__(self, d_in, d=63):
        super().__init__()
        self.d = d
        self.lin = nn.Linear(d_in, 6 * d)  # r, k, v, w, a, kappa
        self.ln = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x, mask):
        B, T, _ = x.shape
        d = self.d
        r, k, v, w_pre, a_pre, kap = self.lin(x).split(d, dim=-1)
        w = torch.exp(-0.6065306597 * torch.sigmoid(w_pre))  # decay in (0.545, 1)
        a = torch.sigmoid(a_pre)
        kh = F.normalize(kap, dim=-1)
        S = x.new_zeros(B, d, d)
        for t in range(T):
            Skh = torch.einsum("bij,bj->bi", S, kh[:, t])
            S = S * w[:, t, None, :] - Skh[:, :, None] * (a[:, t] * kh[:, t])[:, None, :] + v[:, t, :, None] * k[:, t, None, :]
        return self.ln(torch.einsum("bij,bj->bi", S, r[:, -1]))


class GriffinSeq(nn.Module):
    """Griffin RG-LRU block (De et al. 2024): h_t = a_t h_{t-1} + sqrt(1 - a_t^2) (i_t x_t), a_t = sigmoid(Lambda)^(c r_t),
    r_t and i_t sigmoid gates of the projected input, times a GeLU branch. Simplifications: dense gates (the paper uses
    block-diagonal), no temporal conv. Sized to the GRU control at d=83."""
    C_SCALE = 8.0

    def __init__(self, d_in, d=83):
        super().__init__()
        self.lin_x = nn.Linear(d_in, d)
        self.lin_g = nn.Linear(d_in, d)
        self.gate_a = nn.Linear(d, d)
        self.gate_i = nn.Linear(d, d)
        self.lam = nn.Parameter(torch.logit(0.9 + 0.098 * torch.rand(d)))
        self.out_dim = d

    def forward(self, x, mask):
        xt = self.lin_x(x)
        g = F.gelu(self.lin_g(x))
        r = torch.sigmoid(self.gate_a(xt))
        i = torch.sigmoid(self.gate_i(xt))
        log_a = self.C_SCALE * r * F.logsigmoid(self.lam)  # <= 0
        a = torch.exp(log_a)
        u = torch.sqrt(torch.clamp(1 - torch.exp(2 * log_a), min=1e-6)) * (i * xt)
        h = torch.zeros_like(xt[:, 0])
        for t in range(x.shape[1]):
            h = a[:, t] * h + u[:, t]
        return g[:, -1] * h


class Mamba2Seq(nn.Module):
    """Mamba-2 SSD block, one head (Dao and Gu 2024): input-dependent scalar decay exp(dt*A), state size N, depthwise causal conv
    of width 4, SiLU gate, output projection. Sequential scan over the lookback (no chunked SSD kernel). Sized to the GRU control
    at d=97 (about 25.2k sequence parameters)."""
    def __init__(self, d_in, d=97, N=16, K=4):
        super().__init__()
        self.d, self.N, self.K = d, N, K
        self.in_proj = nn.Linear(d_in, 2 * d + 2 * N + 1)  # x, z, B, C, dt
        self.conv = nn.Conv1d(d, d, K, groups=d)
        self.A_log = nn.Parameter(torch.log(1 + 15 * torch.rand(1)))  # A in (-16, -1)
        self.D = nn.Parameter(torch.ones(1))
        self.dt_bias = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Linear(d, d)
        self.out_dim = d

    def forward(self, x, mask):
        B, T, _ = x.shape
        d, N = self.d, self.N
        xs, z, Bm, Cm, dt_pre = self.in_proj(x).split([d, d, N, N, 1], dim=-1)
        xc = F.silu(self.conv(F.pad(xs.transpose(1, 2), (self.K - 1, 0)))).transpose(1, 2)  # causal, (B, T, d)
        dt = F.softplus(dt_pre + self.dt_bias)  # (B, T, 1)
        decay = torch.exp(dt * (-torch.exp(self.A_log)))
        h = x.new_zeros(B, N, d)
        for t in range(T):
            h = decay[:, t, :, None] * h + dt[:, t, :, None] * (Bm[:, t, :, None] * xc[:, t, None, :])
        y = torch.einsum("bn,bnd->bd", Cm[:, -1], h) + self.D * xc[:, -1]
        return self.out_proj(y * F.silu(z[:, -1]))


class MLSTMFastSeq(nn.Module):
    """mLSTM with the same recurrence as MLSTMSeq, evaluated in closed form at the last step (no time loop). With m_T the running
    stabiliser, the weights w_s = exp(F_T - F_s + i_s - m_T) (F = cumulative sum of log forget gates) reproduce C_T and n_T exactly,
    so h_T = o_T * sum_s w_s (k_s . q_T) v_s / max(|sum_s w_s (k_s . q_T)|, exp(-m_T)). heads > 1 gives each head its own memory and
    gates (multi-head mLSTM as in the paper). Verified against the loop form on CPU (scratchpad/pos24/test_new_arms.py)."""
    def __init__(self, d_in, d=94, heads=1, input_gate="exp"):
        super().__init__()
        assert d % heads == 0 and input_gate in ("exp", "sigmoid")
        self.d, self.H, self.dh = d, heads, d // heads
        self.input_gate = input_gate
        self.lin_qkv = nn.Linear(d_in, 3 * d)
        self.lin_gates = nn.Linear(d_in, 2 * heads + d)  # per-head input and forget gates, per-dimension output gate
        self.out_dim = d

    def forward(self, x, mask):
        B, T, _ = x.shape
        H, dh, d = self.H, self.dh, self.d
        qkv = self.lin_qkv(x)
        q = qkv[..., :d].reshape(B, T, H, dh)
        k = (qkv[..., d:2 * d] / (dh ** 0.5)).reshape(B, T, H, dh)
        v = qkv[..., 2 * d:].reshape(B, T, H, dh)
        g = self.lin_gates(x)
        i_pre = g[..., :H]
        if self.input_gate == "sigmoid":  # TFLA variant: input gate is a sigmoid, so its log is logsigmoid
            i_pre = F.logsigmoid(i_pre)
        log_f = F.logsigmoid(g[..., H:2 * H])
        o = torch.sigmoid(g[..., 2 * H:]).reshape(B, T, H, dh)
        F_cum = torch.cumsum(log_f, dim=1)
        logw = F_cum[:, -1:] - F_cum + i_pre               # (B, T, H)
        m_T = logw.max(dim=1, keepdim=True).values         # (B, 1, H)
        w = torch.exp(logw - m_T)                          # (B, T, H), largest weight is 1
        qT = q[:, -1]                                      # (B, H, dh)
        kq = (k * qT[:, None]).sum(-1)                     # (B, T, H)
        num = torch.einsum("bth,bth,bthd->bhd", w, kq, v)  # (B, H, dh)
        den = torch.maximum((w * kq).sum(1).abs(), torch.exp(-m_T[:, 0]))  # (B, H)
        h = o[:, -1] * num / den[..., None]
        return h.reshape(B, d)


class MLSTMMultiHeadSeq(MLSTMFastSeq):
    """Multi-head mLSTM (paper form): 2 heads of width 47 at d = 94, about 25.5k sequence parameters."""
    def __init__(self, d_in, d=94, heads=2):
        super().__init__(d_in, d=d, heads=heads)


class GatedDeltaNetSeq(nn.Module):
    """Gated DeltaNet (Yang et al., 2024, arXiv 2412.06464), one head of width d. Recurrence on the state S (d x d):
    S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T, with L2-normalised q and k, decay alpha_t = exp(-A softplus(.))
    in (0,1) and write strength beta_t = sigmoid(.). Read-out S_T q_T, RMS-normalised, gated by SiLU of a linear map of x_T, then
    an output projection. Sized to the GRU control at d = 73 (about 25.2k sequence parameters)."""
    def __init__(self, d_in, d=73):
        super().__init__()
        self.d = d
        self.lin_qkv = nn.Linear(d_in, 3 * d)
        self.lin_alpha = nn.Linear(d_in, 1)
        self.A_log = nn.Parameter(torch.log(torch.empty(1).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.zeros(1))
        self.lin_beta = nn.Linear(d_in, 1)
        self.lin_gate = nn.Linear(d_in, d)
        self.gain = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d)
        self.out_dim = d

    def forward(self, x, mask):
        B, T, _ = x.shape
        d = self.d
        q, k, v = self.lin_qkv(x).split(d, dim=-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        alpha = torch.exp(-torch.exp(self.A_log) * F.softplus(self.lin_alpha(x) + self.dt_bias))  # (B, T, 1)
        beta = torch.sigmoid(self.lin_beta(x))                                                   # (B, T, 1)
        S = x.new_zeros(B, d, d)
        for t in range(T):
            a, b = alpha[:, t], beta[:, t]                  # (B, 1)
            kt, vt = k[:, t], v[:, t]                       # (B, d)
            Sk = torch.einsum("bij,bj->bi", S, kt)          # (B, d)
            S = a[:, :, None] * S - (a * b)[:, :, None] * Sk[:, :, None] * kt[:, None, :] + b[:, :, None] * vt[:, :, None] * kt[:, None, :]
        o = torch.einsum("bij,bj->bi", S, q[:, -1])
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain
        o = o * F.silu(self.lin_gate(x)[:, -1])
        return self.out(o)


class XLSTMBlockSeq(nn.Module):
    """xLSTM 7B revised block (Beck et al., 2025, arXiv 2503.13427): an mLSTM mixer, then RMS normalisation and a gated (SwiGLU)
    MLP with a residual. Read at the last lookback step. Sized to the 25k track: mixer width 46 (one head), MLP hidden 92,
    about 25.4k sequence parameters. Simplification: the block is read at the last step only; no stacking."""
    def __init__(self, d_in, d=46, h=92):
        super().__init__()
        self.mixer = MLSTMFastSeq(d_in, d, heads=1)
        self.gain = nn.Parameter(torch.ones(d))
        self.w_gate = nn.Linear(d, h)
        self.w_up = nn.Linear(d, h)
        self.w_down = nn.Linear(h, d)
        self.out_dim = d

    def forward(self, x, mask):
        hm = self.mixer(x, mask)
        u = hm * torch.rsqrt(hm.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain
        return hm + self.w_down(F.silu(self.w_gate(u)) * self.w_up(u))


class ChannelGatedDeltaSeq(nn.Module):
    """Channel-gated delta rule, the GDN-2 family as described in the survey of 6 Nov 2026 (arXiv 2605.22791; its exact equations
    were not retrieved from the paper body, so this is the form the pin records). One head of width d = 55. Per-channel decay
    alpha_t and per-channel write gate beta_t in (0,1) replace the scalar gates of Gated DeltaNet:
      S_t = S_{t-1} diag(alpha_t) + (v_t - S_{t-1} diag(alpha_t) k_t) (beta_t * k_t)^T,  read-out S_T q_T.
    With scalar gates this is Gated DeltaNet. q and k are L2-normalised. Sized to the GRU control (about 25.2k sequence parameters)."""
    def __init__(self, d_in, d=55):
        super().__init__()
        self.d = d
        self.lin_qkv = nn.Linear(d_in, 3 * d)
        self.lin_alpha = nn.Linear(d_in, d)
        self.lin_beta = nn.Linear(d_in, d)
        self.lin_gate = nn.Linear(d_in, d)
        self.gain = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d)
        self.out_dim = d

    def forward(self, x, mask):
        B, T, _ = x.shape
        d = self.d
        q, k, v = self.lin_qkv(x).split(d, dim=-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        alpha = torch.sigmoid(self.lin_alpha(x))  # (B, T, d)
        beta = torch.sigmoid(self.lin_beta(x))    # (B, T, d)
        S = x.new_zeros(B, d, d)
        for t in range(T):
            S_dec = S * alpha[:, t, None, :]                          # S diag(alpha)
            resid = v[:, t] - torch.einsum("bij,bj->bi", S_dec, k[:, t])
            S = S_dec + resid[:, :, None] * (beta[:, t] * k[:, t])[:, None, :]
        o = torch.einsum("bij,bj->bi", S, q[:, -1])
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain
        o = o * F.silu(self.lin_gate(x)[:, -1])
        return self.out(o)


class GRUStackSeq(nn.Module):
    """Stacked GRU (cuDNN): `layers` GRU layers of width `hidden`, read at the last lookback step. Deeper arm for the capacity track
    (default 2 layers of width 94, about 99k sequence parameters)."""
    def __init__(self, d_in, hidden=94, layers=2):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True)
        self.out_dim = hidden

    def forward(self, x, mask):
        out, _ = self.gru(x)
        return out[:, -1]


class ArmModel(nn.Module):
    def __init__(self, arch, state_dim, K, cfg=None):
        super().__init__()
        cfg = cfg or {}  # empty cfg reproduces the phase-1 arms exactly
        self.enc = Encoder(N_FEAT, state_dim)
        d_in = self.enc.out_dim
        if arch in ("gru", "gru_lb1"):
            self.seq = GRUSeq(d_in, cfg.get("width", 64))
        elif arch == "lstm":
            self.seq = LSTMSeq(d_in, cfg.get("width", 53))
        elif arch == "tcn":
            self.seq = TCNSeq(d_in, cfg.get("width", 40))
        elif arch == "mingru":
            self.seq = MinGRUSeq(d_in, cfg.get("width", 189))
        elif arch == "minlstm":
            self.seq = MinLSTMSeq(d_in, cfg.get("width", 126))
        elif arch == "lru":
            self.seq = LRUSeq(d_in, cfg.get("width", 80))
        elif arch == "xlstm":
            self.seq = MLSTMSeq(d_in, cfg.get("width", 94))
        elif arch == "rwkv7":
            self.seq = RWKV7Seq(d_in, cfg.get("width", 63))
        elif arch == "griffin":
            self.seq = GriffinSeq(d_in, cfg.get("width", 83))
        elif arch == "mamba2":
            self.seq = Mamba2Seq(d_in, cfg.get("width", 97))
        elif arch == "xlstm_fast":
            self.seq = MLSTMFastSeq(d_in, cfg.get("width", 94), heads=1)
        elif arch == "xlstm_mh":
            self.seq = MLSTMMultiHeadSeq(d_in, cfg.get("width", 94), heads=cfg.get("heads", 2))
        elif arch == "gdn":
            self.seq = GatedDeltaNetSeq(d_in, cfg.get("width", 73))
        elif arch == "gru2":
            self.seq = GRUStackSeq(d_in, cfg.get("width", 94), cfg.get("layers", 2))
        elif arch == "tf2":  # capacity track, pin 14a13a0e: 2-layer causal transformer, width 76, about 100.9k sequence parameters
            self.seq = TFSeq(d_in, K, cfg.get("width", 76), cfg.get("layers", 2), cfg.get("heads", 4), cfg.get("ff", 152))
        elif arch == "xlstm_sig":
            self.seq = MLSTMFastSeq(d_in, cfg.get("width", 94), heads=1, input_gate="sigmoid")
        elif arch == "xlstm_blk":
            self.seq = XLSTMBlockSeq(d_in, cfg.get("width", 46), cfg.get("hidden", 92))
        elif arch == "gdn2":
            self.seq = ChannelGatedDeltaSeq(d_in, cfg.get("width", 55))
        elif arch == "tf":
            self.seq = TFSeq(d_in, K, cfg.get("width", 32), cfg.get("layers", 2), cfg.get("heads", 4), cfg.get("ff", 64))
        else:
            raise SystemExit(f"unknown arm {arch}")
        d = self.seq.out_dim
        self.drop = nn.Dropout(cfg.get("dropout", 0.0))  # identity when 0 (default path unchanged), pin 29 test
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, Xw, Sw, mask):
        return self.head(self.drop(self.seq(self.enc(Xw, Sw, mask), mask))).squeeze(-1)


def add_train_site_distances(D):
    """Append the four train-only distance columns to the shared state. Each state row is matched to its distance row by
    (match_key, round, second) within its map. Matching by key, not by row count, keeps this working when --no_test_load has
    dropped the test-split rows from the state arrays; the distance file still holds them. Every state row must find exactly one
    distance row, or the run stops."""
    extra = np.zeros((D["state"].shape[0], N_DIST), dtype=np.float32)
    dist_cols = [f"p{i}" for i in range(N_DIST)]
    for mi, m in enumerate(tta.MAPS):
        rows = np.flatnonzero(D["map_id"] == mi)
        df = pl.read_parquet(TRAIN_SITES / f"{m}.parquet").select(["match_key", "round", "second"] + dist_cols)
        st = pl.DataFrame({
            "match_key": D["keys"][D["match_idx"][rows]],
            "round": D["round"][rows],
            "second": D["second"][rows],
            "row": rows,
        })
        for c in ("match_key", "round", "second"):
            st = st.with_columns(pl.col(c).cast(df.schema[c]))
        j = st.join(df, on=["match_key", "round", "second"], how="left")
        if j.height != rows.size:
            raise SystemExit(f"{m}: {rows.size} state rows matched to {j.height} rows; the distance file has duplicate keys")
        missing = j[dist_cols[0]].null_count()
        if missing:
            raise SystemExit(f"{m}: {missing} state rows have no distance row (keys do not line up)")
        extra[j["row"].to_numpy()] = j.select(dist_cols).to_numpy()
    # pos24 implementation detail (not in the pin): standardise with training-set statistics only, missing -> 0, clip to +-10
    tr = np.asarray(D["split"]) == 0
    mu = np.nanmean(np.where(np.isnan(extra), np.nan, extra)[tr], axis=0)
    sd = np.nanstd(np.where(np.isnan(extra), np.nan, extra)[tr], axis=0) + 1e-6
    extra = np.clip(np.nan_to_num((extra - mu) / sd, nan=0.0), -10, 10).astype(np.float32)
    print(f"added {N_DIST} train-only site distance columns (matched by key for every map)", flush=True)
    return np.concatenate([D["state"], extra], axis=1)


def gdt_weights(D, rows):
    """Objective G (pin 64d46c76). Per-second weight N * w_t, with w_t = 1 / (G * n_g). Groups are (match, round, grader
    situation); G is the number of groups and n_g the seconds in group g, over the training rows. A batch loss of the mean of
    weight * log loss counts each group once, as the grader's one-draw-per-group does. Situations come from the grader's own
    situations() (gdt_grader.py, a verbatim copy of evaluation/intern_gdt/score_gdt.py)."""
    import gdt_grader as sg
    N = rows.size
    base = pl.DataFrame({"match": D["keys"][D["match_idx"][rows]], "round": D["round"][rows].astype(np.int64),
                         "second": D["second"][rows].astype(np.int64), "row": np.arange(N, dtype=np.int64)})
    cache = (pl.scan_parquet(ROOT / "data" / "combined" / tta.CACHE)
             .select(["match_key", "round", "second", "alive_ct", "alive_t", "bomb_planted", "seconds_since_plant",
                      "st_equip_ct", "st_equip_t"])
             .rename({"match_key": "match"})
             .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
             .collect())
    df = base.join(cache, on=["match", "round", "second"], how="left")
    assert df.height == N, "cache join duplicated or dropped training seconds"
    for c in ["alive_ct", "alive_t", "bomb_planted", "st_equip_ct", "st_equip_t"]:
        assert df[c].null_count() == 0, f"no cache row for some training seconds ({c})"
    df = df.with_columns(
        pl.col("alive_ct").cast(pl.Int64), pl.col("alive_t").cast(pl.Int64),
        pl.col("bomb_planted").cast(pl.Int64).alias("planted"),
        pl.col("st_equip_ct").alias("equip_ct"), pl.col("st_equip_t").alias("equip_t"),
        pl.when(pl.col("bomb_planted") == 0).then(pl.lit(0.0)).otherwise(pl.col("seconds_since_plant").cast(pl.Float64)).alias("since_plant"),
        pl.lit(0).alias("y"),
    )
    sit = sg.situations(df.select(["match", "round", "second", "alive_ct", "alive_t", "planted", "y", "since_plant",
                                   "equip_ct", "equip_t", "row"]))
    sit = sit.with_columns(pl.concat_str([pl.col("match"), pl.col("round").cast(pl.Utf8)] + [pl.col(c).cast(pl.Utf8) for c in sg.CELL],
                                         separator="|").alias("gkey"))
    counts = sit.group_by("gkey").agg(pl.len().alias("n_g"))
    G = counts.height
    sit = sit.join(counts, on="gkey", how="left").sort("row")
    w = N / (G * sit["n_g"].to_numpy().astype(np.float64))
    assert abs(float(w.mean()) - 1.0) < 1e-6, "weights do not average to 1"
    out = np.zeros(len(D["split"]), dtype=np.float64)
    out[rows] = w
    return out, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gru,gru_lb1,lstm,tcn,tf")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_tf", type=int, default=500)
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--dist", action="store_true", help="add the train-only straight-line distance features to the state")
    ap.add_argument("--no_test_load", action="store_true", help="drop test-split rows inside the shared loader so they never enter the training arrays")
    ap.add_argument("--width", type=int, default=0, help="hidden width of the sequence model; 0 keeps the phase-1 default")
    ap.add_argument("--layers", type=int, default=0, help="transformer blocks (tf only); 0 keeps the default of 2")
    ap.add_argument("--heads", type=int, default=0, help="attention heads (tf only); 0 keeps the default of 4")
    ap.add_argument("--ff", type=int, default=0, help="transformer feed-forward width (tf only); 0 keeps the default of 64")
    ap.add_argument("--objective", choices=["logloss", "gdt"], default="logloss",
                    help="training loss: logloss (default) or gdt (GDT-equivalent per-second weights, pin 64d46c76)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout on the sequence output before the head (0 = off)")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay (0 = off)")
    ap.add_argument("--save_weights", action="store_true",
                    help="write each run's final-step model weights to <tag>_final_model.pt (default off)")
    a = ap.parse_args()
    cfg = {k: v for k, v in (("width", a.width), ("layers", a.layers), ("heads", a.heads), ("ff", a.ff), ("dropout", a.dropout)) if v}
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; torch {torch.__version__}", flush=True)

    # One data build for all arms. Lookback 20 is built once; lookback-1 slices the first column.
    D = tta.load_all(20, keep_test=not a.no_test_load)
    if a.dist:
        if out.resolve() == DEFAULT_OUT.resolve():
            raise SystemExit("--dist runs need their own --out_dir (the default folder holds the control runs)")
        D["state"] = add_train_site_distances(D)
    feat_t = torch.as_tensor(D["feat"], dtype=torch.float16, device=device)
    state_t = torch.as_tensor(D["state"], dtype=torch.float32, device=device)
    label_np = D["label"]
    label_t = torch.as_tensor(label_np, dtype=torch.float32, device=device)
    prev_t = torch.as_tensor(D["prev"], dtype=torch.int32, device=device)  # window index table, kept on the GPU
    train_rows = np.flatnonzero((D["split"] == tta.TRAIN) & D["usable"])
    val_rows = np.flatnonzero((D["split"] == tta.VAL) & D["usable"])
    test_note = "test rows dropped by the loader" if a.no_test_load else "test rows loaded by the shared loader, never used"
    print(f"rows: train {train_rows.size:,}, validation {val_rows.size:,} ({test_note})", flush=True)
    if a.no_test_load:
        assert not (D["split"] == tta.TEST).any(), "test-split rows reached the arrays despite --no_test_load"
    state_dim = state_t.shape[1]
    gdt_on = a.objective == "gdt"
    n_groups = None
    if gdt_on:
        ww_np, n_groups = gdt_weights(D, train_rows)
        ww_t = torch.as_tensor(ww_np, dtype=torch.float32, device=device)
        print(f"objective gdt: groups {n_groups:,}, mean weight over train rows {float(ww_np[train_rows].mean()):.6f}", flush=True)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    def gather(rows_t, L):
        # rows_t: row indices already on the GPU; everything below runs on the GPU (no per-step host work)
        idx = prev_t[rows_t, :L].long()
        mask = (idx >= 0).float()
        safe = idx.clamp(min=0)
        Xw = feat_t[safe].float() * mask[..., None, None]
        Sw = state_t[safe]
        return Xw.flip(1), Sw.flip(1), mask.flip(1)

    for arm in arms:
        L = ARMS[arm]
        for seed in seeds:
            tag = f"{arm}_gdt_seed{seed}" if gdt_on else f"{arm}_seed{seed}"
            if (out / f"{tag}_metrics.json").exists():
                print(f"skip {tag}: already done", flush=True)
                continue
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = ArmModel(arm, state_dim, L, cfg).to(device)
            n_seq = sum(p.numel() for p in model.seq.parameters())
            n_total = sum(p.numel() for p in model.parameters())
            opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
            warm = a.warmup_tf if arm == "tf" else 0
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, w=warm: min(1.0, (s + 1) / w) if w else 1.0)

            @torch.no_grad()
            def predict(rows):
                model.eval()
                res = np.empty(rows.size, dtype=np.float32)
                for s in range(0, rows.size, 8192):
                    r = rows[s:s + 8192]
                    Xw, Sw, m = gather(torch.as_tensor(r, dtype=torch.long, device=device), L)
                    res[s:s + 8192] = torch.sigmoid(model(Xw, Sw, m)).cpu().numpy()
                model.train()
                return res

            print(f"== {tag}: lookback {L}, size {cfg or 'default'}, seq params {n_seq:,}, total params {n_total:,}", flush=True)
            best = (np.inf, None, 0)
            final_p = None
            history = []
            order = torch.as_tensor(rng.permutation(train_rows), dtype=torch.long, device=device)
            pos = 0
            t0 = time.time()
            for step in range(1, a.steps + 1):
                if pos >= order.numel():
                    order = torch.as_tensor(rng.permutation(train_rows), dtype=torch.long, device=device)
                    pos = 0
                r = order[pos:pos + a.batch]
                pos += a.batch
                Xw, Sw, m = gather(r, L)
                per = loss_fn(model(Xw, Sw, m), label_t[r])
                loss = (ww_t[r] * per).mean() if gdt_on else per
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                if step in (200, 1000):
                    print(f"  step {step}: {time.time() - t0:.1f} s elapsed", flush=True)
                if step % a.eval_every == 0 or step == a.steps:
                    p_now = predict(val_rows)
                    v = ttr.log_loss(label_np[val_rows], p_now)
                    if step == a.steps:
                        final_p = p_now  # fixed-step predictions: no checkpoint selection
                    history.append({"step": step, "val_log_loss": v, "seconds": round(time.time() - t0, 1)})
                    print(json.dumps(history[-1]), flush=True)
                    if v < best[0]:
                        best = (v, copy.deepcopy(model.state_dict()), step)
            pl.DataFrame({
                "match_key": D["keys"][D["match_idx"][val_rows]],
                "round": D["round"][val_rows].astype(np.int64),
                "second": D["second"][val_rows].astype(np.int64),
                "map_name": D["map_id"][val_rows].astype(np.int64),
                "label": label_np[val_rows].astype(np.int64),
                "p": final_p.astype(np.float64),
            }).write_parquet(out / f"{tag}_final_val_preds.parquet")
            if a.save_weights:  # final-step weights, before the best checkpoint is loaded for the best-checkpoint predictions
                torch.save(model.state_dict(), out / f"{tag}_final_model.pt")
            model.load_state_dict(best[1])
            p_val = predict(val_rows)
            pl.DataFrame({
                "match_key": D["keys"][D["match_idx"][val_rows]],
                "round": D["round"][val_rows].astype(np.int64),
                "second": D["second"][val_rows].astype(np.int64),
                "map_name": D["map_id"][val_rows].astype(np.int64),
                "label": label_np[val_rows].astype(np.int64),
                "p": p_val.astype(np.float64),
            }).write_parquet(out / f"{tag}_val_preds.parquet")
            seconds = time.time() - t0
            metrics = {
                "tag": tag, "arm": arm, "seed": seed, "lookback_s": L, "steps": a.steps, "eval_every": a.eval_every,
                "batch": a.batch, "lr": a.lr, "best_step": best[2], "best_val_log_loss": best[0],
                "undertrained_flag": best[2] == a.steps, "history": history, "seq_params": n_seq,
                "total_params": n_total, "seconds_total": round(seconds, 1),
                "seconds_per_step": round(seconds / a.steps, 4), "train_rows": int(train_rows.size),
                "final_val_log_loss": history[-1]["val_log_loss"],
                "val_rows": int(val_rows.size), "device": str(device), "test_rows_used": 0, "test_rows_in_memory": not a.no_test_load,
                "test_rows_in_arrays": int((D["split"] == tta.TEST).sum()),
                "dist": a.dist, "state_dim": state_dim, "objective": a.objective, "gdt_groups": n_groups,
                "dropout": a.dropout, "weight_decay": a.weight_decay,
            }
            (out / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
            print(json.dumps({k: metrics[k] for k in ("tag", "best_step", "best_val_log_loss", "undertrained_flag")}),
                  flush=True)
            del model, opt
            torch.cuda.empty_cache() if device.type == "cuda" else None


if __name__ == "__main__":
    main()
