"""CPU checks for the round-2 arms: closed-form mLSTM equals the loop form (1 head and multi-head), GDN and stacked GRU run with
finite outputs and gradients, and parameter counts against the 25,344 GRU control and the ~100k capacity target."""
import importlib.util
import sys

import torch
import torch.nn.functional as F

SRC = "G:/pureskill/csds-cs2/scripts/model/train_phase1_pos24.py"
sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/model")
spec = importlib.util.spec_from_file_location("tp24", SRC)
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

torch.manual_seed(1)
D_IN, B, T = 66, 32, 20
x = torch.randn(B, T, D_IN)
mask = torch.ones(B, T, dtype=torch.bool)
n_par = lambda m: sum(p.numel() for p in m.parameters())


def mlstm_loop_heads(seq, x):
    """Reference: the paper's stabilised recurrence, looped over time, one memory per head."""
    B, T, _ = x.shape
    H, dh, d = seq.H, seq.dh, seq.d
    qkv = seq.lin_qkv(x)
    q = qkv[..., :d].reshape(B, T, H, dh)
    k = (qkv[..., d:2 * d] / dh ** 0.5).reshape(B, T, H, dh)
    v = qkv[..., 2 * d:].reshape(B, T, H, dh)
    g = seq.lin_gates(x)
    i_pre = g[..., :H]
    log_f = F.logsigmoid(g[..., H:2 * H])
    o = torch.sigmoid(g[..., 2 * H:]).reshape(B, T, H, dh)
    C = torch.zeros(B, H, dh, dh)
    n = torch.zeros(B, H, dh)
    m = torch.full((B, H), -1e30)
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        m_new = torch.maximum(log_f[:, t] + m, i_pre[:, t])
        fp = torch.exp(log_f[:, t] + m - m_new)
        ip = torch.exp(i_pre[:, t] - m_new)
        C = fp[:, :, None, None] * C + ip[:, :, None, None] * (vt[:, :, :, None] * kt[:, :, None, :])
        n = fp[:, :, None] * n + ip[:, :, None] * kt
        m = m_new
    qT = q[:, -1]
    num = torch.einsum("bhij,bhj->bhi", C, qT)
    den = torch.maximum((n * qT).sum(-1).abs(), torch.exp(-m))
    return (o[:, -1] * num / den[..., None]).reshape(B, d)


# 1. closed form equals the original loop class (one head)
orig = tp.MLSTMSeq(D_IN, 94)
fast1 = tp.MLSTMFastSeq(D_IN, 94, heads=1)
fast1.load_state_dict(orig.state_dict())
y_orig, y_fast = orig(x, mask), fast1(x, mask)
print("1-head closed form vs loop class: max abs diff", float((y_orig - y_fast).abs().max()),
      "max |y|", float(y_orig.abs().max()))

# 2. multi-head closed form equals the per-head loop reference
mh = tp.MLSTMMultiHeadSeq(D_IN, 94, heads=2)
y_mh, y_ref = mh(x, mask), mlstm_loop_heads(mh, x)
print("2-head closed form vs per-head loop: max abs diff", float((y_mh - y_ref).abs().max()),
      "max |y|", float(y_mh.abs().max()))

# 3. gradients and finiteness for the new arms
gru_params = n_par(tp.GRUSeq(D_IN, 64))
print("GRU control seq params:", gru_params)
for name, m in [("xlstm_mh", mh), ("gdn", tp.GatedDeltaNetSeq(D_IN, 73)), ("gru2", tp.GRUStackSeq(D_IN, 94, 2))]:
    y = m(x, mask)
    y.pow(2).mean().backward()
    ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    print(f"{name}: out {tuple(y.shape)} finite {bool(torch.isfinite(y).all())} grads_finite {ok} "
          f"params {n_par(m)} vs GRU control {100 * (n_par(m) - gru_params) / gru_params:+.2f}%")

# 4. capacity track: transformer widths that land near 100k sequence parameters
for w in (48, 56, 64):
    t2 = tp.TFSeq(D_IN, 20, w, 2, 4, 2 * w)
    print(f"tf 2-layer width {w}: seq params {n_par(t2)}")
print("gru2 width 94 (2 layers):", n_par(tp.GRUStackSeq(D_IN, 94, 2)))
