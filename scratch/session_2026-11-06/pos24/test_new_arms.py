"""CPU check for the four new sequence arms: shapes, finite outputs, gradients, and sequence parameter counts vs the GRU control."""
import importlib.util
import sys

import torch

SRC = "G:/pureskill/csds-cs2/scripts/model/train_phase1_pos24.py"
sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/model")
spec = importlib.util.spec_from_file_location("tp24", SRC)
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

torch.manual_seed(0)
D_IN, B, T = 66, 16, 20
x = torch.randn(B, T, D_IN)
mask = torch.ones(B, T, dtype=torch.bool)
n = lambda m: sum(p.numel() for p in m.parameters())
gru_params = n(tp.GRUSeq(D_IN, 64))
print("GRU control seq params:", gru_params)
for name, cls in [("xlstm", tp.MLSTMSeq), ("rwkv7", tp.RWKV7Seq), ("griffin", tp.GriffinSeq), ("mamba2", tp.Mamba2Seq)]:
    m = cls(D_IN)
    y = m(x, mask)
    loss = y.pow(2).mean()
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    print(f"{name}: out {tuple(y.shape)} finite {bool(torch.isfinite(y).all())} grads_finite {grads_ok} "
          f"params {n(m)} vs GRU {100 * (n(m) - gru_params) / gru_params:+.2f}%")
# the arm dispatch builds through ArmModel with the default widths
for arch in ["xlstm", "rwkv7", "griffin", "mamba2"]:
    am = tp.ArmModel(arch, 0, 20)
    print(arch, "ArmModel seq params", n(am.seq))
