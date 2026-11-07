import importlib.util, sys, torch
sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/model")
spec = importlib.util.spec_from_file_location("tp24", "G:/pureskill/csds-cs2/scripts/model/train_phase1_pos24.py")
tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
torch.manual_seed(2)
x = torch.randn(32, 20, 66); mask = torch.ones(32, 20, dtype=torch.bool)
n = lambda m: sum(p.numel() for p in m.parameters())
ref = n(tp.GRUSeq(66, 64))
orig = tp.MLSTMSeq(66, 94); fast = tp.MLSTMFastSeq(66, 94, heads=1); fast.load_state_dict(orig.state_dict())
print("exp-gate closed form vs loop, max abs diff:", float((orig(x, mask) - fast(x, mask)).abs().max().detach()))
for name, m in [("xlstm_sig", tp.MLSTMFastSeq(66, 94, heads=1, input_gate="sigmoid")),
                ("xlstm_blk", tp.XLSTMBlockSeq(66, 46, 92)),
                ("gdn2", tp.ChannelGatedDeltaSeq(66, 55))]:
    y = m(x, mask); y.pow(2).mean().backward()
    ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    print(f"{name}: out {tuple(y.shape)} finite {bool(torch.isfinite(y).all())} grads_finite {ok} params {n(m)} vs GRU control {100*(n(m)-ref)/ref:+.2f}%")
