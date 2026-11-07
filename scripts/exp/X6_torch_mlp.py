"""X6: torch MLP 10 -> 128 -> 128 -> 64 -> 1 on CUDA, per-second win-probability cache.

Run with the torch env interpreter:
    <cs2gpu_torch env>\python.exe -I -X utf8 G:\pureskill\csds-cs2\scripts\exp\X6_torch_mlp.py
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
import copy
from pathlib import Path

import numpy as np

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import pt_eval as E  # noqa: E402  (sets MKL_THREADING_LAYER before numpy heavy use)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ID = "X6"
PRED_DIR = Path(r"G:\pureskill\csds-cs2\data\viz\pt_preds")
EPOCHS = 30
BATCH = 8192
SEED = 0


def build_model(dev):
    return nn.Sequential(
        nn.Linear(10, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(dev)


def predict(model, X_t):
    model.eval()
    with torch.no_grad():
        z = model(X_t).squeeze(1).double()
        return torch.sigmoid(z).cpu().numpy().astype(np.float64)


def run():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in this torch env")
    dev = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(0)
    print(f"device: {dev_name} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    D = E.load()
    Xtr = D["train"]["X"].astype(np.float32)
    ytr = D["train"]["y"].astype(np.float32)
    Xva = D["val"]["X"].astype(np.float32)
    yva = D["val"]["y"].astype(np.float64)
    Xte = D["test"]["X"].astype(np.float32)
    print(f"rows train={len(ytr)} val={len(yva)} test={len(D['test']['y'])} feats={Xtr.shape[1]}", flush=True)

    mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = Xtr.std(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)

    Xtr_t = torch.from_numpy((Xtr - mu) / sd).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    Xva_t = torch.from_numpy((Xva - mu) / sd).to(dev)
    Xte_t = torch.from_numpy((Xte - mu) / sd).to(dev)

    model = build_model(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    N = Xtr_t.shape[0]

    best_ll = float("inf")
    best_ep = -1
    best_state = None

    torch.cuda.synchronize()
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(N, device=dev)
        run_loss = 0.0
        n_seen = 0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            out = model(Xtr_t[idx]).squeeze(1)
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * idx.numel()
            n_seen += idx.numel()
        p_va = predict(model, Xva_t)
        val_ll = float(E.ll_vec(p_va, yva).mean())
        if val_ll < best_ll:
            best_ll = val_ll
            best_ep = ep
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {ep:2d} train_bce={run_loss / n_seen:.5f} val_logloss={val_ll:.5f} "
              f"best_ep={best_ep} best_val={best_ll:.5f} elapsed={time.time() - t0:.1f}s", flush=True)
    torch.cuda.synchronize()
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)
    p_val = predict(model, Xva_t)
    p_test = predict(model, Xte_t)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PRED_DIR / f"{ID}_val.npy", p_val)
    np.save(PRED_DIR / f"{ID}_test.npy", p_test)
    print(f"saved preds: val={p_val.shape} test={p_test.shape} best_epoch={best_ep} "
          f"best_val_logloss={best_ll:.6f} train_seconds={train_seconds:.1f}", flush=True)

    res = E.report(ID, D["test"], p_test, extra={
        "model": "torch_mlp_10-128-128-64-1_relu_dropout0.1",
        "device": dev_name,
        "best_epoch": best_ep,
        "epochs_max": EPOCHS,
        "batch": BATCH,
        "lr": 2e-3,
        "train_seconds": round(train_seconds, 2),
        "val_logloss_best": best_ll,
    })
    print("RESULT", res, flush=True)
    return dev_name, train_seconds, best_ep, best_ll


def main():
    import time as _t
    last_err = None
    for attempt in range(4):
        try:
            return run()
        except torch.cuda.OutOfMemoryError as e:  # retry per the harness instructions
            last_err = e
            print(f"CUDA OOM on attempt {attempt + 1}, waiting 30s", flush=True)
            _t.sleep(30)
    raise last_err


if __name__ == "__main__":
    main()
