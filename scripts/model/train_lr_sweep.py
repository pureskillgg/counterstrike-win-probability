"""Learning-rate and schedule sweep for the sequence arms (design pre-registered in the Meridian pin for the sweep).

Separate from train_phase1_arms.py, so the running distance-input and control jobs keep the code they loaded. Differences:
  schedule  linear warm-up for --warmup steps, then cosine decay to zero at --steps (the same for every arm and learning rate)
  EMA       an exponential moving average of the weights (decay --ema), kept alongside the raw weights
  scoring   the model at the final step, raw and EMA. No checkpoint is selected on validation.
  lockstep  every (arm, learning rate) config for one seed is trained in one process, on the same mini-batches (same seed and
            batch order), so the learning-rate comparison is paired and the shared data is loaded once. Each config starts from the
            same initial weights for the same arm and seed.

Validation only. The shared loader builds test rows in memory, but nothing here uses them for training, batch order, validation or
selection.

Outputs per config and seed (in --out_dir):
  <arm>_lr<lr>_seed<s>_final_val_preds.parquet      raw weights at the final step
  <arm>_lr<lr>_seed<s>_ema_final_val_preds.parquet  EMA weights at the final step
  <arm>_lr<lr>_seed<s>_metrics.json                 validation log loss history (raw and EMA), final values, parameter counts

  ...tabicl python.exe -I -X utf8 -u scripts/model/train_lr_sweep.py --cfgs gru:1e-3,tcn:3e-4 --seeds 0 --out_dir data/diagnostics/lr_sweep_screen
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import train_temporal_round_model as ttr  # noqa: E402
import train_temporal_allmaps as tta  # noqa: E402
import train_phase1_arms as P1  # noqa: E402

LOOKBACK = 20
ALLOWED = ("gru", "gru_lb1", "tcn", "tf", "lstm")


def lr_mult(s, warm, total):
    """Learning-rate multiplier after s optimiser steps: linear warm-up, then cosine decay to zero at total."""
    if s < warm:
        return (s + 1) / warm
    prog = min(1.0, (s - warm) / max(1, total - warm))
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def lr_tag(lr):
    return f"{lr:g}"


def tag_of(arm, lr, seed):
    return f"{arm}_lr{lr_tag(lr)}_seed{seed}"


def parse_cfgs(text):
    out = []
    for item in text.split(","):
        arm, lr = item.strip().split(":")
        if arm not in ALLOWED:
            raise SystemExit(f"unknown arm {arm}")
        cfg = (arm, float(lr))
        if cfg in out:
            raise SystemExit(f"duplicate config {arm}:{lr} (the same output file would be written twice)")
        out.append(cfg)
    return out


def ema_update(pair, d):
    ema_params, live_params = pair
    with torch.no_grad():
        torch._foreach_mul_(ema_params, d)
        torch._foreach_add_(ema_params, live_params, alpha=1.0 - d)


def write_preds(path, D, val_rows, label_np, p):
    pl.DataFrame({
        "match_key": D["keys"][D["match_idx"][val_rows]],
        "round": D["round"][val_rows].astype(np.int64),
        "second": D["second"][val_rows].astype(np.int64),
        "map_name": D["map_id"][val_rows].astype(np.int64),
        "label": label_np[val_rows].astype(np.int64),
        "p": p.astype(np.float64),
    }).write_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True, help="comma-separated arm:lr pairs, e.g. gru:1e-3,tcn:3e-4")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    cfgs = parse_cfgs(a.cfgs)
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.steps <= 0 or a.eval_every <= 0 or a.batch <= 0:
        raise SystemExit("steps, eval_every and batch must be positive")
    settings = {"steps": a.steps, "warmup": a.warmup, "ema_decay": a.ema, "eval_every": a.eval_every, "batch": a.batch}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; torch {torch.__version__}; configs {cfgs}; seeds {seeds}", flush=True)

    D = tta.load_all(LOOKBACK)
    feat_t = torch.as_tensor(D["feat"], dtype=torch.float16, device=device)
    state_t = torch.as_tensor(D["state"], dtype=torch.float32, device=device)
    label_np = D["label"]
    label_t = torch.as_tensor(label_np, dtype=torch.float32, device=device)
    prev_t = torch.as_tensor(D["prev"], dtype=torch.int32, device=device)
    train_rows = np.flatnonzero((D["split"] == tta.TRAIN) & D["usable"])
    val_rows = np.flatnonzero((D["split"] == tta.VAL) & D["usable"])
    state_dim = state_t.shape[1]
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"rows: train {train_rows.size:,}, validation {val_rows.size:,} (test rows loaded, never used)", flush=True)

    def gather(rows_t, L):
        # same window construction as train_phase1_arms.py: oldest second first, the current second last
        idx = prev_t[rows_t, :L].long()
        mask = (idx >= 0).float()
        safe = idx.clamp(min=0)
        Xw = feat_t[safe].float() * mask[..., None, None]
        Sw = state_t[safe]
        return Xw.flip(1), Sw.flip(1), mask.flip(1)

    def window(arm, Xw, Sw, m):
        # gru_lb1 reads only the current second, the last column after the flip; the other arms read the full window
        L = P1.ARMS[arm]
        return (Xw[:, -L:], Sw[:, -L:], m[:, -L:]) if L < LOOKBACK else (Xw, Sw, m)

    @torch.no_grad()
    def predict(model, arm, rows):
        model.eval()
        res = np.empty(rows.size, dtype=np.float32)
        for s in range(0, rows.size, 8192):
            r = rows[s:s + 8192]
            Xw, Sw, m = window(arm, *gather(torch.as_tensor(r, dtype=torch.long, device=device), LOOKBACK))
            res[s:s + 8192] = torch.sigmoid(model(Xw, Sw, m)).cpu().numpy()
        model.train()
        return res

    for seed in seeds:
        todo = []
        for arm, lr in cfgs:
            mpath = out / f"{tag_of(arm, lr, seed)}_metrics.json"
            if not mpath.exists():
                todo.append((arm, lr))
                continue
            old = json.loads(mpath.read_text(encoding="utf-8"))
            if any(old.get(key) != val for key, val in settings.items()):
                raise SystemExit(f"{mpath.name} was written with other settings ("
                                 f"{ {key: old.get(key) for key in settings} }); use a new --out_dir")
            print(f"skip {mpath.name}: already done with the same settings", flush=True)
        if not todo:
            continue
        rng = np.random.default_rng(seed)
        nets, emas, opts, scheds, pairs = [], [], [], [], []
        for arm, lr in todo:
            torch.manual_seed(seed)  # the same initial weights for every learning rate of this arm and seed
            net = P1.ArmModel(arm, state_dim, P1.ARMS[arm]).to(device)
            ema = copy.deepcopy(net)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_mult(s, a.warmup, a.steps))
            nets.append(net)
            emas.append(ema)
            opts.append(opt)
            scheds.append(sch)
            pairs.append((list(ema.parameters()), list(net.parameters())))
        print(f"== seed {seed}: " + ", ".join(tag_of(arm, lr, seed) for arm, lr in todo), flush=True)

        order = torch.as_tensor(rng.permutation(train_rows), dtype=torch.long, device=device)
        pos = 0
        hist = [[] for _ in todo]
        finals = [None] * len(todo)
        t0 = time.time()
        for step in range(1, a.steps + 1):
            if pos >= order.numel():
                order = torch.as_tensor(rng.permutation(train_rows), dtype=torch.long, device=device)
                pos = 0
            r = order[pos:pos + a.batch]
            pos += a.batch
            Xw, Sw, m = gather(r, LOOKBACK)
            y = label_t[r]
            for k in range(len(todo)):
                Xk, Sk, mk = window(todo[k][0], Xw, Sw, m)
                loss = loss_fn(nets[k](Xk, Sk, mk), y)
                opts[k].zero_grad()
                loss.backward()
                opts[k].step()
                scheds[k].step()
                ema_update(pairs[k], a.ema)
            if step in (200, 1000):
                print(f"  step {step}: {time.time() - t0:.1f} s elapsed", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                for k, (arm, lr) in enumerate(todo):
                    p_raw = predict(nets[k], arm, val_rows)
                    p_ema = predict(emas[k], arm, val_rows)
                    v_raw = ttr.log_loss(label_np[val_rows], p_raw)
                    v_ema = ttr.log_loss(label_np[val_rows], p_ema)
                    hist[k].append({"step": step, "raw_val_log_loss": v_raw, "ema_val_log_loss": v_ema,
                                    "seconds": round(time.time() - t0, 1)})
                    print(json.dumps({"tag": tag_of(arm, lr, seed), **hist[k][-1]}), flush=True)
                    if step == a.steps:
                        finals[k] = (p_raw, p_ema)

        seconds = time.time() - t0
        for k, (arm, lr) in enumerate(todo):
            tag = tag_of(arm, lr, seed)
            p_raw, p_ema = finals[k]
            write_preds(out / f"{tag}_final_val_preds.parquet", D, val_rows, label_np, p_raw)
            write_preds(out / f"{tag}_ema_final_val_preds.parquet", D, val_rows, label_np, p_ema)
            metrics = {
                "tag": tag, "arm": arm, "lr": lr, "seed": seed, "lookback_s": P1.ARMS[arm],
                "schedule": f"linear warm-up {a.warmup} steps, cosine decay to zero at {a.steps} steps",
                "ema_decay": a.ema, "warmup": a.warmup, "steps": a.steps, "eval_every": a.eval_every, "batch": a.batch,
                "final_raw_val_log_loss": hist[k][-1]["raw_val_log_loss"],
                "final_ema_val_log_loss": hist[k][-1]["ema_val_log_loss"],
                "history": hist[k],
                "seq_params": sum(p.numel() for p in nets[k].seq.parameters()),
                "total_params": sum(p.numel() for p in nets[k].parameters()),
                "seconds_total": round(seconds, 1), "train_rows": int(train_rows.size), "val_rows": int(val_rows.size),
                "selection": "none: scored at the final step", "device": str(device),
                "test_rows_used": 0, "test_rows_in_memory": True,
            }
            # metrics are written last: their presence marks the config as done
            (out / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
            print(json.dumps({"done": tag, "final_raw": metrics["final_raw_val_log_loss"],
                              "final_ema": metrics["final_ema_val_log_loss"]}), flush=True)
        del nets, emas, opts, scheds, pairs
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
