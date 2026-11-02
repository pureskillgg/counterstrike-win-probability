"""GRU lookback test under the sweep's protocol: cosine schedule, EMA, final-step scoring, lockstep configs on identical mini-batches.

Separate from train_lr_sweep.py so the pinned sweep keeps the code it ran. Differences:
  windows   built per batch by window_index.py (no N x lookback table), so 60-second windows fit on the GPU
  configs   gru:<lr>:<lookback>, lookback 20 or 60 (the same network, a different window length)
  reference the 20-second config is the same GRU as the pinned sweep GRU; it must reproduce that run (reference gate)

Validation only. The shared loader builds test rows in memory, but nothing here uses them for training, batch order,
validation or selection.

Outputs per config and seed (in --out_dir):
  gru_L<L>_lr<lr>_seed<s>_final_val_preds.parquet       raw weights at the final step
  gru_L<L>_lr<lr>_seed<s>_ema_final_val_preds.parquet   EMA weights at the final step
  gru_L<L>_lr<lr>_seed<s>_metrics.json                  history (raw and EMA), final values, parameter counts; written last

  ...tabicl python.exe -I -X utf8 -u scripts/model/train_window_sweep.py --cfgs gru:1e-3:20,gru:1e-3:60 --seeds 0 --out_dir <dir>
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
import train_temporal_allmaps as tta  # noqa: E402
import train_phase1_arms as P1  # noqa: E402
import window_index as W  # noqa: E402
import train_temporal_round_model as ttr  # noqa: E402

ALLOWED_L = (20, 40, 60)


def lr_mult(s, warm, total):
    """Learning-rate multiplier after s optimiser steps: linear warm-up, then cosine decay to zero at total."""
    if s < warm:
        return (s + 1) / warm
    prog = min(1.0, (s - warm) / max(1, total - warm))
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def tag_of(lr, L, seed):
    return f"gru_L{L}_lr{lr:g}_seed{seed}"


def parse_cfgs(text):
    out = []
    for item in text.split(","):
        arm, lr, L = item.strip().split(":")
        if arm != "gru":
            raise SystemExit("only the GRU is allowed in this trainer")
        L = int(L)
        if L not in ALLOWED_L:
            raise SystemExit(f"lookback {L} not allowed (allowed: {ALLOWED_L})")
        cfg = (float(lr), L)
        if cfg in out:
            raise SystemExit(f"duplicate config {item} (the same output file would be written twice)")
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
    ap.add_argument("--cfgs", required=True, help="comma-separated gru:lr:lookback items, e.g. gru:1e-3:20,gru:1e-3:60")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dist", action="store_true", help="append the four train-only site-distance columns to the shared state (14 values, not 10)")
    ap.add_argument("--drop_maps", default="", help="comma-separated map names whose training rows are left out; validation and scoring are unchanged")
    a = ap.parse_args()
    cfgs = parse_cfgs(a.cfgs)
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.steps <= 0 or a.eval_every <= 0 or a.batch <= 0:
        raise SystemExit("steps, eval_every and batch must be positive")
    settings = {"steps": a.steps, "warmup": a.warmup, "ema_decay": a.ema, "eval_every": a.eval_every, "batch": a.batch, "drop_maps": a.drop_maps}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; torch {torch.__version__}; configs {cfgs}; seeds {seeds}", flush=True)

    D = tta.load_all(ALLOWED_L[-1], with_prev=False)
    if a.dist:
        D["state"] = P1.add_train_site_distances(D)
    feat_t = torch.as_tensor(D["feat"], dtype=torch.float16, device=device)
    state_t = torch.as_tensor(D["state"], dtype=torch.float32, device=device)
    label_np = D["label"]
    label_t = torch.as_tensor(label_np, dtype=torch.float32, device=device)
    index = W.build(D["match_idx"], D["round"], D["second"], device)
    drop_ids = [tta.MAPS.index(m) for m in a.drop_maps.split(",") if m]
    keep_train = (D["split"] == tta.TRAIN) & D["usable"] & ~np.isin(D["map_id"], drop_ids)
    train_rows = np.flatnonzero(keep_train)
    print(f"training rows kept: {train_rows.size:,} (dropped maps: {a.drop_maps or 'none'})", flush=True)
    val_rows = np.flatnonzero((D["split"] == tta.VAL) & D["usable"])
    state_dim = state_t.shape[1]
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"rows: train {train_rows.size:,}, validation {val_rows.size:,} (test rows loaded, never used)", flush=True)

    def gather(rows_t, L):
        # oldest second first, the current second last, zero-filled and masked where the second is absent
        idx = W.window(index, rows_t, L)
        mask = (idx >= 0).float()
        safe = idx.clamp(min=0)
        Xw = feat_t[safe].float() * mask[..., None, None]
        Sw = state_t[safe]
        return Xw.flip(1), Sw.flip(1), mask.flip(1)

    @torch.no_grad()
    def predict(model, L, rows):
        model.eval()
        res = np.empty(rows.size, dtype=np.float32)
        for s in range(0, rows.size, 8192):
            r = rows[s:s + 8192]
            Xw, Sw, m = gather(torch.as_tensor(r, dtype=torch.long, device=device), L)
            res[s:s + 8192] = torch.sigmoid(model(Xw, Sw, m)).cpu().numpy()
        model.train()
        return res

    for seed in seeds:
        todo = []
        for lr, L in cfgs:
            mpath = out / f"{tag_of(lr, L, seed)}_metrics.json"
            if not mpath.exists():
                todo.append((lr, L))
                continue
            old = json.loads(mpath.read_text(encoding="utf-8"))
            if any(old.get(key) != val for key, val in settings.items()):
                raise SystemExit(f"{mpath.name} was written with other settings; use a new --out_dir")
            print(f"skip {mpath.name}: already done with the same settings", flush=True)
        if not todo:
            continue
        rng = np.random.default_rng(seed)
        nets, emas, opts, scheds, pairs = [], [], [], [], []
        for lr, L in todo:
            torch.manual_seed(seed)  # the same initial weights for every config of this seed (same structure)
            net = P1.ArmModel("gru", state_dim, L).to(device)
            ema = copy.deepcopy(net)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_mult(s, a.warmup, a.steps))
            nets.append(net)
            emas.append(ema)
            opts.append(opt)
            scheds.append(sch)
            pairs.append((list(ema.parameters()), list(net.parameters())))
        print(f"== seed {seed}: " + ", ".join(tag_of(lr, L, seed) for lr, L in todo), flush=True)

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
            y = label_t[r]
            windows = {}
            for k in range(len(todo)):
                L = todo[k][1]
                if L not in windows:
                    windows[L] = gather(r, L)
                Xw, Sw, m = windows[L]
                loss = loss_fn(nets[k](Xw, Sw, m), y)
                opts[k].zero_grad()
                loss.backward()
                opts[k].step()
                scheds[k].step()
                ema_update(pairs[k], a.ema)
            if step in (200, 1000):
                print(f"  step {step}: {time.time() - t0:.1f} s elapsed", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                for k, (lr, L) in enumerate(todo):
                    p_raw = predict(nets[k], L, val_rows)
                    p_ema = predict(emas[k], L, val_rows)
                    v_raw = ttr.log_loss(label_np[val_rows], p_raw)
                    v_ema = ttr.log_loss(label_np[val_rows], p_ema)
                    hist[k].append({"step": step, "raw_val_log_loss": v_raw, "ema_val_log_loss": v_ema,
                                    "seconds": round(time.time() - t0, 1)})
                    print(json.dumps({"tag": tag_of(lr, L, seed), **hist[k][-1]}), flush=True)
                    if step == a.steps:
                        finals[k] = (p_raw, p_ema)

        seconds = time.time() - t0
        for k, (lr, L) in enumerate(todo):
            tag = tag_of(lr, L, seed)
            p_raw, p_ema = finals[k]
            write_preds(out / f"{tag}_final_val_preds.parquet", D, val_rows, label_np, p_raw)
            write_preds(out / f"{tag}_ema_final_val_preds.parquet", D, val_rows, label_np, p_ema)
            metrics = {
                "tag": tag, "arm": "gru", "lr": lr, "lookback_s": L, "seed": seed,
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
        del nets, emas, opts, scheds, pairs, windows
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
