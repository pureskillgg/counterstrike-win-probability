"""Phase 1 sequence-architecture arms on all maps (pre-registered, decision 6fb5578a). Validation only for every decision. The shared loader (tta.load_all) builds feature and label arrays for every split, test included; the trainer never uses test rows for training, batch order, validation or checkpoint selection.

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
ARMS = {"gru": 20, "gru_lb1": 1, "lstm": 20, "tcn": 20, "tf": 20}
N_DIST = 4
TRAIN_SITES = ROOT / "data" / "diagnostics" / "allmaps_geo_trainsites"


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
        elif arch == "tf":
            self.seq = TFSeq(d_in, K, cfg.get("width", 32), cfg.get("layers", 2), cfg.get("heads", 4), cfg.get("ff", 64))
        else:
            raise SystemExit(f"unknown arm {arch}")
        d = self.seq.out_dim
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, Xw, Sw, mask):
        return self.head(self.seq(self.enc(Xw, Sw, mask), mask)).squeeze(-1)


def add_train_site_distances(D):
    """Append the four train-only distance columns to the shared state. Each state row is matched to its distance row by
    (match_key, round, second) within its map. Matching by key, not by row count, keeps this working when --no_test_load has
    dropped the test-split rows from the state arrays; the distance file still holds them. Every state row must find exactly one
    distance row, or the run stops."""
    extra = np.zeros((D["state"].shape[0], N_DIST), dtype=np.float32)
    dist_cols = [f"st{i}" for i in range(N_DIST)]
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
    print(f"added {N_DIST} train-only site distance columns (matched by key for every map)", flush=True)
    return np.concatenate([D["state"], extra], axis=1)


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
    a = ap.parse_args()
    cfg = {k: v for k, v in (("width", a.width), ("layers", a.layers), ("heads", a.heads), ("ff", a.ff)) if v}
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
            tag = f"{arm}_seed{seed}"
            if (out / f"{tag}_metrics.json").exists():
                print(f"skip {tag}: already done", flush=True)
                continue
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = ArmModel(arm, state_dim, L, cfg).to(device)
            n_seq = sum(p.numel() for p in model.seq.parameters())
            n_total = sum(p.numel() for p in model.parameters())
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
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
                loss = loss_fn(model(Xw, Sw, m), label_t[r])
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
                "dist": a.dist, "state_dim": state_dim,
            }
            (out / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
            print(json.dumps({k: metrics[k] for k in ("tag", "best_step", "best_val_log_loss", "undertrained_flag")}),
                  flush=True)
            del model, opt
            torch.cuda.empty_cache() if device.type == "cuda" else None


if __name__ == "__main__":
    main()
