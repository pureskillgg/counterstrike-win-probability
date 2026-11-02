"""A graph / set model of one second of a round, trained on the per-player tensors (GPU).   Answers: do player-level structure, positions, view angles and a few seconds of
history beat the tabular snapshot model?

  ...tabicl\\python.exe -I -X utf8 scripts\\model\\gnn\\train_graph_model.py --name hybrid1 [--frames 1|3] [--hybrid 1|0] [--nopos] [--epochs 3] [--d 96] [--layers 3]

Rows, split and label are those of exp_positions.py (frozen match split, every other second of every used 2026 round). Model:
  nodes   the 10 player slots (5 CT, 5 T); input = player features (P) + side + the clock / bomb / map vector (GLOB), MLP -> d
  edges   every pair (i, j) among the living players, 8 features computed on the fly: distance, bearing of j in i's frame (cos, sin), bearing of i in j's frame (cos, sin), same team,
          height difference, inverse distance. Edge features enter the attention as a per-head bias, i.e. a graph attention with edge features / a set transformer with relative geometry
          (this is the 'heterogeneous' part: allied and enemy edges differ by the same-team flag; there is no map graph)
  layers  --layers rounds of masked attention (heads 4) + feed-forward, dead players masked out
  pooling mean over each team's living nodes + the clock / bomb / map vector (+ the final model's 59 aggregate inputs when --hybrid 1)
  time    --frames 3: the same encoder is applied to the rows 4 s, 2 s and 0 s before (same round, else repeated) and a GRU runs over the three graph embeddings
Loss: log loss, AdamW, cosine schedule, bf16 autocast, early stopping on validation matches (checkpoint with the best validation log loss). Writes
data/gnn/pred_<name>.npy (test probabilities) and data/gnn/result_<name>.json. nopos: positions, view direction, height and distances are zeroed (player-level inputs without geometry).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import wp_common as W  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
DEV = "cuda"
P_NAMES = None


class PairAttn(nn.Module):
    def __init__(self, d, heads, de, drop):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.e = nn.Sequential(nn.Linear(de, 48), nn.GELU(), nn.Linear(48, heads))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))

    def forward(self, x, mask, E):
        B, n, d = x.shape
        q = self.q(x).view(B, n, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, n, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, n, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) + self.e(E).permute(0, 3, 1, 2)
        att = att.masked_fill(~mask[:, None, None, :], -1e4)
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, n, d)
        x = self.ln1(x + self.o(y))
        return self.ln2(x + self.ff(x))


class GraphNet(nn.Module):
    def __init__(self, f_player, n_glob, n_g59, d, layers, heads, frames, hybrid, drop):
        super().__init__()
        self.frames, self.hybrid, self.d = frames, hybrid, d
        self.node = nn.Sequential(nn.Linear(f_player + 1 + n_glob, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([PairAttn(d, heads, 8, drop) for _ in range(layers)])
        self.glob = nn.Sequential(nn.Linear(n_glob, d), nn.GELU())
        self.g59 = nn.Sequential(nn.Linear(n_g59, d), nn.GELU(), nn.Linear(d, d), nn.GELU()) if hybrid else None
        d_frame = 3 * d
        self.gru = nn.GRU(d_frame, d_frame, batch_first=True) if frames > 1 else None
        d_head = d_frame * (2 if frames > 1 else 1) + (d if hybrid else 0)
        self.head = nn.Sequential(nn.Linear(d_head, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    @staticmethod
    def edges(P, GL):
        # P [B,10,F] float32 (px py z yaw_sin yaw_cos ...), GL [B,G]; world units via the map scale in GL[:,18] (world units per 3000 radar-normalised length)
        ws = GL[:, 18].view(-1, 1, 1) * 3000.0
        px, py, z = P[..., 0], P[..., 1], P[..., 2]
        ux, uy = P[..., 4], P[..., 3]  # view direction in world axes: (cos yaw, sin yaw)
        dx = (px[:, None, :] - px[:, :, None]) * ws  # from i to j, world x
        dy = -(py[:, None, :] - py[:, :, None]) * ws  # radar y points down; world y up
        dist = torch.sqrt(dx * dx + dy * dy + 1.0)
        nx, ny = dx / dist, dy / dist
        cos_ij = ux[:, :, None] * nx + uy[:, :, None] * ny
        sin_ij = ux[:, :, None] * ny - uy[:, :, None] * nx
        cos_ji = -(ux[:, None, :] * nx + uy[:, None, :] * ny)
        sin_ji = -(ux[:, None, :] * ny - uy[:, None, :] * nx)
        side = (torch.arange(10, device=P.device) >= 5).float()
        same = (side[:, None] == side[None, :]).float().expand(P.size(0), 10, 10)
        dz = (z[:, None, :] - z[:, :, None]) * 400.0 / 200.0
        return torch.stack([dist / 3000.0, cos_ij, sin_ij, cos_ji, sin_ji, same, dz, 1000.0 / dist], -1)

    def encode(self, P, mask, GL):
        B = P.size(0)
        side = (torch.arange(10, device=P.device) >= 5).float().view(1, 10, 1).expand(B, 10, 1)
        x = self.node(torch.cat([P, side, GL[:, None, :].expand(B, 10, GL.size(1))], -1))
        E = self.edges(P, GL)
        for blk in self.blocks:
            x = blk(x, mask, E)
        m = mask.float().unsqueeze(-1)
        sidef = (torch.arange(10, device=P.device) < 5).float().view(1, 10, 1)
        ct = (x * m * sidef).sum(1) / ((m * sidef).sum(1) + 1e-3)
        t = (x * m * (1 - sidef)).sum(1) / ((m * (1 - sidef)).sum(1) + 1e-3)
        return torch.cat([ct, t, self.glob(GL)], -1)

    def forward(self, frames, g59):
        # frames: list of (P, mask, GL) oldest first; the last one is the current row
        embs = [self.encode(*fr) for fr in frames]
        cur = embs[-1]
        if self.gru is not None:
            seq = torch.stack(embs, 1)
            _, h = self.gru(seq)
            z = torch.cat([cur, h[-1]], -1)
        else:
            z = cur
        if self.hybrid:
            z = torch.cat([z, self.g59(g59)], -1)
        return self.head(z).squeeze(-1)


def load_gpu(path, dtype, n=None, chunk=300000):
    a = np.load(path, mmap_mode="r")
    n = a.shape[0] if n is None else min(n, a.shape[0])
    out = torch.empty((n,) + a.shape[1:], dtype=dtype, device=DEV)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        out[i:j] = torch.from_numpy(np.ascontiguousarray(a[i:j])).to(DEV)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", type=int, default=1, choices=[1, 3])
    ap.add_argument("--hybrid", type=int, default=1)
    ap.add_argument("--nopos", action="store_true")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--rows", type=int, default=0, help="debug: use only the first N rows of the data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    g = os.path.join(ROOT, "data", "gnn")
    meta = json.load(open(os.path.join(g, "meta.json"), encoding="utf-8"))
    t0 = time.time()
    idx = pl.read_parquet(os.path.join(g, "index.parquet"))
    N = idx.height if not a.rows else a.rows
    sp = W.make_split("random", idx.select("match_key").unique().with_columns(pl.lit(1).alias("w_round"), pl.lit(0).alias("label_ct_win"), pl.lit("2026").alias("era"), pl.lit("").alias("match_date")), seed=0)
    code = {k: 0 for k in sp["train"]} | {k: 1 for k in sp["val"]} | {k: 2 for k in sp["test"]}
    split = np.array([code.get(k, -1) for k in idx["match_key"].to_list()], dtype=np.int8)[:N]
    y_np = idx["label_ct_win"].to_numpy().astype(np.float32)[:N]
    print(f"rows {N:,}: train {int((split == 0).sum()):,} val {int((split == 1).sum()):,} test {int((split == 2).sum()):,} ({time.time() - t0:.0f}s)", flush=True)
    P = load_gpu(os.path.join(g, "P.npy"), torch.float16, N)
    MK = load_gpu(os.path.join(g, "MASK.npy"), torch.uint8, N)
    GL = load_gpu(os.path.join(g, "GLOB.npy"), torch.float32, N)
    G59 = np.load(os.path.join(g, "G59.npy"), mmap_mode="r")
    tr_rows = np.where(split == 0)[0]
    samp = np.sort(np.random.default_rng(0).choice(tr_rows, size=min(500000, len(tr_rows)), replace=False))
    sub = np.nan_to_num(np.asarray(G59[samp], dtype=np.float32))
    mu, sd = sub.mean(0), sub.std(0) + 1e-6
    G59g = torch.empty((N, G59.shape[1]), dtype=torch.float16, device=DEV)
    mu_t, sd_t = torch.from_numpy(mu).to(DEV), torch.from_numpy(sd).to(DEV)
    for i in range(0, N, 300000):
        c = torch.from_numpy(np.nan_to_num(np.asarray(G59[i:min(i + 300000, N)], dtype=np.float32))).to(DEV)
        G59g[i:i + len(c)] = ((c - mu_t) / sd_t).clamp(-8, 8).half()
    del sub
    prev1 = torch.from_numpy(idx["prev1"].to_numpy()[:N]).to(DEV)
    prev2 = torch.from_numpy(idx["prev2"].to_numpy()[:N]).to(DEV)
    y = torch.from_numpy(y_np).to(DEV)
    pos_cols = [meta["P_names"].index(c) for c in ("px", "py", "z", "yaw_sin", "yaw_cos", "dist_bomb", "dist_site")]
    print(f"on GPU: P {P.shape} {P.element_size() * P.numel() / 1e9:.1f} GB ({time.time() - t0:.0f}s)", flush=True)

    def batch(rows):
        rows = rows.long()
        fr = []
        ids = [rows]
        if a.frames == 3:
            p1 = prev1[rows]
            p1 = torch.where(p1 >= 0, p1, rows)
            p2 = prev2[rows]
            p2 = torch.where(p2 >= 0, p2, p1)
            ids = [p2, p1, rows]
        for r in ids:
            Pb = P[r].float()
            if a.nopos:
                Pb[..., pos_cols] = 0.0
            fr.append((Pb, MK[r].bool(), GL[r]))
        return fr, G59g[rows].float(), y[rows]

    model = GraphNet(len(meta["P_names"]), len(meta["GLOB_names"]), G59.shape[1], a.d, a.layers, a.heads, a.frames, bool(a.hybrid), a.drop).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(tr_rows) // a.batch
    total = int(a.epochs * steps_per_epoch) if not a.max_steps else a.max_steps
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.08, anneal_strategy="cos")
    tr_t = torch.from_numpy(tr_rows).to(DEV)
    va_t = torch.from_numpy(np.where(split == 1)[0]).to(DEV)
    te_np = np.where(split == 2)[0]
    te_t = torch.from_numpy(te_np).to(DEV)
    print(f"model {n_par:,} parameters; {total:,} steps ({a.epochs} epochs of {steps_per_epoch:,}); frames {a.frames}, hybrid {a.hybrid}, nopos {a.nopos}", flush=True)

    @torch.no_grad()
    def predict(rows, bs=16384):
        model.eval()
        out = []
        for i in range(0, len(rows), bs):
            fr, g59, _ = batch(rows[i:i + bs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(model(fr, g59).float()))
        model.train()
        return torch.cat(out)

    def logloss(p, rows):
        yy = y[rows.long()]
        p = p.clamp(1e-6, 1 - 1e-6)
        return float(-(yy * p.log() + (1 - yy) * (1 - p).log()).mean())

    best, best_state, step, bad = 9.0, None, 0, 0
    hist = []
    perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
    ptr = 0
    model.train()
    t1 = time.time()
    while step < total:
        if ptr + a.batch > len(perm):
            perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
            ptr = 0
        rows = perm[ptr:ptr + a.batch]
        ptr += a.batch
        fr, g59, yb = batch(rows)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(fr, g59)
        loss = Fn.binary_cross_entropy_with_logits(logit.float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % a.eval_every == 0 or step == total:
            vl = logloss(predict(va_t), va_t)
            hist.append({"step": step, "train_loss": round(float(loss), 5), "val_logloss": round(vl, 5), "seconds": round(time.time() - t1)})
            print(f"step {step}/{total}: train {float(loss):.4f}  val {vl:.5f}  ({time.time() - t1:.0f}s)", flush=True)
            if vl < best:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
    model.load_state_dict(best_state)
    pte = predict(te_t).cpu().numpy().astype(np.float64)
    np.save(os.path.join(g, f"pred_{a.name}.npy"), pte)
    np.save(os.path.join(g, f"pred_{a.name}_rows.npy"), te_np)
    sub = idx[te_np].select("match_key", "round", "tick", "w_round")
    r = W.evaluate(sub, y_np[te_np].astype(np.int64), pte)
    res = {"name": a.name, "args": vars(a), "parameters": n_par, "best_val_logloss": best, "test": {k: r[k] for k in ("n_matches", "n_rows", "logloss", "logloss_ci95", "auc", "ece10", "ece10_ci95")},
           "history": hist, "train_seconds": round(time.time() - t1)}
    json.dump(res, open(os.path.join(g, f"result_{a.name}.json"), "w", encoding="utf-8"), indent=1)
    print(f"TEST {a.name}: log loss {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]}  AUC {r['auc']:.4f}  ECE {r['ece10']:.4f}  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
