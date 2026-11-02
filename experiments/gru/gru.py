import json, ast, gc, os, time
import numpy as np, polars as pl, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = "/mnt/g/pureskill/csds-cs2"
SCR = "/mnt/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
OUT = SCR + "/gru"
MODE = os.environ.get("MODE", "full")
MAXM = int(os.environ.get("MAXM", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
H = int(os.environ.get("HIDDEN", "64"))
PAT = int(os.environ.get("PATIENCE", "3"))
SUF = "_smoke" if MAXM else ""
W = 20
BS_R, BS_W = 128, 8192

torch.manual_seed(0)
np.random.seed(0)
dev = torch.device("cuda")
OFFS = torch.arange(W - 1, -1, -1, device=dev)

meta = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]
KEYS = ["match_key", "round", "second"]
CACHE = ROOT + "/data/combined/cache_snapshot_2026.parquet"


def load(split):
    keys = sorted(k for k, v in man.items() if v == split and "/2026/" in k)
    if MAXM:
        keys = keys[:MAXM]
    df = (pl.scan_parquet(CACHE).select(KEYS + ["label_ct_win"] + FEATS)
          .filter(pl.col("match_key").is_in(keys)).sort(KEYS).collect())
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in FEATS])


def structure(df):
    g = (df.with_row_index("row").group_by(["match_key", "round"], maintain_order=True)
         .agg([pl.col("row").min().alias("start"), pl.len().alias("length")]))
    return g["start"].to_numpy().astype(np.int64), g["length"].to_numpy().astype(np.int64)


print("loading train", flush=True)
tr_df = load("train")
mu = np.array(tr_df.select(FEATS).mean().row(0), dtype=np.float32)
sd = np.maximum(np.array(tr_df.select(FEATS).std().row(0), dtype=np.float32), 1e-6)


def prep(df):
    starts, lengths = structure(df)
    X = ((df.select(FEATS).to_numpy().astype(np.float32) - mu) / sd).astype(np.float32)
    Y = df["label_ct_win"].to_numpy().astype(np.float32)
    keys = df.select(KEYS + ["label_ct_win"])
    d = {"X": torch.from_numpy(X).to(dev), "Y": torch.from_numpy(Y).to(dev),
         "starts": torch.from_numpy(starts).to(dev), "lens": torch.from_numpy(lengths).to(dev),
         "rs": torch.from_numpy(np.repeat(starts, lengths)).to(dev), "N": len(Y)}
    del X, Y
    gc.collect()
    return d, keys


dtr, _ = prep(tr_df)
del tr_df
gc.collect()
print("train rows " + str(dtr["N"]) + ", rounds " + str(len(dtr["starts"])) + ", max round len " + str(int(dtr["lens"].max())), flush=True)
print("loading val", flush=True)
dva, _ = prep(load("val"))
print("loading test", flush=True)
dte, kte = prep(load("test"))
print("val rows " + str(dva["N"]) + ", test rows " + str(dte["N"]) + ", GPU mem " + str(round(torch.cuda.memory_allocated() / 1e9, 2)) + " GB", flush=True)
F = len(FEATS)


def batch_full(d, idx):
    s = d["starts"][idx]
    l = d["lens"][idx]
    L = int(l.max())
    t = torch.arange(L, device=dev)
    m = t[None, :] < l[:, None]
    ii = torch.where(m, s[:, None] + t[None, :], s[:, None])
    return d["X"][ii], d["Y"][ii], m, ii


def batch_win(d, ridx):
    rs = d["rs"][ridx]
    pos = ridx - rs
    ii = ridx[:, None] - OFFS[None, :]
    valid = OFFS[None, :] <= pos[:, None]
    ii = torch.where(valid, ii, rs[:, None])
    x = d["X"][ii] * valid[..., None]
    x = torch.cat([x, valid[..., None].float()], dim=-1)
    return x, d["Y"][ridx]


class Net(nn.Module):
    def __init__(self, fin, h, last_only):
        super().__init__()
        self.last_only = last_only
        self.gru = nn.GRU(fin, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1]).squeeze(-1) if self.last_only else self.head(o).squeeze(-1)


model = Net(F if MODE == "full" else F + 1, H, last_only=(MODE != "full")).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = nn.functional.binary_cross_entropy_with_logits


def train_epoch():
    model.train()
    tot, n = 0.0, 0
    if MODE == "full":
        order = torch.randperm(len(dtr["starts"]), device=dev)
        for b in range(0, len(order), BS_R):
            x, y, m, _ = batch_full(dtr, order[b:b + BS_R])
            loss = bce(model(x)[m], y[m])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n += 1
    else:
        order = torch.randperm(dtr["N"], device=dev)
        for b in range(0, dtr["N"], BS_W):
            x, y = batch_win(dtr, order[b:b + BS_W])
            loss = bce(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n += 1
    return tot / max(n, 1)


@torch.no_grad()
def predict(d):
    model.eval()
    P = torch.empty(d["N"], device=dev)
    if MODE == "full":
        nr = len(d["starts"])
        for b in range(0, nr, 512):
            idx = torch.arange(b, min(b + 512, nr), device=dev)
            x, _, m, ii = batch_full(d, idx)
            P[ii[m]] = torch.sigmoid(model(x)[m])
    else:
        for b in range(0, d["N"], 32768):
            ridx = torch.arange(b, min(b + 32768, d["N"]), device=dev)
            x, _ = batch_win(d, ridx)
            P[ridx] = torch.sigmoid(model(x))
    return P.cpu().numpy()


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e


yv = dva["Y"].cpu().numpy()
best_ll, best_state, bad = 1e9, None, 0
for ep in range(EPOCHS):
    t = time.time()
    tl = train_epoch()
    pv = predict(dva)
    vll = float(ll(yv, pv).mean())
    print("epoch " + str(ep) + ": train loss " + str(round(tl, 5)) + "  val LL " + str(round(vll, 5)) + "  (" + str(int(time.time() - t)) + "s)", flush=True)
    if vll < best_ll - 1e-5:
        best_ll, bad = vll, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= PAT:
            print("early stop", flush=True)
            break

model.load_state_dict(best_state)
pt = predict(dte)
yt = dte["Y"].cpu().numpy()
res = {"mode": MODE, "hidden": H, "best_val_logloss": best_ll, "test_logloss": float(ll(yt, pt).mean()),
       "test_auc": float(roc_auc_score(yt, pt)), "test_ece10": float(ece(yt, pt)), "test_rows": int(dte["N"])}
print(json.dumps(res, indent=1), flush=True)
out = kte.with_columns(pl.Series("p_gru", pt))
out.write_parquet(OUT + "/gru_" + MODE + "_test_preds" + SUF + ".parquet")
json.dump(res, open(OUT + "/gru_" + MODE + "_results" + SUF + ".json", "w"), indent=1)

LAGP = SCR + "/lag/lag_test_preds.parquet"
if not MAXM and os.path.exists(LAGP):
    base = pl.read_parquet(LAGP).select(KEYS + ["p_base59"])
    j = out.join(base, on=KEYS, how="inner")
    yj = j["label_ct_win"].cast(pl.Float64).to_numpy()
    pg = j["p_gru"].to_numpy()
    pb = j["p_base59"].to_numpy()
    d = ll(yj, pg) - ll(yj, pb)
    codes, inv = np.unique(j["match_key"].to_numpy(), return_inverse=True)
    sum_d = np.bincount(inv, weights=d)
    cnt = np.bincount(inv)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(codes), size=(2000, len(codes)))
    boot = sum_d[idx].sum(1) / cnt[idx].sum(1)
    pair = {"common_rows": j.height, "base59_logloss_common": float(ll(yj, pb).mean()),
            "gru_logloss_common": float(ll(yj, pg).mean()), "delta_gru_minus_base": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    res["paired_vs_base59"] = pair
    json.dump(res, open(OUT + "/gru_" + MODE + "_results" + SUF + ".json", "w"), indent=1)
    print("paired: " + json.dumps(pair), flush=True)
