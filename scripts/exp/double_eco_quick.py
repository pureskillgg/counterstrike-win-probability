"""Quick double-eco count from round_buy_types (150-match sample). 'Double eco' = a team is eco in two consecutive
non-pistol rounds. Team tracking across the halftime swap (24-round matches, rounds 1-24 only). Buy classes use the
UNVERIFIED default cut-offs (eco < 10,000 team equipment at freeze end). Also eco_vs_eco = both teams eco in one round."""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import polars as pl

d = pl.read_parquet(r"G:\pureskill\csds-cs2\data\state_sample\round_buy_types.parquet").filter(pl.col("round") <= 24)
rows = d.select(["match_key", "round", "ct_buy", "t_buy", "ct_win", "is_pistol"]).to_dicts()
tr = {}
for x in rows:
    r = x["round"]
    ct_win = int(x["ct_win"])
    a_buy, b_buy = (x["t_buy"], x["ct_buy"]) if r <= 12 else (x["ct_buy"], x["t_buy"])
    a_win = (1 - ct_win) if r <= 12 else ct_win
    tr[(x["match_key"], "A", r)] = (a_buy, a_win)
    tr[(x["match_key"], "B", r)] = (b_buy, 1 - a_win)

pairs = []
for (m, t, r), (buy, win) in tr.items():
    nxt = tr.get((m, t, r + 1))
    if nxt is None or r < 2 or r in (12, 13) or r + 1 == 13:
        continue
    pairs.append((m, buy, win, nxt[0], nxt[1]))
print("consecutive non-pistol team-round pairs:", len(pairs))
eco_first = [p for p in pairs if p[1] == "eco"]
double = [p for p in eco_first if p[3] == "eco"]
print("team eco at r:", len(eco_first), "| eco again at r+1 (double eco):", len(double),
      f"| P(eco again | eco) = {len(double) / max(len(eco_first), 1):.3f}")
base_eco = np.mean([p[3] == "eco" for p in pairs if p[1] != "eco"]) if any(p[1] != "eco" for p in pairs) else float("nan")
print(f"P(eco at r+1 | not eco at r) = {base_eco:.3f}")


def rate(sel, k):
    ms = sorted({p[0] for p in sel})
    if not ms:
        return None
    um = {m: i for i, m in enumerate(ms)}
    mid = np.array([um[p[0]] for p in sel])
    y = np.array([p[k] for p in sel], dtype=float)
    M = len(ms)
    idx = np.random.default_rng(0).integers(0, M, size=(1000, M))
    sn = np.bincount(mid, weights=y, minlength=M)
    sd = np.bincount(mid, minlength=M).astype(float)
    rep = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1)
    return float(y.mean()), [float(np.percentile(rep, 2.5)), float(np.percentile(rep, 97.5))], len(sel), M


for label, sel, k in (("win rate in the FIRST eco round of a double eco", double, 2),
                      ("win rate in the SECOND eco round of a double eco", double, 4),
                      ("win rate in an eco round after a non-eco round", [p for p in pairs if p[1] != "eco" and p[3] == "eco"], 4),
                      ("win rate in a full-buy round right after a double eco (r+2 not tracked here)", [], 4)):
    if sel:
        print(label, rate(sel, k))

# what the team buys at r+2 after a double eco
ext = []
for (m, t, r), (buy, win) in tr.items():
    n1, n2 = tr.get((m, t, r + 1)), tr.get((m, t, r + 2))
    if n1 and n2 and buy == "eco" and n1[0] == "eco" and r >= 2 and r not in (11, 12, 13) and r + 1 != 13 and r + 2 != 13:
        ext.append(n2[0])
if ext:
    from collections import Counter

    print("buy at r+2 after eco, eco:", dict(Counter(ext)), "n =", len(ext))

# both teams eco in the same round
both = d.filter((pl.col("ct_buy") == "eco") & (pl.col("t_buy") == "eco") & (~pl.col("is_pistol")))
print("both teams eco in one non-pistol round:", both.height, "of", d.filter(~pl.col("is_pistol")).height)
