"""D8 follow-up: is map_name the right key, or is the per-match halftime schedule?

Halftime signature per match (same logic as scripts/swarm2/D3_swap_signature / D3_swap_format_map):
round_state round_freeze_end t_score/ct_score per round; a side swap is flagged between round r and r+1
when the score deltas are not a normal single-round win. sig = tuple of rounds r after which a swap is flagged.

Tests:
  F-1 signature x map counts (day 12-25 matches).
  F-2 round-13 bomb mismatch (same universe as D8_map_test: player_count==20, dedup, bomb outcome) by signature,
      with day control (stratified permutation): sig | day vs sig unadjusted; map | day inside the (12,) group.
  F-3 D4 complete-match classes (same universe as scripts/swarm2/D4_map_day.py) x signature x map.
      Re-reconcile header totals using each match's detected swap schedule instead of the assumed MR12.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D8_format_check.py
"""
import collections
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(16)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
OUTCOMES = ["bomb_defused", "bomb_exploded"]
B_PERM = 5000    # ASSUMPTION: permutation replicates
SEED = 20261013  # ASSUMPTION: fixed RNG seed
RNG = np.random.default_rng(SEED)


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read_dir(fd, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files_of(fd)], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(BASE / f"basic_{ch}{SUF}", cols)


def het_stat(g, s, y, G, S):
    ns = np.bincount(s, minlength=S).astype(float)
    ys = np.bincount(s, weights=y, minlength=S)
    ps = np.divide(ys, ns, out=np.zeros(S), where=ns > 0)
    pe = ps[s]
    og = np.bincount(g, weights=y, minlength=G)
    eg = np.bincount(g, weights=pe, minlength=G)
    vg = np.bincount(g, weights=pe * (1 - pe), minlength=G)
    ok = vg > 0
    d = og[ok] - eg[ok]
    return float(np.sum(d * d / vg[ok])), int(ok.sum())


def permute_within(g, s):
    a = np.argsort(s, kind="stable")
    sa = s[a]
    pos = np.lexsort((RNG.random(len(g)), sa))
    out = np.empty_like(g)
    out[a] = g[a][pos]
    return out


def perm_test(g, s, y, G, S, B=B_PERM):
    obs, k = het_stat(g, s, y, G, S)
    null = np.empty(B)
    for b in range(B):
        null[b] = het_stat(permute_within(g, s), s, y, G, S)[0]
    return obs, k, (1 + np.sum(null >= obs - 1e-9)) / (B + 1)


def codes(vals):
    u = sorted(set(vals))
    idx = {v: i for i, v in enumerate(u)}
    return np.array([idx[v] for v in vals], dtype=np.int64), u


# ---------------------------------------------------------------- signature (D3 logic)
H = read_dir(BASE / "header", ["key", "map_name", "t_starters_score_final", "ct_starters_score_final"])
hmap = dict(zip(H["key"].to_list(), H["map_name"].to_list()))
htot = {k: (int(a), int(b)) for k, a, b in zip(H["key"].to_list(), H["t_starters_score_final"].to_list(),
                                                H["ct_starters_score_final"].to_list())}
RE_raw = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE = RE_raw.unique(["match_key", "round"], keep="first")
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)
sig = {}
for m, rs in by_m.items():
    swaps = []
    for r in sorted(rs):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0) or (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0:
            swaps.append(r)
    sig[m] = tuple(swaps)
del RS, fe, live

SIG = pl.DataFrame({"match_key": list(sig.keys()), "sig": [str(sig[m]) for m in sig.keys()]})
SIG = SIG.with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int32).alias("day"),
    pl.col("match_key").map_elements(lambda m: hmap.get(m), return_dtype=pl.String).alias("map_name"),
)
S12 = SIG.filter(pl.col("day") >= 12)
print("=== F-1 signature x map (day 12-25 matches) ===")
print("matches by signature:", S12.group_by("sig").len().sort("len", descending=True).rows())
print(S12.group_by(["map_name", "sig"]).len().pivot(on="sig", index="map_name", values="len").fill_null(0).sort("map_name"))

# ---------------------------------------------------------------- F-2 round 13 by signature
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
ev = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
OUT = ev.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside")
).select(["match_key", "round", "bside"])
RE20u = RE_raw.filter(pl.col("player_count") == 20).unique(["match_key", "round"], keep="first")
J = (RE20u.join(OUT, on=["match_key", "round"], how="inner")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int32).alias("day"))
     .with_columns((pl.col("bside") != pl.col("winner_team_code")).alias("mm"))
     .join(SIG.select(["match_key", "sig"]), on="match_key", how="left"))
print(f"\nJ rounds with bomb outcome: {J.height}; mismatches {int(J['mm'].sum())}; null sig: {J['sig'].null_count()}")
R13 = J.filter((pl.col("round") == 13) & (pl.col("day") >= 12))
R13 = R13.with_columns(pl.col("match_key").map_elements(lambda m: hmap.get(m), return_dtype=pl.String).alias("map_name"))
print(f"round-13 planted rows days 12-25: {R13.height}; mismatches {int(R13['mm'].sum())}")
print("\nround-13 mismatch by signature (days 12-25):")
print(R13.group_by("sig").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"), pl.col("mm").mean().alias("rate"))
      .sort("n", descending=True))
print("\nround-13 mismatch by signature x map (mm/n):")
cells = {(r["sig"], r["map_name"]): (r["mm"], r["n"]) for r in
         R13.group_by(["sig", "map_name"]).agg(pl.col("mm").sum().alias("mm"), pl.len().alias("n")).iter_rows(named=True)}
sigs_r13 = sorted({k[0] for k in cells})
maps_r13 = sorted({k[1] for k in cells})
print(f"{'sig':<12}" + "".join(f"{m[:10]:>12}" for m in maps_r13))
for sg in sigs_r13:
    print(f"{sg:<12}" + "".join(f"{str(cells[(sg, m)][0]) + '/' + str(cells[(sg, m)][1]) if (sg, m) in cells else '-':>12}"
                                for m in maps_r13))

dy = R13["day"].to_list()
dg_codes, dy_u = codes(dy)
sg_codes, sg_u = codes(R13["sig"].to_list())
mp_codes, mp_u = codes(R13["map_name"].to_list())
y = R13["mm"].cast(pl.Float64).to_numpy()
zeros = np.zeros(len(y), dtype=np.int64)
o1, k1, p1 = perm_test(sg_codes, zeros, y, len(sg_u), 1)
o2, k2, p2 = perm_test(sg_codes, dg_codes, y, len(sg_u), len(dy_u))
print(f"\nF-2 tests (round 13, days 12-25, n={len(y)}, B={B_PERM}):")
print(f"  signature unadjusted: stat={o1:.2f} groups={k1} perm p={p1:.4f}")
print(f"  signature | day:   stat={o2:.2f} groups={k2} perm p={p2:.4f}")
m12 = (R13["sig"] == "(12,)").to_numpy()
sub = R13.filter(pl.col("sig") == "(12,)")
if sub.height:
    yy = sub["mm"].cast(pl.Float64).to_numpy()
    dd, _ = codes(sub["day"].to_list())
    mm_, _ = codes(sub["map_name"].to_list())
    o3, k3, p3 = perm_test(mm_, dd, yy, len(set(mm_.tolist())), len(set(dd.tolist())))
    o4, k4, p4 = perm_test(mm_, np.zeros(len(yy), dtype=np.int64), yy, len(set(mm_.tolist())), 1)
    print(f"  within sig (12,) only (n={sub.height}, mismatches {int(sub['mm'].sum())}): "
          f"map unadjusted stat={o4:.2f} perm p={p4:.4f}; map | day stat={o3:.2f} perm p={p3:.4f}")

# ---------------------------------------------------------------- F-3 reconcile header totals with detected schedule
RX = RE.select(["match_key", "round", "winner_team_code"])
outs4 = OUT.select(["match_key", "round", "bside"])
R4 = RX.join(outs4, on=["match_key", "round"], how="left")
mm13 = set(R4.filter((pl.col("round") == 13) & pl.col("bside").is_not_null()
                     & (pl.col("bside") != pl.col("winner_team_code")))["match_key"].to_list())
rows_by = {}
for r in R4.sort(["match_key", "round"]).iter_rows(named=True):
    rows_by.setdefault(r["match_key"], []).append(r)


def side_a(r, swaps):
    """Team that started on T: T (2) before the first swap, then alternates after each swap round."""
    k = sum(1 for s in swaps if s < r)
    return T if k % 2 == 0 else CT


def wins_a(rounds, swaps):
    return sum(1 for x in rounds if int(x["winner_team_code"]) == side_a(int(x["round"]), swaps))


out = []
for m, rounds in rows_by.items():
    if m not in htot:
        continue
    tA, tB = htot[m]
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    day = int(m.split("/")[3])
    r13 = next((x for x in rounds if int(x["round"]) == 13), None)
    # D4 baseline: MR12 swap
    wa12 = wins_a(rounds, (12,))
    misfit12 = (wa12, n - wa12) != (tA, tB)
    flip12 = False
    if misfit12 and r13 is not None:
        nwa = wa12 - 1 if int(r13["winner_team_code"]) == side_a(13, (12,)) else wa12 + 1
        flip12 = (nwa, n - nwa) == (tA, tB)
    cls = "affected" if m in mm13 else ("r13_flip_misfit" if flip12 else ("other_misfit" if misfit12 else "fits"))
    # detected schedule
    sw = sig.get(m, ())
    wad = wins_a(rounds, sw)
    fit_det = (wad, n - wad) == (tA, tB)
    fit_det_any = fit_det
    if not fit_det and r13 is not None:
        sA13 = side_a(13, sw)
        nwa = wad - 1 if int(r13["winner_team_code"]) == sA13 else wad + 1
        fit_det_any = (nwa, n - nwa) == (tA, tB)
    out.append({"match_key": m, "map_name": hmap.get(m), "day": day, "cls": cls, "sig": str(sw), "n": n,
                "fit_det": fit_det, "fit_det_any": fit_det_any})
D = pl.DataFrame(out)
print(f"\n=== F-3 complete regulation matches: {D.height} (D4 log: 1246) ===")
print("class counts (D4 log: affected 211, r13_flip_misfit 825, other_misfit 116, fits 94):")
print(D.group_by("cls").len().sort("cls"))
print("\nclass x signature (complete matches):")
print(D.group_by(["cls", "sig"]).len().sort(["cls", "len"], descending=[False, True]))
print("\nother_misfit matches by map and signature:")
print(D.filter(pl.col("cls") == "other_misfit").group_by(["map_name", "sig"]).len().sort(["map_name", "sig"]))
om = D.filter(pl.col("cls") == "other_misfit")
print(f"\nother_misfit: {om.height}; reconcile under detected schedule (no flip): {int(om['fit_det'].sum())}; "
      f"under detected schedule with one r13 flip: {int(om['fit_det_any'].sum())}")
print("\nfit under detected schedule by signature (all complete matches): ")
print(D.group_by("sig").agg(pl.len().alias("n"), pl.col("fit_det").sum().alias("fit_det"),
                           pl.col("fit_det_any").sum().alias("fit_det_any")).sort("n", descending=True))
print("\nDONE")
