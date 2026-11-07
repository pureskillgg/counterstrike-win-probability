"""D8 robustness: is the residual round-13 map effect driven by one map, and do duplicate matches inflate it?

  R-1 header duplicates: unique match_id / sharecode / demo_id across the 2024 header, and among the 605
      day 12-25 round-13 planted rows.
  R-2 leave-one-map-out: round-13 day-stratified map permutation test (days 12-25) with each map removed.
  R-3 Mantel-Haenszel odds ratio, each map vs the other maps, stratified by day (days 12-25, round 13).
Same universe as scripts/swarm2/D8_map_test.py (player_count == 20, dedup, bomb outcome).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D8_residual_check.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
OUTCOMES = ["bomb_defused", "bomb_exploded"]
B_PERM = 2000    # ASSUMPTION: permutations for leave-one-out
SEED = 20261014  # ASSUMPTION: fixed RNG seed
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
    return float(np.sum(d * d / vg[ok]))


def permute_within(g, s):
    a = np.argsort(s, kind="stable")
    sa = s[a]
    pos = np.lexsort((RNG.random(len(g)), sa))
    out = np.empty_like(g)
    out[a] = g[a][pos]
    return out


def perm_p(g, s, y, G, S, B=B_PERM):
    obs = het_stat(g, s, y, G, S)
    null = np.array([het_stat(permute_within(g, s), s, y, G, S) for _ in range(B)])
    return obs, (1 + np.sum(null >= obs - 1e-9)) / (B + 1)


def codes(vals):
    u = sorted(set(vals))
    idx = {v: i for i, v in enumerate(u)}
    return np.array([idx[v] for v in vals], dtype=np.int64), u


H = read_dir(BASE / "header", ["key", "map_name", "match_id", "sharecode", "demo_id", "match_date",
                               "t_starters_score_final", "ct_starters_score_final"])
print("=== R-1 header duplicates ===")
print(f"header rows {H.height}; unique key {H['key'].n_unique()}; unique match_id {H['match_id'].n_unique()}; "
      f"unique sharecode {H['sharecode'].n_unique()}; unique demo_id {H['demo_id'].n_unique()}")
dupe_key = H.group_by(["match_date", "map_name", "t_starters_score_final", "ct_starters_score_final"]).len()
print(f"groups sharing (match_date, map, scores): {dupe_key.filter(pl.col('len') > 1).height} "
      f"covering {int(dupe_key.filter(pl.col('len') > 1)['len'].sum())} headers")

RE_raw = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE20u = RE_raw.filter(pl.col("player_count") == 20).unique(["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
ev = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
OUT = ev.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside")
).select(["match_key", "round", "bside"])
hmap = dict(zip(H["key"].to_list(), H["map_name"].to_list()))
hid = dict(zip(H["key"].to_list(), zip(H["match_id"].to_list(), H["sharecode"].to_list(), H["demo_id"].to_list())))
R13 = (RE20u.join(OUT, on=["match_key", "round"], how="inner")
       .filter((pl.col("round") == 13))
       .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int32).alias("day"))
       .filter(pl.col("day") >= 12)
       .with_columns((pl.col("bside") != pl.col("winner_team_code")).alias("mm")))
R13 = R13.with_columns(pl.col("match_key").map_elements(lambda m: hmap.get(m), return_dtype=pl.String).alias("map_name"),
                       pl.col("match_key").map_elements(lambda m: hid.get(m, (None, None, None))[0], return_dtype=pl.String).alias("match_id"),
                       pl.col("match_key").map_elements(lambda m: hid.get(m, (None, None, None))[1], return_dtype=pl.String).alias("sharecode"),
                       pl.col("match_key").map_elements(lambda m: hid.get(m, (None, None, None))[2], return_dtype=pl.String).alias("demo_id"))
print(f"day 12-25 round-13 planted rows {R13.height}; unique match_id {R13['match_id'].n_unique()}; "
      f"unique sharecode {R13['sharecode'].n_unique()}; unique demo_id {R13['demo_id'].n_unique()}")

print("\n=== R-2 leave-one-map-out (round 13, days 12-25, day-stratified map permutation) ===")
maps = sorted(R13["map_name"].unique().to_list())
dd_all, dd_u = codes(R13["day"].to_list())
y_all = R13["mm"].cast(pl.Float64).to_numpy()
mp_all = R13["map_name"].to_list()
obs_all, p_all = perm_p(codes(mp_all)[0], dd_all, y_all, len(maps), len(dd_u))
print(f"all 8 maps: n={len(y_all)} stat={obs_all:.2f} perm p={p_all:.4f} (B={B_PERM})")
for drop in maps:
    keep = np.array([m != drop for m in mp_all])
    sub_maps = [m for m in mp_all if m != drop]
    gcodes, gu = codes(sub_maps)
    dsub, du = codes([d for d, k in zip(R13["day"].to_list(), keep) if k])
    ysub = y_all[keep]
    o, p = perm_p(gcodes, dsub, ysub, len(gu), len(du))
    print(f"drop {drop:<12} n={int(keep.sum()):>4} stat={o:.2f} perm p={p:.4f}  maps left={len(gu)}")

print("\n=== R-3 Mantel-Haenszel OR, each map vs other maps, stratified by day (round 13, days 12-25) ===")
days = sorted(set(R13["day"].to_list()))
for mp in maps:
    num = den = 0.0
    a_tot = b_tot = c_tot = d_tot = 0
    for d in days:
        sub = R13.filter(pl.col("day") == d)
        a = int(sub.filter((pl.col("map_name") == mp) & pl.col("mm")).height)
        b = int(sub.filter((pl.col("map_name") == mp) & ~pl.col("mm")).height)
        c = int(sub.filter((pl.col("map_name") != mp) & pl.col("mm")).height)
        dn = int(sub.filter((pl.col("map_name") != mp) & ~pl.col("mm")).height)
        n = a + b + c + dn
        if n == 0:
            continue
        num += a * dn / n
        den += b * c / n
    OR = num / den if den > 0 else float("nan")
    print(f"  {mp:<12} MH OR (map vs rest, day-stratified) = {OR:.2f}")
print("\nDONE")
