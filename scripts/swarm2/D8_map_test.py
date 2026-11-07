"""D8 hypothesis test: is the 2024 round-13 side mismatch concentrated on particular maps?

Universe (same as scripts/swarm/e2_side_check_r13.py): 2024 tome round_end rows with player_count == 20,
joined to bomb_state bomb_defused / bomb_exploded outcomes. bside = 3 (CT) if a defuse row exists, else 2 (T).
mismatch = bside != winner_team_code. map_name comes from the header (header.key == match_key).
day = the DD in match_key csds/2024/02/DD/...

Tests:
  D8-0 validation: reproduce e2 totals (15,436 rounds, 499 mismatches) and check duplicates.
  D8-1 round-13 mismatch by map (days 12-25 and days 01-07), chi-square heterogeneity.
  D8-2 round-13 map effect with day controlled: stratified permutation tests (map | day vs map unadjusted),
       the reverse (day | map vs day unadjusted), day-standardised O/E per map, early vs late days within map.
  D8-3 other rounds (round != 13) by map in days 12-25 and days 01-07, same permutation tests; round 25 by map.
  D8-4 map x day grid of round-13 mismatches (mm/n) for days 12-25.
  D8-5 secondary: match-level non-round-13 header misfit (same classes as scripts/swarm2/D4_map_day.py),
       by map, with day-controlled permutation test.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D8_map_test.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(16)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
OUTCOMES = ["bomb_defused", "bomb_exploded"]
B_PERM = 5000    # ASSUMPTION: permutation replicates per test
SEED = 20261013  # ASSUMPTION: fixed RNG seed for reproducibility
RNG = np.random.default_rng(SEED)


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read_dir(fd, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files_of(fd)], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(BASE / f"basic_{ch}{SUF}", cols)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def het_stat(g, s, y, G, S):
    """Sum over groups g of (O_g - E_g)^2 / V_g, with E from stratum (s) mean rates.

    Groups g are compared after controlling for strata s (S=1 means no control).
    """
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
    """Shuffle group labels g uniformly within each stratum s (strata and their sizes preserved)."""
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
    p = (1 + np.sum(null >= obs - 1e-9)) / (B + 1)
    return obs, k, p


def tabg(df, keys):
    return (df.group_by(keys)
            .agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"), pl.col("mm").mean().alias("rate"))
            .sort(keys))


def print_ci(df, key):
    print(f"{key:<13}{'n':>6}{'mm':>6}{'rate':>9}   95% Wilson CI")
    for r in df.sort(key).iter_rows(named=True):
        lo, hi = wilson(r["mm"], r["n"])
        print(f"{str(r[key]):<13}{r['n']:>6}{r['mm']:>6}{r['mm'] / r['n']:>9.3f}   [{lo:.3f}, {hi:.3f}]")


def chi2_het(df):
    """Chi-square of map x mismatch (2 columns). Expected counts reported, not corrected."""
    piv = df.group_by("map_name").agg(pl.col("mm").sum().alias("mm"), pl.len().alias("n")).sort("map_name")
    tbl = np.column_stack([piv["mm"].to_numpy(), (piv["n"] - piv["mm"]).to_numpy()])
    chi2, p, dof, exp = stats.chi2_contingency(tbl, correction=False)
    return chi2, dof, p, float(exp.min()), int((exp < 5).sum()), exp.size


def codes(vals):
    u = sorted(set(vals))
    idx = {v: i for i, v in enumerate(u)}
    return np.array([idx[v] for v in vals], dtype=np.int64), u


# ---------------------------------------------------------------- load
H = read_dir(BASE / "header", ["key", "map_name", "t_starters_score_final", "ct_starters_score_final"])
assert H["key"].n_unique() == H.height, "header keys not unique"
HM = H.select([pl.col("key").alias("match_key"), "map_name"])

RE_raw = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE20 = RE_raw.filter(pl.col("player_count") == 20)
dups = RE20.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
conf = (RE20.group_by(["match_key", "round"]).agg(pl.col("winner_team_code").n_unique().alias("nw"))
        .filter(pl.col("nw") > 1).height)
print("=== D8-0 validation ===")
print(f"round_end rows all player_count: {RE_raw.height}; player_count==20: {RE20.height}")
print(f"duplicate (match,round) groups among player_count==20: {dups.height} "
      f"(rows {int(dups['len'].sum()) if dups.height else 0}); groups with conflicting winner: {conf}")
RE20u = RE20.unique(["match_key", "round"], keep="first")

BS = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
ev = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
both = ev.filter(pl.col("evs").list.len() > 1).height
print(f"rounds with a bomb outcome row: {ev.height}; with both defused and exploded rows: {both}")
OUT = ev.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside")
).select(["match_key", "round", "bside"])

e2 = RE20.join(OUT, on=["match_key", "round"], how="inner")
print(f"e2 reproduction (no dedupe): rounds with outcome {e2.height}, "
      f"mismatches {int((e2['bside'] != e2['winner_team_code']).sum())} (e2 log: 15436 / 499)")

U = (RE20u.join(OUT, on=["match_key", "round"], how="left")
     .join(HM, on="match_key", how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int32).alias("day")))
print(f"null map_name: {U['map_name'].null_count()}; null day: {U['day'].null_count()}")
U = U.with_columns(
    pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dg"),
    pl.col("bside").is_not_null().alias("planted"),
    (pl.col("bside") != pl.col("winner_team_code")).fill_null(False).alias("mm"),
)
J = U.filter(pl.col("planted"))
print(f"deduplicated universe with outcome: {J.height} rounds, {int(J['mm'].sum())} mismatches")
print(f"day groups in J: {J.group_by('dg').len().sort('dg').rows()}")

# ---------------------------------------------------------------- D8-1
print("\n=== D8-1 round-13 mismatch by map ===")
r13 = J.filter(pl.col("round") == 13)
print("round-13 planted rows, all days, by day group x map (n / mm / rate):")
print(tabg(r13, ["dg", "map_name"]))
R13 = r13.filter(pl.col("day") >= 12)
R13_07 = r13.filter(pl.col("day") <= 7)
print(f"\nround-13 days 12-25 by map: total {R13.height} rows, {int(R13['mm'].sum())} mismatches")
print_ci(tabg(R13, ["map_name"]), "map_name")
print(f"\nround-13 days 01-07 by map: total {R13_07.height} rows, {int(R13_07['mm'].sum())} mismatches")
print_ci(tabg(R13_07, ["map_name"]), "map_name") if R13_07.height else print("no rows")
chi2, dof, p, emin, n_lt5, ncell = chi2_het(R13)
print(f"\nchi-square map x mismatch, round 13 days 12-25: chi2={chi2:.2f} dof={dof} p={p:.3g} "
      f"min expected={emin:.2f} cells with expected<5: {n_lt5}/{ncell} (no correction, uncorrected asymptotics)")

# ---------------------------------------------------------------- D8-2
print("\n=== D8-2 round-13 map effect with day control (days 12-25) ===")
mp_names = sorted(R13["map_name"].unique().to_list())
dy_vals = sorted(R13["day"].unique().to_list())
g_map, _ = codes(R13["map_name"].to_list())
g_day, _ = codes(R13["day"].to_list())
y13 = R13["mm"].cast(pl.Float64).to_numpy()
zeros = np.zeros(len(y13), dtype=np.int64)
M, D = len(mp_names), len(dy_vals)
print(f"rows {len(y13)}; maps {M}; days {D}; B={B_PERM}; seed={SEED} (ASSUMPTION)")

o1, k1, p1 = perm_test(g_map, zeros, y13, M, 1)
o2, k2, p2 = perm_test(g_map, g_day, y13, M, D)
o3, k3, p3 = perm_test(g_day, zeros, y13, D, 1)
o4, k4, p4 = perm_test(g_day, g_map, y13, D, M)
print(f"{'test':<34}{'stat':>10}{'groups':>8}{'perm p':>10}")
print(f"{'map, unadjusted':<34}{o1:>10.2f}{k1:>8}{p1:>10.4f}")
print(f"{'map | day (day-stratified)':<34}{o2:>10.2f}{k2:>8}{p2:>10.4f}")
print(f"{'day, unadjusted':<34}{o3:>10.2f}{k3:>8}{p3:>10.4f}")
print(f"{'day | map (map-stratified)':<34}{o4:>10.2f}{k4:>8}{p4:>10.4f}")

ns_d = np.bincount(g_day, minlength=D).astype(float)
ys_d = np.bincount(g_day, weights=y13, minlength=D)
p_d = ys_d / ns_d
E_m = np.bincount(g_map, weights=p_d[g_day], minlength=M)
O_m = np.bincount(g_map, weights=y13, minlength=M)
N_m = np.bincount(g_map, minlength=M).astype(float)
print("\nday-standardised O/E per map (E uses the day-level round-13 rate across all maps):")
print(f"{'map':<13}{'n':>5}{'O':>5}{'E':>8}{'O/E':>7}")
for i, m in enumerate(mp_names):
    print(f"{m:<13}{int(N_m[i]):>5}{int(O_m[i]):>5}{E_m[i]:>8.2f}{O_m[i] / E_m[i]:>7.2f}")
print("day-level rates used for E (day: n, mm, rate):")
print("  " + "; ".join(f"{d}: {int(ns_d[i])}, {int(ys_d[i])}, {p_d[i]:.2f}" for i, d in enumerate(dy_vals)))

print("\nwithin-map early (12-16) vs late (17-25) round-13 rate:")
early = R13.filter(pl.col("day") <= 16)
late = R13.filter(pl.col("day") >= 17)
e_t = tabg(early, ["map_name"]).rename({"n": "n_early", "mm": "mm_early", "rate": "rate_early"})
l_t = tabg(late, ["map_name"]).rename({"n": "n_late", "mm": "mm_late", "rate": "rate_late"})
print(e_t.join(l_t, on="map_name", how="full", coalesce=True).sort("map_name"))

# ---------------------------------------------------------------- D8-3
print("\n=== D8-3 other rounds (round != 13) by map ===")
OTH = J.filter(pl.col("round") != 13)
OTH12 = OTH.filter(pl.col("day") >= 12)
OTH07 = OTH.filter(pl.col("day") <= 7)
print(f"other rounds, all days: {OTH.height} rows, {int(OTH['mm'].sum())} mismatches")
print(f"other rounds, days 12-25: {OTH12.height} rows, {int(OTH12['mm'].sum())} mismatches")
print(f"other rounds, days 01-07: {OTH07.height} rows, {int(OTH07['mm'].sum())} mismatches")
print("\nother-round mismatches (all days) by day group x map:")
print(OTH.filter(pl.col("mm")).group_by(["dg", "map_name"]).len().sort(["dg", "map_name"]))
print("\nother rounds days 12-25 by map:")
print_ci(tabg(OTH12, ["map_name"]), "map_name")
oth_g, _ = codes(OTH12["map_name"].to_list())
oth_d, _ = codes(OTH12["day"].to_list())
y_oth = OTH12["mm"].cast(pl.Float64).to_numpy()
mo = len(set(OTH12["map_name"].to_list()))
do = len(set(OTH12["day"].to_list()))
oo1, ok1, pp1 = perm_test(oth_g, np.zeros(len(y_oth), dtype=np.int64), y_oth, mo, 1)
oo2, ok2, pp2 = perm_test(oth_g, oth_d, y_oth, mo, do)
print(f"\nother rounds days 12-25: map unadjusted stat={oo1:.2f} perm p={pp1:.4f}; "
      f"map | day stat={oo2:.2f} perm p={pp2:.4f}")
print("\nround 25 (days 12-25) by map:")
R25 = J.filter((pl.col("round") == 25) & (pl.col("day") >= 12))
print_ci(tabg(R25, ["map_name"]), "map_name")

# ---------------------------------------------------------------- D8-4
print("\n=== D8-4 round-13 grid mm/n by day x map (days 12-25) ===")
cells = {(r["day"], r["map_name"]): (r["mm"], r["n"]) for r in
         R13.group_by(["day", "map_name"]).agg(pl.col("mm").sum().alias("mm"), pl.len().alias("n")).iter_rows(named=True)}
hdr = "day  " + "".join(f"{m[:10]:>12}" for m in mp_names)
print(hdr)
for d in dy_vals:
    row = f"{d:<5}"
    for m in mp_names:
        if (d, m) in cells:
            mm_, n_ = cells[(d, m)]
            row += f"{str(mm_) + '/' + str(n_):>12}"
        else:
            row += f"{'-':>12}"
    print(row)
all_mm_days = [d for d in dy_vals if d in (17, 25)]
for d in all_mm_days:
    maps_d = {m: cells[(d, m)] for m in mp_names if (d, m) in cells}
    print(f"day {d}: maps with round-13 rows {len(maps_d)}; "
          f"maps with mm<n: {[m for m, (a, b) in maps_d.items() if a < b] or 'none'}")

# ---------------------------------------------------------------- D8-5 (secondary, header misfit)
print("\n=== D8-5 secondary: match-level non-round-13 header misfit by map ===")
RX = RE_raw.select(["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
outs4 = OUT.select(["match_key", "round", "bside"])
R4 = RX.join(outs4, on=["match_key", "round"], how="left")
mm13 = set(R4.filter((pl.col("round") == 13) & pl.col("bside").is_not_null()
                     & (pl.col("bside") != pl.col("winner_team_code")))["match_key"].to_list())
by = {}
for r in R4.sort(["match_key", "round"]).iter_rows(named=True):
    by.setdefault(r["match_key"], []).append(r)
hdr_map = dict(zip(H["key"].to_list(), H["map_name"].to_list()))
hdr_tot = {k: (int(a), int(b)) for k, a, b in zip(H["key"].to_list(), H["t_starters_score_final"].to_list(),
                                                    H["ct_starters_score_final"].to_list())}
diff_hist = {"d01-07": {}, "d12-25": {}}
rows = []
for m, rounds in by.items():
    if m not in hdr_tot:
        continue
    tA, tB = hdr_tot[m]
    n = len(rounds)
    dg = "d01-07" if int(m.split("/")[3]) <= 7 else "d12-25"
    dd = n - (tA + tB)
    diff_hist[dg][dd] = diff_hist[dg].get(dd, 0) + 1
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    wa = sum(1 for x in rounds if int(x["winner_team_code"]) == (T if int(x["round"]) <= 12 else CT))
    misfit = (wa, n - wa) != (tA, tB)
    r13 = next((x for x in rounds if int(x["round"]) == 13), None)
    flip_fix = False
    if misfit and r13 is not None:
        nwa = wa - 1 if int(r13["winner_team_code"]) == CT else wa + 1
        flip_fix = (nwa, n - nwa) == (tA, tB)
    cls = "affected" if m in mm13 else ("r13_flip_misfit" if flip_fix else ("other_misfit" if misfit else "fits"))
    rows.append({"match_key": m, "map_name": hdr_map[m], "day": int(m.split("/")[3]), "cls": cls,
                 "dg": dg})
D5 = pl.DataFrame(rows)
print(f"complete regulation matches: {D5.height}; by class (D4 log: affected 211, r13_flip_misfit 825, "
      f"other_misfit 116, fits 94):")
print(D5.group_by("cls").len().sort("cls"))
print("complete matches by day group:", D5.group_by("dg").len().sort("dg").rows())
for dg in ["d01-07", "d12-25"]:
    h = sorted(diff_hist[dg].items(), key=lambda kv: -kv[1])[:6]
    print(f"round count minus header total, {dg} (top 6, value: matches): {h}")
D5x = D5.with_columns((pl.col("cls") == "other_misfit").cast(pl.Float64).alias("y"))
print("\nclass counts by map (complete matches):")
print(D5.group_by(["map_name", "cls"]).len().pivot(on="cls", index="map_name", values="len").fill_null(0).sort("map_name"))
print("\nother_misfit rate by map (complete matches, days 12-25):")
for r in D5x.group_by("map_name").agg(pl.len().alias("n"), pl.col("y").sum().alias("k")).sort("map_name").iter_rows(named=True):
    print(f"  {r['map_name']:<12} n={r['n']:>4} other_misfit={int(r['k']):>3} rate={r['k'] / r['n']:.3f}")
g5, _ = codes(D5x["map_name"].to_list())
s5, _ = codes(D5x["day"].to_list())
y5 = D5x["y"].to_numpy()
m5 = len(set(D5x["map_name"].to_list()))
d5 = len(set(D5x["day"].to_list()))
q1 = perm_test(g5, np.zeros(len(y5), dtype=np.int64), y5, m5, 1)
q2 = perm_test(g5, s5, y5, m5, d5)
print(f"\nother_misfit by map: unadjusted stat={q1[0]:.2f} perm p={q1[2]:.4f}; "
      f"map | day stat={q2[0]:.2f} perm p={q2[2]:.4f}")
print("\nDONE")
