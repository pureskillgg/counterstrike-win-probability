"""D9 tests: does the 2024 round side-vs-bomb mismatch associate with match metadata or events?

Input: per-round and per-match tables from scripts\\swarm2\\D9_build.py (scratchpad parquet).
Method (all within-day unless stated):
  - Stratified permutation test: exposure labels permuted WITHIN match day (2000 perms), statistic =
    sum over days of Pearson chi-square of (exposure x mismatch). Keeps each day's exposure mix fixed,
    so an exposure that merely tracks the date cannot pass.
  - Mantel-Haenszel odds ratio (binary exposures), strata = match day.
  - Crude chi-square (no date control) shown for contrast.
  - Date windows: defect window = days 12-25; clean window = days 01-07.
ASSUMPTIONS: day = match_key path day (csds/2024/02/DD); population P_main = bomb-outcome rounds with
player_count == 20 (same as the side check); P_all = all bomb-outcome rounds (any player_count).
Permutation p-values are one-sided on the stratified chi-square statistic; ~25 tests, so a Bonferroni
line of 0.05/25 = 0.002 is shown for reference.
"""
import collections
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
R = pl.read_parquet(SCR / "D9_round.parquet")
M = pl.read_parquet(SCR / "D9_match.parquet")
N_PERM = 2000

print("== 0. Date agreement: key day vs header match_date day:",
      M.filter(pl.col("day") == pl.col("date_day")).height, "of", M.height, "agree")

mcols = ["match_key", "match_type", "platform", "map_name", "build_num", "game_build", "rushb_version",
         "ppp_version", "providence", "is_dup_group", "is_lowest_key_in_group", "dup_group_size",
         "frac_pc20", "pc_min", "pc_max", "match_dc_rows", "match_dc_bot_rows", "dup_sig", "number_of_points"]
RJ = R.join(M.select(mcols), on="match_key", how="left").with_columns(
    (pl.col("player_count") == 20).alias("pc20"),
    (pl.col("dc_rows") > 0).alias("dc_in_round"),
    (pl.col("dc_bot_rows") > 0).alias("dc_bot_in_round"),
    (pl.col("match_dc_rows") > 0).alias("dc_in_match"),
    ((pl.col("n_T") == 5) & (pl.col("n_CT") == 5)).alias("five_v_five"),
    (pl.col("personal_bots") > 0).alias("bots_in_round"),
    (pl.col("n_deaths") == 0).alias("no_deaths"),
    (pl.col("n_plant_rows") == 0).alias("no_plant_row"),
    (pl.col("n_round_start") == 0).alias("no_round_start"),
    (pl.col("planter_side") == 3).alias("planter_ct"),
    (pl.col("is_dup_group") & ~pl.col("is_lowest_key_in_group")).alias("dup_non_lowest"),
    (pl.col("frac_pc20") < 1).alias("match_mixed_pc"),
)
P_all = RJ.filter(pl.col("has_outcome"))
P_main = P_all.filter(pl.col("pc20"))
print("P_all rows:", P_all.height, "mismatch:", int(P_all["mismatch"].sum()))
print("P_main rows:", P_main.height, "mismatch:", int(P_main["mismatch"].sum()))
print("P_main round 13 rows:", P_main.filter(pl.col("is_r13")).height,
      "mismatch:", int(P_main.filter(pl.col("is_r13"))["mismatch"].sum()))


def within_day(df, exp_col, seed=0):
    d = df.select(["day", exp_col, "mismatch"]).with_columns(
        pl.col(exp_col).cast(pl.String).fill_null("NULL"),
        pl.col("mismatch").cast(pl.Int64))
    day = d["day"].to_numpy()
    exp = d[exp_col].to_numpy()
    y = d["mismatch"].to_numpy()
    days = np.unique(day)
    levels = np.unique(exp)
    dcode = np.searchsorted(days, day)
    ecode = np.searchsorted(levels, exp)
    n_days, n_lev = len(days), len(levels)

    def stat(ec):
        T = np.zeros((n_days, n_lev, 2))
        np.add.at(T, (dcode, ec, y), 1)
        tot = 0.0
        for s in range(n_days):
            tab = T[s]
            r = tab.sum(1) > 0
            c = tab.sum(0) > 0
            if r.sum() < 2 or c.sum() < 2:
                continue
            sub = tab[np.ix_(r, c)]
            e = np.outer(sub.sum(1), sub.sum(0)) / sub.sum()
            tot += ((sub - e) ** 2 / e).sum()
        return tot

    obs = stat(ecode)
    rng = np.random.default_rng(seed)
    groups = [np.where(dcode == s)[0] for s in range(n_days)]
    ge = 0
    for _ in range(N_PERM):
        ep = ecode.copy()
        for g in groups:
            ep[g] = rng.permutation(ecode[g])
        if stat(ep) >= obs:
            ge += 1
    p_strat = (ge + 1) / (N_PERM + 1)
    T = np.zeros((n_lev, 2))
    np.add.at(T, (ecode, y), 1)
    chi_crude, p_crude, _, _ = stats.chi2_contingency(T, correction=False) if T.shape[0] > 1 else (0, 1, 0, 0)
    rates = {lv: (int(T[i].sum()), int(T[i, 1]), round(float(T[i, 1] / max(T[i].sum(), 1)), 4))
             for i, lv in enumerate(levels)}
    return {"p_strat": p_strat, "p_crude": float(p_crude), "crude_levels": rates,
            "n_days_with_var": int(sum(1 for s in range(n_days)
                                       if len(np.unique(ecode[groups[s]])) > 1))}


def mh_or(df, exp_col):
    d = df.drop_nulls([exp_col]).select(["day", exp_col, "mismatch"])
    num = 0.0
    den = 0.0
    used = 0
    for s in d["day"].unique().to_list():
        g = d.filter(pl.col("day") == s)
        e = g[exp_col].cast(pl.Boolean).to_numpy()
        y = g["mismatch"].cast(pl.Int64).to_numpy()
        a = int(((e) & (y == 1)).sum())
        b = int(((e) & (y == 0)).sum())
        c = int(((~e) & (y == 1)).sum())
        dd = int(((~e) & (y == 0)).sum())
        n = a + b + c + dd
        if n == 0 or (a + b) == 0 or (c + dd) == 0:
            continue
        used += 1
        num += a * dd / n
        den += b * c / n
    return (num / den if den > 0 else float("nan")), used


def day_table(df, exp_col, label):
    d = df.drop_nulls([exp_col])
    out = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in d.select(["day", exp_col, "mismatch"]).iter_rows():
        s, e, y = r
        k = str(e)
        out[(s, k)][0] += 1
        out[(s, k)][1] += int(y or 0)
    print(f"   per-day [n, mismatch] for {label}:")
    days = sorted({k[0] for k in out})
    levels = sorted({k[1] for k in out})
    for s in days:
        cells = []
        for lv in levels:
            n, m = out.get((s, lv), [0, 0])[:2]
            cells.append(f"{lv}={n}/{m}")
        print(f"     day {s}: " + "  ".join(cells))


def report(name, df, exp_col, binary=False):
    res = within_day(df, exp_col)
    line = (f"  {name:<22} p_strat={res['p_strat']:.4f}  p_crude={res['p_crude']:.4g}  "
            f"days_with_variation={res['n_days_with_var']}")
    if binary:
        orr, used = mh_or(df, exp_col)
        line += f"  MH_OR={orr:.3f} (strata used {used})"
    print(line)
    print(f"     crude [n, mismatch, rate] by level: {res['crude_levels']}")
    return res


results = {}

print("\n== D9-2. player_count (all bomb-outcome rounds, any player_count)")
print("   P_all: pc==20 is the exposure; pc20 rows vs non-20 rows, mismatch by day window")
for label, grp in [("defect window 12-25", P_all.filter(pl.col("day") >= "12")),
                   ("clean window 01-07", P_all.filter(pl.col("day") < "12"))]:
    t = grp.group_by("pc20").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
    print(f"   {label}: by pc20 -> {t.sort('pc20').rows()}")
results["D9-2 pc20 all rounds"] = report("pc20 (all rounds)", P_all, "pc20", binary=True)
results["D9-2 pc20 round13"] = report("pc20 (round 13)", P_all.filter(pl.col("is_r13")), "pc20", binary=True)
print("   non-20 round count by day (P_all):")
print("   ", P_all.filter(~pl.col("pc20")).group_by("day").len().sort("day").rows())
day_table(P_all.filter(pl.col("is_r13")), "pc20", "round 13, pc20 vs other")

print("\n== D9-1. header categorical fields (population P_main, all rounds with outcome)")
for col in ["match_type", "platform", "map_name", "build_num", "game_build", "rushb_version", "ppp_version",
            "providence", "match_mixed_pc"]:
    results[f"D9-1 {col} all"] = report(f"{col} (all)", P_main, col)
    results[f"D9-1 {col} r13"] = report(f"{col} (r13)", P_main.filter(pl.col("is_r13")), col)

print("\n   day x header field crosstab (matches, not rounds), to see whether each field tracks date:")
mm = M.select(["day", "match_type", "build_num", "game_build", "rushb_version", "ppp_version", "providence"])
for col in ["build_num", "game_build", "rushb_version", "ppp_version", "providence", "match_type"]:
    cnt = collections.defaultdict(int)
    for s, v in mm.select(["day", col]).iter_rows():
        cnt[(s, str(v))] += 1
    levels = sorted({k[1] for k in cnt})
    days = sorted({k[0] for k in cnt})
    print(f"   {col}: levels={levels}")
    for s in days:
        print(f"     day {s}: " + "  ".join(f"{lv}={cnt.get((s, lv), 0)}" for lv in levels))

print("\n== D9-3. duplicate groups (bakeoff signature)")
print("   matches in dup groups:", M.filter(pl.col("is_dup_group")).height,
      "| non-lowest copies:", M.filter(pl.col("is_dup_group") & ~pl.col("is_lowest_key_in_group")).height,
      "| dup groups:", M.filter(pl.col("dup_group_size") > 1).select("dup_sig").n_unique())
results["D9-3 is_dup_group all"] = report("is_dup_group (all)", P_main, "is_dup_group", binary=True)
results["D9-3 is_dup_group r13"] = report("is_dup_group (r13)", P_main.filter(pl.col("is_r13")),
                                          "is_dup_group", binary=True)
results["D9-3 dup_non_lowest all"] = report("dup_non_lowest (all)", P_main, "dup_non_lowest", binary=True)
results["D9-3 dup_non_lowest r13"] = report("dup_non_lowest (r13)", P_main.filter(pl.col("is_r13")),
                                            "dup_non_lowest", binary=True)
dd = P_main.filter(pl.col("is_r13") & pl.col("is_dup_group"))
grp = dd.group_by("dup_sig").agg(pl.len().alias("copies"), pl.col("mismatch").n_unique().alias("n_distinct_mm"),
                                 pl.col("mismatch").sum().alias("mm_copies"))
print(f"   r13 dup groups with >=2 copies carrying r13 outcome: {grp.height}; "
      f"groups where copies disagree on mismatch: {grp.filter(pl.col('n_distinct_mm') > 1).height}")
kept_r13 = P_main.filter(pl.col("is_r13") & ~pl.col("dup_non_lowest"))
print(f"   r13 mismatches after bakeoff dedupe (drop non-lowest copies): "
      f"{int(kept_r13['mismatch'].sum())} of {kept_r13.height}")
kept_all = P_main.filter(~pl.col("dup_non_lowest"))
print(f"   all-round mismatches after dedupe: {int(kept_all['mismatch'].sum())} of {kept_all.height}")

print("\n== D9-4. disconnect and bot events (player_disconnect has rows; bot_takeover has 0 rows, 0 columns)")
results["D9-4 dc_in_round all"] = report("dc_in_round (all)", P_main, "dc_in_round", binary=True)
results["D9-4 dc_in_round r13"] = report("dc_in_round (r13)", P_main.filter(pl.col("is_r13")),
                                         "dc_in_round", binary=True)
results["D9-4 dc_bot_in_round all"] = report("dc_bot_in_round (all)", P_main, "dc_bot_in_round", binary=True)
results["D9-4 dc_in_match all"] = report("dc_in_match (all)", P_main, "dc_in_match", binary=True)
results["D9-4 dc_in_match r13"] = report("dc_in_match (r13)", P_main.filter(pl.col("is_r13")),
                                         "dc_in_match", binary=True)
results["D9-4 bots_in_round all"] = report("bots_in_round (all)", P_main, "bots_in_round", binary=True)
results["D9-4 bots_in_round r13"] = report("bots_in_round (r13)", P_main.filter(pl.col("is_r13")),
                                           "bots_in_round", binary=True)

print("\n== D9-5. round-level event sanity flags")
for flag in ["five_v_five", "no_deaths", "no_plant_row", "no_round_start", "planter_ct"]:
    results[f"D9-5 {flag} all"] = report(f"{flag} (all)", P_main, flag, binary=True)
    results[f"D9-5 {flag} r13"] = report(f"{flag} (r13)", P_main.filter(pl.col("is_r13")), flag, binary=True)

print("\n== D9-6. win_reason message vs bomb outcome vs winner (P_main, mismatch rounds vs others)")
bo = P_main.with_columns(
    pl.when(pl.col("defused")).then(pl.lit("defused")).otherwise(pl.lit("exploded")).alias("bomb_outcome"))
c = collections.Counter()
for r in bo.select(["is_r13", "mismatch", "win_reason_message", "bomb_outcome", "winner_team_code"]).iter_rows():
    is13, mm_, msg, bomb, win = r
    c[("r13" if is13 else "other", "MISMATCH" if mm_ else "match", str(msg), bomb, str(win))] += 1
for k, v in sorted(c.items(), key=lambda kv: (-kv[1]))[:30]:
    print("  ", k, v)
