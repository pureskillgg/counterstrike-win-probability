"""verify_D9_2_a: independent recompute of the D9-2 key numbers (2024-02 tome, read-only).

Reads the raw tome channels directly (no scratchpad parquet from D9_build.py).
Columns are projected and scans are lazy. Memory target < 3 GB.

Definitions (ASSUMPTIONS, marked):
  - bomb outcome: bomb_defused -> side 3 (CT), bomb_exploded -> side 2 (T); same rule as the claim.
  - mismatch: outcome present and outcome side != round_end winner_team_code.
  - population: round 13, round_end player_count == 20 (pc20), rounds with a bomb outcome.
  - folder_day: the csds/2024/02/DD/ path day of match_key (the claim's "day").
  - cal_date: the UTC calendar date from header match_date (first 10 characters).
  - switch time: derived from the data (earliest 9961 match_date), not assumed.
  - permutation tests: build label permuted within stratum (2000 perms); statistic = number of
    build-9961 mismatch rows (Mantel-type conditional test). Floor of p is 1/2001 = 0.0005.
  - Haldane pseudo-count 0.5 is shown only as a sensitivity check; zero cells dominate.
"""
import collections
import glob
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
N_PERM = 2000


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    miss = [c for c in cols if c not in have]
    print(f"[load] {name}: files={len(files_of(fd))} cols={[c for c in cols if c in have]} missing={miss}")
    return lf.select([c for c in cols if c in have]).collect()


def strat_perm(df, stratum, label="is9961", y="mm", seed=1):
    s = df[stratum].cast(pl.String).to_numpy()
    L = df[label].cast(pl.Int64).to_numpy()
    yy = df[y].cast(pl.Int64).to_numpy()
    idx = [np.where(s == k)[0] for k in np.unique(s)]
    obs = int(((L == 1) & (yy == 1)).sum())
    rng = np.random.default_rng(seed)
    Lp = L.copy()
    ge = 0
    for _ in range(N_PERM):
        for ix in idx:
            Lp[ix] = rng.permutation(L[ix])
        if int(((Lp == 1) & (yy == 1)).sum()) >= obs:
            ge += 1
    return obs, (ge + 1) / (N_PERM + 1), len(idx)


def fisher_two_sided(a, b, c, d):
    """Exact two-sided Fisher for [[a,b],[c,d]] (rows: build 9961 / 9842, cols: mm / ok)."""
    n1, n2, m, N = a + b, c + d, a + c, a + b + c + d

    def pmf(x):
        return math.comb(m, x) * math.comb(N - m, n1 - x) / math.comb(N, n1)

    p_obs = pmf(a)
    lo, hi = max(0, n1 - (N - m)), min(n1, m)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-7)))


def mh_or(tables, k=0.0):
    num = den = 0.0
    for a, b, c, d in tables:
        a2, b2, c2, d2 = a + k, b + k, c + k, d + k
        n = a2 + b2 + c2 + d2
        if n == 0:
            continue
        num += a2 * d2 / n
        den += b2 * c2 / n
    return (num / den) if den > 0 else float("inf")


# ------------------------------------------------------------------ load
print("== 0. load")
H = scan("header", ["key", "match_date", "build_num", "game_directory", "ppp_version",
                    "rushb_version", "providence", "map_name"])
print("header rows", H.height, "unique key", H["key"].n_unique())
H = H.with_columns(
    pl.col("key").alias("match_key"),
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("folder_day"),
    pl.col("match_date").str.slice(0, 10).alias("cal_date"),
    pl.col("game_directory").str.extract(r"csgo_v(\d+)", 1).alias("game_build"),
)
iso_ok = H.filter(pl.col("match_date").str.contains(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")).height
print("match_date ISO-Z format ok:", iso_ok, "of", H.height)

RE_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message",
                            "player_count"])
dup_groups = RE_raw.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
print("round_end rows", RE_raw.height, "| (match,round) groups with >1 row:", dup_groups)
RE = RE_raw.group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("re_tick"),
    pl.col("winner_team_code").first().alias("winner"),
    pl.col("winner_team_code").n_unique().alias("n_winner_vals"),
    pl.col("player_count").first().alias("pc"),
    pl.col("win_reason_message").first().alias("win_msg"),
)

BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
print("bomb_state event types:", BS.group_by("event_type").len().sort("event_type").rows())
OUT = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
outc = OUT.sort("tick").group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_defused").any().alias("defused"),
    (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
    pl.len().alias("n_out"),
    pl.col("event_type").first().alias("first_out"),
    pl.col("tick").min().alias("out_tick"),
)
print("rounds with BOTH defused and exploded:",
      outc.filter(pl.col("defused") & pl.col("exploded")).height)
plant = BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(
    pl.len().alias("n_plant"))

RS = scan("round_start", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("rs_tick"))

# ------------------------------------------------------------------ per-round table
R = (RE.join(outc, on=["match_key", "round"], how="left")
       .join(plant, on=["match_key", "round"], how="left")
       .join(RS, on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "build_num", "game_build", "ppp_version", "rushb_version",
                       "folder_day", "cal_date", "map_name", "providence"]),
             on="match_key", how="left"))
R = R.with_columns(
    pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False),
    pl.col("n_out").fill_null(0), pl.col("n_plant").fill_null(0),
).with_columns(
    (pl.col("defused") | pl.col("exploded")).alias("has_out"),
    pl.when(pl.col("defused")).then(pl.lit(3, dtype=pl.Int64))
      .when(pl.col("exploded")).then(pl.lit(2, dtype=pl.Int64))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"),
    (pl.col("pc") == 20).alias("pc20"),
    (pl.col("round") == 13).alias("is_r13"),
    (pl.col("build_num") == 9961).alias("is9961"),
)
R = R.with_columns(
    pl.when(pl.col("has_out")).then((pl.col("bomb_side") != pl.col("winner")).cast(pl.Int64))
      .otherwise(None).alias("mm"),
)
print("rounds total", R.height, "| with outcome", int(R["has_out"].sum()),
      "| winner values per (match,round) >1:", R.filter(pl.col("n_winner_vals") > 1).height)

P = R.filter(pl.col("has_out") & pl.col("pc20"))
print("pc20 bomb-outcome rounds:", P.height, "mismatch:", int(P["mm"].sum()))

R13 = P.filter(pl.col("is_r13"))
print("\n== 1. round 13, pc20, bomb outcome: build x mismatch (claim: 9842 344/0, 9961 441/434)")
t = R13.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("build_num")
print(t)
print("non-pc20 round 13 with outcome, by build:",
      R.filter(pl.col("has_out") & pl.col("is_r13") & ~pl.col("pc20")).group_by("build_num").agg(
          pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("build_num").rows())

print("\n== 2. claim's folder-day strata (key path day): build x day, round 13 pc20")
print(R13.group_by(["folder_day", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
          .sort(["folder_day", "build_num"]))

print("\n== 3. calendar match_date (UTC) strata: build x cal_date, round 13 pc20")
print(R13.group_by(["cal_date", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
          .sort(["cal_date", "build_num"]))

print("\n== 4. folder day vs calendar date agreement (all 4435 matches)")
agree = H.filter(pl.col("folder_day") == pl.col("cal_date").str.slice(8, 2)).height
print("folder day == match_date day:", agree, "of", H.height, f"({100*agree/H.height:.1f}%)")
print("folder day == match_date day, 9961 only:",
      H.filter((pl.col("build_num") == 9961) & (pl.col("folder_day") == pl.col("cal_date").str.slice(8, 2))).height,
      "of", H.filter(pl.col("build_num") == 9961).height)

print("\n== 5. build switch time (from data)")
sw = H.group_by("build_num").agg(pl.col("match_date").min().alias("min_match_date"),
                                 pl.col("match_date").max().alias("max_match_date"),
                                 pl.len().alias("n_matches")).sort("build_num")
print(sw)
T9961 = H.filter(pl.col("build_num") == 9961)["match_date"].min()
print("earliest 9961 match_date:", T9961)
R13x = R13.join(H.select(["match_key", "match_date"]), on="match_key", how="left")
print("round 13 pc20 by build, match_date before earliest-9961 vs at/after:")
print(R13x.with_columns((pl.col("match_date") >= T9961).alias("after_switch"))
          .group_by(["build_num", "after_switch"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
          .sort(["build_num", "after_switch"]))
print("9842 round-13 pc20 rounds dated at/after first 9961 match (calendar side of the switch):",
      R13x.filter((pl.col("build_num") == 9842) & (pl.col("match_date") >= T9961)).height)

print("\n== 6. batch / pipeline nesting: ppp_version x build (all 4435 matches) and round 13 pc20")
print(H.group_by(["ppp_version", "build_num"]).len().sort(["ppp_version", "build_num"]))
print(R13.group_by(["ppp_version", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
          .sort(["ppp_version", "build_num"]))
print("ppp 6.8.0 match count:", H.filter(pl.col("ppp_version") == "6.8.0").height,
      "| folders 12-25 match count:", H.filter(pl.col("folder_day") >= "12").height,
      "| identical sets:", H.filter(pl.col("ppp_version") == "6.8.0").height == H.filter(pl.col("folder_day") >= "12").height)
print("\n== 7. game_build (directory version) x build_num, round 13 pc20")
print(R13.group_by(["game_build", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
          .sort(["build_num", "game_build"]))
print("game_build x build_num over all matches:")
print(H.group_by(["build_num", "game_build"]).len().sort(["build_num", "game_build"]))
print("rushb_version x build_num over all matches:",
      H.group_by(["build_num", "rushb_version"]).len().sort(["build_num", "rushb_version"]).rows())

print("\n== 8. stratified permutation (build label permuted within stratum), round 13 pc20")
obs, p, k = strat_perm(R13, "folder_day")
print(f"  folder_day strata={k}: obs 9961-mismatches={obs}  p={p:.4f} (claim p=0.0005)")
obs2, p2, k2 = strat_perm(R13, "cal_date")
print(f"  cal_date strata={k2}: obs 9961-mismatches={obs2}  p={p2:.4f}")
both_cal = (R13.group_by(["cal_date", "build_num"]).len()
              .group_by("cal_date").agg(pl.len().alias("nb")).filter(pl.col("nb") == 2))
print("  calendar dates with BOTH builds present in round 13 pc20:", both_cal.height,
      both_cal["cal_date"].sort().to_list())
both_fold = (R13.group_by(["folder_day", "build_num"]).len()
               .group_by("folder_day").agg(pl.len().alias("nb")).filter(pl.col("nb") == 2))
print("  folder days with BOTH builds present in round 13 pc20:", both_fold.height,
      both_fold["folder_day"].sort().to_list())

print("\n== 9. within-stratum exact comparison (Fisher, 9961 vs 9842), calendar and folder strata")


def strata_tables(df, col):
    out = []
    for key in sorted(df[col].unique().to_list()):
        g = df.filter(pl.col(col) == key)
        a = g.filter(pl.col("is9961") & (pl.col("mm") == 1)).height
        b = g.filter(pl.col("is9961") & (pl.col("mm") == 0)).height
        c = g.filter(~pl.col("is9961") & (pl.col("mm") == 1)).height
        d = g.filter(~pl.col("is9961") & (pl.col("mm") == 0)).height
        if (a + b) > 0 and (c + d) > 0:
            out.append((key, a, b, c, d))
    return out


for col in ["folder_day", "cal_date"]:
    tabs = strata_tables(R13, col)
    print(f"  stratum={col}: strata with both builds = {len(tabs)}")
    for key, a, b, c, d in tabs:
        pv = fisher_two_sided(a, b, c, d)
        print(f"    {key}: 9961 mm/n={a}/{a+b}  9842 mm/n={c}/{c+d}  fisher_p={pv:.3g}")
    tb = [(a, b, c, d) for _, a, b, c, d in tabs]
    print(f"    MH OR (no correction) = {mh_or(tb, 0.0)}")
    for kk in [0.5, 0.25, 1.0]:
        print(f"    MH OR (pseudo-count {kk}) = {mh_or(tb, kk):.4g}")

print("\n== 10. folder day 12 detail (claim: 9842 0/89, 9961 94/96; calendar view of same day rows)")
d12 = R13.filter(pl.col("folder_day") == "12")
print(d12.group_by(["build_num", "cal_date"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
         .sort(["build_num", "cal_date"]))

print("\n== 11. duplicate-copy dedupe (round_end fingerprint per match), round 13 pc20")
parts = collections.defaultdict(list)
for mk, rd, tk, w in RE_raw.select(["match_key", "round", "tick", "winner_team_code"]).iter_rows():
    parts[mk].append(f"{rd}:{tk}:{w}")
fp = {mk: hashlib.md5("|".join(sorted(v)).encode("utf-8")).hexdigest() for mk, v in parts.items()}
R13f = R13.with_columns(pl.col("match_key").replace_strict(fp, default=None, return_dtype=pl.String).alias("fp"))
print("  r13 pc20 rows:", R13f.height, "| distinct fingerprints:", R13f["fp"].n_unique())
dd = R13f.sort("match_key").group_by("fp").agg(pl.col("is9961").first().alias("is9961"),
                                               pl.col("mm").first().alias("mm"),
                                               pl.len().alias("copies"))
print("  after dedupe (one row per fingerprint):")
print(dd.group_by("is9961").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("is9961"))
print("  copies by build:", R13f.group_by("is9961").agg(
    (pl.len() - pl.col("fp").n_unique()).alias("extra_copies")).sort("is9961").rows())

print("\n== 12. other round indices, pc20 bomb-outcome rounds, build x round index (mismatch counts)")
oth = P.filter(~pl.col("is_r13")).group_by(["build_num", "round"]).agg(
    pl.len().alias("n"), pl.col("mm").sum().alias("mm")).filter(pl.col("mm") > 0).sort(["build_num", "round"])
print("  rounds with >=1 mismatch (non-13):")
print(oth)
print("  9842 non-13 mismatches total:",
      int(P.filter(~pl.col("is_r13") & (pl.col("build_num") == 9842))["mm"].sum()),
      "| 9961 non-13 rows:", P.filter(~pl.col("is_r13") & pl.col("is9961")).height,
      "mismatches:", int(P.filter(~pl.col("is_r13") & pl.col("is9961"))["mm"].sum()))
for rr in [9, 25, 28]:
    g = P.filter(pl.col("round") == rr).group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
    print(f"  round {rr}:", g.sort("build_num").rows())

print("\nDONE part a")
