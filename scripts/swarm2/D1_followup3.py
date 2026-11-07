"""D1 follow-up 3 (swarm2): explain the low-coverage / zero-flip matches.

H1 the 344 matches with 4 recorded players are a smaller format (round_end player_count != 20), not a defect.
H2 their side flip, if any, happens at a different round than 12->13.
Outputs: flip boundary signature per group; round_end player_count per group; rounds per match.
ASSUMPTIONS as in D1_flip_test_v2.py (A1-A8).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"G:\pureskill\csds-cs2\scripts\swarm2")
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
PI_raw = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
k = ["match_key", "round", "player_id_fixed"]
dup_matches = sorted(PI_raw.filter(pl.len().over(k) > 1)["match_key"].unique().to_list())
PI = PI_raw.filter(~pl.col("match_key").is_in(dup_matches))
G = pl.read_csv(OUT / "D1_coverage_table.csv", infer_schema_length=0).select(["match_key", "n_cov", "n_flip", "low_cov", "day", "bomb_mm"])
G = G.with_columns(pl.col("n_cov").cast(pl.Int64), pl.col("n_flip").cast(pl.Int64))

# per-match player_info size per round and max round
pinfo = (PI.group_by(["match_key", "round"]).agg(pl.len().alias("np"))
         .group_by("match_key").agg(pl.col("np").max().alias("max_np_any_round"),
                                    pl.col("np").median().alias("median_np"),
                                    pl.col("round").n_unique().alias("n_pi_rounds"),
                                    pl.col("round").max().alias("max_pi_round")))
# player_count from round_end (mode over rounds)
pc = (RE.group_by(["match_key", "player_count"]).len()
      .sort(["match_key", "len"], descending=[False, True])
      .group_by("match_key").agg(pl.col("player_count").first().alias("pc_mode"),
                                 pl.col("player_count").n_unique().alias("n_pc_values")))

# flip boundaries for every match: fraction covered that changed, b=1..29
parts = []
for b in range(1, 30):
    a = PI.filter(pl.col("round") == b).select(["match_key", "player_id_fixed", pl.col("team_code").alias("ta")])
    c = PI.filter(pl.col("round") == b + 1).select(["match_key", "player_id_fixed", pl.col("team_code").alias("tb")])
    g = (a.join(c, on=["match_key", "player_id_fixed"], how="inner")
          .with_columns((pl.col("ta") != pl.col("tb")).alias("flip"))
          .group_by("match_key").agg(pl.len().alias("n_both"), pl.col("flip").sum().alias("n_flip"))
          .with_columns(pl.lit(b).alias("b")))
    parts.append(g)
FL = pl.concat(parts).with_columns((pl.col("n_flip") / pl.col("n_both")).alias("ff"))
# boundaries where the match is clearly flipped (all covered changed, >=2 covered)
sig = (FL.filter((pl.col("ff") >= 0.9) & (pl.col("n_both") >= 2))
       .group_by("match_key").agg(pl.col("b").sort().alias("flip_bounds")))
sig = sig.with_columns(pl.col("flip_bounds").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("flip_bounds_str"))

X = (G.join(pinfo, on="match_key", how="left").join(pc, on="match_key", how="left")
     .join(sig.select(["match_key", "flip_bounds_str"]), on="match_key", how="left")
     .with_columns(pl.col("flip_bounds_str").fill_null(""),
                   pl.when(pl.col("low_cov")=="true").then(pl.lit("low_cov")).otherwise(pl.lit("high_cov")).alias("cov_grp")))
X = X.with_columns(pl.when(pl.col("n_cov") == 4).then(pl.lit("n_cov4")).when(pl.col("n_cov") == 0).then(pl.lit("n_cov0"))
                   .when(pl.col("n_cov") >= 6).then(pl.lit("n_cov_ge6")).otherwise(pl.lit("other")).alias("cov_class"))

hdr("H1 round_end player_count (mode) by coverage class")
print(X.group_by(["cov_class", "pc_mode"]).len().sort(["cov_class", "len"], descending=[False, True]))

hdr("H1b player_info players per round (median across rounds) by coverage class")
print(X.group_by(["cov_class", "median_np"]).len().sort(["cov_class", "len"], descending=[False, True]).head(20))

hdr("H2 flip boundary signature (boundaries 1..29 where >=90% of covered players changed) by coverage class")
print(X.group_by(["cov_class", "flip_bounds_str"]).len().sort(["cov_class", "len"], descending=[False, True]).head(40))

hdr("H2b: n_cov4 matches -- flip boundaries and pc_mode")
x4 = X.filter(pl.col("cov_class") == "n_cov4")
print(x4.group_by(["flip_bounds_str", "pc_mode"]).len().sort("len", descending=True))

hdr("H3 r13 bomb-mismatch matches: flip boundary signature (should be 12 only, or 12 and 27)")
print(X.filter(pl.col("bomb_mm") == "true").group_by("flip_bounds_str").len().sort("len", descending=True))
print("r13 mismatch matches total:", X.filter(pl.col("bomb_mm") == "true").height)
print("r13 mismatch pc_mode distribution:", X.filter(pl.col("bomb_mm") == "true")["pc_mode"].value_counts().sort("pc_mode").to_dicts())

hdr("H4 player_info rows present for n_cov0 matches: rounds present and max round")
x0 = X.filter(pl.col("cov_class") == "n_cov0")
print("n_cov0 matches:", x0.height, " with any player_info:", x0.filter(pl.col("n_pi_rounds").is_not_null()).height)
print(x0.group_by(["pc_mode"]).len().sort("len", descending=True))
print("max player_info round among n_cov0 matches:", x0["max_pi_round"].value_counts().sort("max_pi_round").to_dicts()[:20])

X.write_csv(OUT / "D1_match_signature_table.csv")
print("\nwrote D1_match_signature_table.csv")
