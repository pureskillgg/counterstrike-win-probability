"""D1 follow-up 2 (swarm2).

G1 round 27 and round 28: bomb evidence vs winner_team_code vs score side (is the score side itself reliable at 27?)
G2 round 9 disagreements (winner_team_code vs score side) in d12-25: how many matches, clustering, bomb evidence
G3 low-coverage matches (fewer than 6 covered players at 12->13): player-id stability and team-count swap signature
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
MIN_COV = 6
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_id_fixed", "player_team_code"])
PI_raw = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
RS = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
k = ["match_key", "round", "player_id_fixed"]
dup_matches = sorted(PI_raw.filter(pl.len().over(k) > 1)["match_key"].unique().to_list())
PI = PI_raw.filter(~pl.col("match_key").is_in(dup_matches))

outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
BO = (outcome.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
      .select(["match_key", "round", "bomb_side"]))

sc = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_before"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_before"),
        pl.col("t_score").sort_by("tick").last().alias("t_after"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_after")))
sc = sc.with_columns(
    pl.when((pl.col("t_after") - pl.col("t_before")) == 1).then(pl.lit(2))
      .when((pl.col("ct_after") - pl.col("ct_before")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side")).select(["match_key", "round", "score_side"])

A = (RE.filter(pl.col("player_count") == 20)
     .join(BO, on=["match_key", "round"], how="left")
     .join(sc, on=["match_key", "round"], how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
     .with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg")))
A = A.with_columns(
    (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_mm"),
    (pl.col("score_side").is_not_null() & (pl.col("score_side") != pl.col("winner_team_code"))).alias("score_mm"))

# r13 status for context
mm13 = A.filter((pl.col("round") == 13) & pl.col("bomb_mm")).select("match_key").unique().with_columns(pl.lit(True).alias("r13mm"))
A = A.join(mm13, on="match_key", how="left").with_columns(pl.col("r13mm").fill_null(False))

# ----------------------------------------------------------------------------- G1
hdr("G1 round 27 and 28 (overtime halves): winner vs bomb vs score; bomb-verified rows only")
for rr in [27, 28, 26]:
    x = A.filter(pl.col("round") == rr)
    print(f"\n-- round {rr}: rows={x.height}, rows with bomb outcome={x.filter(pl.col('bomb_side').is_not_null()).height}")
    print("   score-side disagreements with winner (all rows with score):",
          x.filter(pl.col("score_side").is_not_null()).select(pl.col("score_mm").sum()).item(),
          "of", x.filter(pl.col("score_side").is_not_null()).height)
    xb = x.filter(pl.col("bomb_side").is_not_null())
    if xb.height:
        print(xb.group_by(["dg", "bomb_side", "winner_team_code", "score_side"]).len().sort(["dg", "bomb_side", "winner_team_code", "score_side"]))
hdr("G1b round 27 non-bomb rows: winner vs score (d01-07 and d12-25)")
x27 = A.filter((pl.col("round") == 27) & pl.col("score_side").is_not_null())
print(x27.group_by(["dg", "winner_team_code", "score_side"]).len().sort(["dg", "winner_team_code", "score_side"]))

# ----------------------------------------------------------------------------- G2
hdr("G2 round 9 disagreements (winner vs score) in d12-25")
x9 = A.filter((pl.col("round") == 9) & pl.col("score_side").is_not_null())
print("round 9 rows with score side:", x9.height, " disagreements:", int(x9["score_mm"].sum()))
print(x9.group_by(["dg", "score_mm", "bomb_side"]).len().sort(["dg", "score_mm", "bomb_side"]))
d9 = x9.filter(pl.col("score_mm") & (pl.col("dg") == "d12-25"))
print("distinct d12-25 matches with a round-9 disagreement:", d9["match_key"].n_unique())
print("round-9 disagreements per match:", d9.group_by("match_key").len()["len"].value_counts().sort("len").to_dicts())
print("of those matches, with an r13 bomb mismatch:",
      d9.join(mm13, on="match_key", how="left").filter(pl.col("r13mm").fill_null(False)).select("match_key").unique().height)
print("round-9 disagreement rows with bomb evidence:", d9.filter(pl.col("bomb_side").is_not_null()).height)
# do the disagreeing matches disagree on other rounds too?
other = A.filter((pl.col("round") != 9) & pl.col("score_side").is_not_null() & pl.col("score_mm") & (pl.col("dg") == "d12-25"))
print("disagreements at other rounds (any) in those matches:",
      other.join(d9.select("match_key").unique(), on="match_key", how="inner").height)
print("\nexample match 9-disagreement, rounds 7..11 winner vs score vs bomb:")
ex = d9.select("match_key").head(2)["match_key"].to_list()
for mk in ex:
    print(mk)
    print(A.filter((pl.col("match_key") == mk) & pl.col("round").is_between(7, 11)).select(
        ["round", "winner_team_code", "score_side", "bomb_side", "player_count"]).sort("round"))

# ----------------------------------------------------------------------------- G3
hdr("G3 low-coverage matches at 12->13: id stability and team-count signature")
pcount = (PI.group_by(["match_key", "round"]).agg(pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("nT"),
                                                  (pl.col("team_code") == 3).sum().alias("nCT")))
p12 = pcount.filter(pl.col("round") == 12).select(["match_key", pl.col("np").alias("np12"), pl.col("nT").alias("nT12"), pl.col("nCT").alias("nCT12")])
p13 = pcount.filter(pl.col("round") == 13).select(["match_key", pl.col("np").alias("np13"), pl.col("nT").alias("nT13"), pl.col("nCT").alias("nCT13")])
pids = PI.select(["match_key", "round", "player_id_fixed"]).unique()
ov = (pids.filter(pl.col("round") == 12).select(["match_key", "player_id_fixed"])
      .join(pids.filter(pl.col("round") == 13).select(["match_key", "player_id_fixed"]), on=["match_key", "player_id_fixed"], how="inner")
      .group_by("match_key").len().rename({"len": "overlap_12_13"}))
b12 = (pl.concat(parts := [PI.filter(pl.col("round") == 12).select(["match_key", "player_id_fixed", pl.col("team_code").alias("ta")])])
       .join(PI.filter(pl.col("round") == 13).select(["match_key", "player_id_fixed", pl.col("team_code").alias("tb")]),
             on=["match_key", "player_id_fixed"], how="inner")
       .with_columns((pl.col("ta") != pl.col("tb")).alias("flip")).group_by("match_key")
       .agg(pl.len().alias("n_cov"), pl.col("flip").sum().alias("n_flip")))
G = (pl.DataFrame({"match_key": PI["match_key"].unique().to_list()})
     .join(p12, on="match_key", how="left").join(p13, on="match_key", how="left")
     .join(ov, on="match_key", how="left").join(b12, on="match_key", how="left")
     .with_columns(pl.col("n_cov").fill_null(0), pl.col("n_flip").fill_null(0), pl.col("overlap_12_13").fill_null(0))
     .join(A.filter(pl.col("round") == 13).select(["match_key", "bomb_side", "score_mm", "bomb_mm"]), on="match_key", how="left")
     .join(RE.filter(pl.col("round") == 13).select(["match_key"]).unique(), on="match_key", how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
     .with_columns(
         (pl.col("n_cov") < MIN_COV).alias("low_cov"),
         ((pl.col("nT12") == pl.col("nCT13")) & (pl.col("nCT12") == pl.col("nT13")) & (pl.col("nT12") != pl.col("nCT12"))).alias("count_swap"),
         ((pl.col("np12") == 10) & (pl.col("np13") == 10)).alias("ten_ten"),
         pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg")))
print("matches (dup-excluded universe):", G.height, " low coverage (n_cov<6):", G.filter(pl.col("low_cov")).height)
print("\nlow-coverage matches: distribution of n_cov, overlap, 10-player rounds")
lo = G.filter(pl.col("low_cov"))
print(lo.group_by("n_cov").len().sort("n_cov"))
print("low-coverage: overlap distribution:", lo["overlap_12_13"].value_counts().sort("overlap_12_13").to_dicts())
print("low-coverage: rounds 12 and 13 both have 10 players:", int(lo["ten_ten"].sum()), "of", lo.height)
print("low-coverage: team counts swapped between r12 and r13 (count signature):", int(lo["count_swap"].sum()), "of", lo.height)
print("low-coverage: day groups:", lo.group_by("dg").len().sort("dg").to_dicts())
print("\nhigh-coverage (n_cov>=6): ten_ten and count_swap rates:")
hi = G.filter(~pl.col("low_cov"))
print("  n", hi.height, " ten_ten", int(hi["ten_ten"].sum()), " count_swap", int(hi["count_swap"].sum()))
print("\nlow-coverage among r13 mismatch matches:", G.filter(pl.col("low_cov") & pl.col("bomb_mm")).height)
print("\nlow-coverage matches with n_cov == 4 and zero flips (spawn-zero-flip signature):",
      G.filter((pl.col("n_cov") == 4) & (pl.col("n_flip") == 0)).height)
print("\nexample low-coverage rows (first 12):")
print(lo.select(["match_key", "np12", "np13", "overlap_12_13", "n_cov", "n_flip", "nT12", "nCT12", "nT13", "nCT13", "count_swap"]).head(12))
G.write_csv(OUT / "D1_coverage_table.csv")
print("\nwrote D1_coverage_table.csv")
