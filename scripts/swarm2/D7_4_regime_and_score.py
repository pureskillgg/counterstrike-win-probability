"""D7-4: (a) exact onset of the tick-lag regime (lag = bomb-outcome tick - round_end tick) per match;
(b) which label source reproduces the header final scores for 24-round (regulation-only) matches.
ASSUMPTION: player_count == 20. ASSUMPTION: starter-T team plays T in rounds 1-12 and CT in 13-24 (one halftime swap);
ASSUMPTION: header t_starters_score_final = rounds won by the team that started on T.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
R = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_all.parquet"))
H = pl.concat([pl.read_parquet(f, columns=["key", "mdt", "map_name", "t_starters_score_final", "ct_starters_score_final"])
               for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))]) if False else None
HDR = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt")).rename({"key": "match_key"}).select(["match_key", "mdt", "t_starters_score_final", "ct_starters_score_final"])
O = R.filter(pl.col("bomb_side").is_not_null() & (pl.col("player_count") == 20)).with_columns(
    (pl.col("out_tick") - pl.col("tick")).alias("lag"))

print("== (a) per-match lag regime, matches with >=1 outcome round ==")
M = (O.group_by("match_key").agg(
        pl.col("mdt").first(),
        pl.len().alias("n_out"),
        (pl.col("lag") == -429).sum().alias("n_m429"),
        (pl.col("lag") == 0).sum().alias("n_zero"),
        pl.col("mm").sum().alias("n_mm"),
     ).with_columns(
        pl.when(pl.col("n_m429") > 0).then(pl.lit("neg429")).when(pl.col("n_zero") == pl.col("n_out")).then(pl.lit("zero")).otherwise(pl.lit("other")).alias("regime")
     ).sort("mdt"))
print("regime counts:", M.group_by("regime").len().rows())
first_neg = M.filter(pl.col("regime") == "neg429").head(1).select(["mdt", "n_out", "n_m429"]).rows()
last_zero_before = M.filter((pl.col("regime") == "zero") & (pl.col("mdt") < first_neg[0][0])).tail(1).select(["mdt"]).rows()
print("first match with a -429 lag:", first_neg, "| last all-zero match before it:", last_zero_before)
print("all-zero matches after first -429 match:", M.filter((pl.col("regime") == "zero") & (pl.col("mdt") > first_neg[0][0])).height,
      "| all-zero matches before first -429:", M.filter((pl.col("regime") == "zero") & (pl.col("mdt") < first_neg[0][0])).height)
print("other-regime matches (mixed, no -429) by period:",
      M.filter(pl.col("regime") == "other").with_columns(pl.when(pl.col("mdt") < pl.datetime(2024,2,7)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("p")).group_by("p").len().rows())
print("first mismatch match time (any round):", M.filter(pl.col("n_mm") > 0).head(1).select("mdt").rows())
print("matches with -429 lag but time < 2024-02-07 00:00:", M.filter((pl.col("n_m429") > 0) & (pl.col("mdt") < pl.datetime(2024,2,7))).height)

print("\n== (b) header-score test: matches with max round 24 (regulation only, one halftime swap) ==")
mr = R.group_by("match_key").agg(pl.col("round").max().alias("maxr"))
reg = mr.filter(pl.col("maxr") == 24).select("match_key")
Rr = (R.join(reg, on="match_key", how="inner")
        .filter(pl.col("player_count") == 20)
        .with_columns(
           pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("bomb_side")).otherwise(pl.col("winner_team_code")).alias("bomb_or_win"),
        ))
print("regulation-only matches:", Rr["match_key"].n_unique(), "rounds:", Rr.height,
      "| rows with 24 rounds per match:", Rr.group_by("match_key").len().filter(pl.col("len") == 24).height)
def starter_t(df, col):
    # starter-T team wins: T rounds 1-12 (side code 2) + CT rounds 13-24 (side code 3)
    return df.with_columns(
        pl.when(((pl.col("round") <= 12) & (pl.col(col) == 2)) | ((pl.col("round") >= 13) & (pl.col(col) == 3))).then(1).otherwise(0).alias("w_st")
    ).group_by("match_key").agg(pl.col("w_st").sum().alias(f"st_{col}"))
ws = starter_t(Rr, "winner_team_code")
bs = starter_t(Rr, "bomb_or_win")
T = (ws.join(bs, on="match_key").join(HDR, on="match_key", how="inner")
       .with_columns(
           (pl.col("st_winner_team_code") == pl.col("t_starters_score_final")).alias("win_ok"),
           (pl.col("st_bomb_or_win") == pl.col("t_starters_score_final")).alias("bomb_ok"),
       ))
mm_match = (Rr.filter(pl.col("mm")).select("match_key").unique().with_columns(pl.lit(True).alias("has_mm")))
T = (T.join(mm_match, on="match_key", how="left").with_columns(pl.col("has_mm").fill_null(False))
       .with_columns(pl.when(pl.col("mdt") < pl.datetime(2024,2,7)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period")))
print("matches with header score:", T.height)
print(T.group_by(["period", "has_mm"]).agg(pl.len().alias("n"), pl.col("win_ok").sum().alias("winner_code_matches_header"),
        pl.col("bomb_ok").sum().alias("bomb_derived_matches_header")).sort(["period", "has_mm"]))
print("\nmismatch-containing matches where neither source matches header:",
      T.filter(pl.col("has_mm") & ~pl.col("win_ok") & ~pl.col("bomb_ok")).height)
print("sample of mismatch-containing matches (first 8):")
print(T.filter(pl.col("has_mm")).select(["mdt", "match_key", "t_starters_score_final", "st_winner_team_code", "st_bomb_or_win", "win_ok", "bomb_ok"]).sort("mdt").head(8).with_columns(pl.col("match_key").str.slice(0,36)))
