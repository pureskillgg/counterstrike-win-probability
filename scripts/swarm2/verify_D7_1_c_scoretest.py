"""D7-1 verifier, part C: independent score test (no bomb events, no ledger).
Header final scores (t_starters_score_final, ct_starters_score_final) are compared with round_end winners
under an assumed side schedule. ASSUMPTION: winner_team_code 2=T, 3=CT (side codes), as the bomb check implies.
ASSUMPTION (schedule A): MR12, starters on T for rounds 1-12 and on CT for rounds 13-24; OT (>24) excluded.
Residual = header score - predicted. Residual 0 means the round winners agree with the header final score.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
H = pl.concat([pl.read_parquet(f, columns=["key", "mdt" if False else "match_date", "build_num", "t_starters_score_final",
                                           "ct_starters_score_final", "map_name"]) for f in files(BASE / "header")], how="vertical_relaxed")
H = H.with_columns(pl.col("match_date").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S").alias("mdt"),
                   pl.col("key").str.extract(r"^csds/(\d{4}/\d\d/\d\d)/", 1).alias("key_day"))
RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"]) for f in files(BASE / f"basic_round_end{SUF}")]).collect()
print("round_end rows:", RE.height, "| matches with rounds:", RE["match_key"].n_unique(), "| header:", H.height)
M = RE.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("round").max().alias("maxr"),
                                  pl.col("round").n_unique().alias("n_unique_rounds"),
                                  (pl.col("winner_team_code") == 2).sum().alias("n_T_wins_all"),
                                  pl.col("player_count").min().alias("pc_min"))
print("n_rounds == n_unique_rounds for all matches:", (M["n_rounds"] == M["n_unique_rounds"]).all())
print("maxr distribution (top):", M.group_by("maxr").len().sort("len", descending=True).head(12).rows())
D = H.join(M, left_on="key", right_on="match_key", how="left")
print("header matches without rounds:", D.filter(pl.col("n_rounds").is_null()).height)
D = D.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("final_sum"))
print("final_sum == n_rounds (all):", (D["final_sum"] == D["n_rounds"]).sum(), "of", D.height)
print("final_sum == n_rounds by build:", D.group_by(["build_num"]).agg(
    (pl.col("final_sum") == pl.col("n_rounds")).sum().alias("eq"), pl.len().alias("n")).rows())

# schedule A on regulation-only matches (maxr <= 24, final_sum == n_rounds)
RR = RE.join(H.select(["key", "build_num", "mdt", "key_day"]).rename({"key": "match_key"}), on="match_key", how="left")
W = RR.with_columns(
    pl.when(pl.col("round") <= 12).then(pl.col("winner_team_code") == 2).otherwise(pl.col("winner_team_code") == 3).alias("starter_won"))
Ws = W.group_by("match_key").agg(pl.col("starter_won").sum().alias("pred_t"), pl.col("round").max().alias("maxr"),
                                  pl.col("build_num").first(), pl.col("mdt").first())
T = D.select(["key", "t_starters_score_final", "ct_starters_score_final", "build_num", "mdt", "n_rounds", "maxr", "final_sum", "key_day"])
C = T.join(Ws.rename({"match_key": "key"}).drop(["build_num", "mdt", "maxr"]), on="key", how="left")
C = C.with_columns((pl.col("t_starters_score_final") - pl.col("pred_t")).alias("resid_t"))
reg = C.filter((pl.col("maxr") <= 24) & (pl.col("final_sum") == pl.col("n_rounds")))
print("\nC1 regulation-only (maxr<=24, final_sum==n_rounds) matches:", reg.height)
print("C1 residual (t header - schedule A prediction) distribution by build:")
print(reg.group_by(["build_num", "resid_t"]).len().sort(["build_num", "resid_t"]).rows())
print("C1 residual by UTC match day (build, resid):")
print(reg.with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("md")).group_by(["md", "resid_t"]).len().sort(["md", "resid_t"]).rows()[:80])
