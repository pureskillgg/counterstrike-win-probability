"""D7-1 verifier, part C2: probe the header final-score fields against round_end winner counts (no assumptions beyond codes)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "t_starters_score_final", "ct_starters_score_final"])
               for f in files(BASE / "header")], how="vertical_relaxed")
RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code"]) for f in files(BASE / f"basic_round_end{SUF}")]).collect()
G = RE.group_by("match_key").agg(pl.len().alias("n"), (pl.col("winner_team_code") == 2).sum().alias("w2"), (pl.col("winner_team_code") == 3).sum().alias("w3"), pl.col("round").max().alias("maxr"))
X = H.join(G, left_on="key", right_on="match_key", how="left")
print("score fields: null counts:", X.select(pl.col("t_starters_score_final").null_count(), pl.col("ct_starters_score_final").null_count()).rows())
print("by build: t_score==w2 & ct_score==w3 :", X.group_by("build_num").agg(
    ((pl.col("t_starters_score_final") == pl.col("w2")) & (pl.col("ct_starters_score_final") == pl.col("w3"))).sum().alias("eq_w2w3"),
    ((pl.col("t_starters_score_final") == pl.col("w3")) & (pl.col("ct_starters_score_final") == pl.col("w2"))).sum().alias("eq_w3w2"),
    pl.len().alias("n")).rows())
print("by build: sum of scores vs n rounds, equal count:", X.group_by("build_num").agg(
    ((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")) == pl.col("n")).sum().alias("sum_eq_n"),
    ((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")) == pl.col("w2") + pl.col("w3")).sum().alias("sum_eq_w2w3"),
    pl.len().alias("n")).rows())
print("\n9842 sample (key day, scores, n, w2, w3, maxr):")
print(X.filter(pl.col("build_num") == 9842).head(8).select(["key", "t_starters_score_final", "ct_starters_score_final", "n", "w2", "w3", "maxr"]).with_columns(pl.col("key").str.slice(0, 30)))
print("\n9961 sample:")
print(X.filter(pl.col("build_num") == 9961).head(8).select(["key", "t_starters_score_final", "ct_starters_score_final", "n", "w2", "w3", "maxr"]).with_columns(pl.col("key").str.slice(0, 30)))
print("\nscore field value distribution (build 9842):", X.filter(pl.col("build_num") == 9842).select(pl.col("t_starters_score_final").value_counts().head(3)).rows() if False else "skipped")
print("9842 t_score min/max:", X.filter(pl.col("build_num") == 9842).select(pl.col("t_starters_score_final").min().alias("a"), pl.col("t_starters_score_final").max().alias("b"), pl.col("ct_starters_score_final").min().alias("c"), pl.col("ct_starters_score_final").max().alias("d")).rows())
print("9961 t_score min/max:", X.filter(pl.col("build_num") == 9961).select(pl.col("t_starters_score_final").min().alias("a"), pl.col("t_starters_score_final").max().alias("b"), pl.col("ct_starters_score_final").min().alias("c"), pl.col("ct_starters_score_final").max().alias("d")).rows())
