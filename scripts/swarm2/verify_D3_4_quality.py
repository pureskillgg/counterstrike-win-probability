"""D3-4 verifier: data-quality checks on day 12..25 matches (2024 tome). Read-only, eager load of 3 narrow channels."""
import glob, sys, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def folder(ch):
    return TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
def read(ch, cols=None):
    fs = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")
RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message"])
print("round_end rows:", RE.height)
d = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"),
                                           pl.col("tick").n_unique().alias("nt"))
print("distinct (match,round):", d.height, "| groups with n>1:", d.filter(pl.col("n") > 1).height,
      "| groups with conflicting winner codes:", d.filter(pl.col("nw") > 1).height)
print("winner_team_code values:", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("win_reason_message by winner_team_code:")
print(RE.group_by(["winner_team_code", "win_reason_message"]).len().sort(["winner_team_code", "len"], descending=[False, True]))
FE = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).filter(
    pl.col("event_type") == "round_freeze_end")
f = FE.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("t_score").n_unique().alias("nt"),
                                           pl.col("ct_score").n_unique().alias("nc"))
print("freeze_end rows:", FE.height, "| distinct (match,round):", f.height, "| groups n>1:", f.filter(pl.col("n") > 1).height,
      "| conflicting t_score in group:", f.filter(pl.col("nt") > 1).height, "| conflicting ct_score:", f.filter(pl.col("nc") > 1).height)
H = read("header", ["key", "t_starters_score_final", "ct_starters_score_final", "map_name"])
H = H.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_total"),
                   pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
H12 = H.filter(pl.col("day") >= 12)
rc = RE.group_by("match_key").agg(pl.col("round").n_unique().alias("n_rounds"), pl.col("round").max().alias("max_round"))
j = H12.join(rc, left_on="key", right_on="match_key", how="left")
print("day12-25 headers:", j.height, "| with round_end rows:", j.filter(pl.col("n_rounds").is_not_null()).height)
j = j.with_columns((pl.col("hdr_total") - pl.col("n_rounds")).alias("hdr_minus_rounds"))
print("hdr_total - n_rounds distribution:")
print(j.group_by("hdr_minus_rounds").len().sort("len", descending=True).head(15))
print("header total (t_final+ct_final) distribution, top 15:")
print(j.group_by("hdr_total").len().sort("len", descending=True).head(15))
print("header final scores: max t_starters_score_final", j["t_starters_score_final"].max(), "max ct_starters_score_final", j["ct_starters_score_final"].max())
