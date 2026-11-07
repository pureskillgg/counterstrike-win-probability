"""verify D4-5, step 4: debug the header final-score test on a few matches (clean days 01-07 and one disputed).

Prints, per match: header finals, round_end winners by round, the freeze_end score fields, and the
round_state final cumulative fields, so the team-to-side convention can be read directly.
Run via the project interpreter. Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)
base = TOMES["2024"]
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final", "match_date"]).rename(
    {"key": "match_key"})

fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
last = rs.sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_last"), pl.col("ct_score").last().alias("c_last"),
    pl.col("round").max().alias("max_round_state"))

# choose one clean match and two disputed matches (dis = a round-13 row where fe score side != round_end)
keys_clean = (re_.filter(pl.col("round") == 5).select("match_key").head(2))["match_key"].to_list()
keys_all = re_.select("match_key").unique()["match_key"].to_list()
probe_keys = keys_clean + keys_all[:0]
# disputed: find matches whose round-13 fe score delta disagrees with round_end
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "c1"})
D = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t1") - pl.col("t_score")).alias("dT"), (pl.col("c1") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"))
J = D.join(re_.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner")
disp = J.filter((pl.col("round") == 13) & (pl.col("ss") != 0) & (pl.col("ss") != pl.col("winner_team_code")))
probe_keys += disp.select("match_key").head(2)["match_key"].to_list()

for k in probe_keys:
    h = H.filter(pl.col("match_key") == k).row(0, named=True)
    print("\n==================================================")
    print("match", k[-40:], "date", h["match_date"], "header T-start final", h["t_starters_score_final"],
          "CT-start final", h["ct_starters_score_final"])
    w = re_.filter(pl.col("match_key") == k).sort("round")
    f = fe.filter(pl.col("match_key") == k).sort("round")
    m = w.join(f, on=["match_key", "round"], how="full", coalesce=True).sort("round")
    print(m.select(["round", "winner_team_code", "win_reason_code", "t_score", "ct_score"]))
    print("round_state last t/c:", last.filter(pl.col("match_key") == k).row(0, named=True))
    print("count round_end rows:", w.height, "T wins:", int((w["winner_team_code"] == T).sum()),
          "CT wins:", int((w["winner_team_code"] == CT).sum()))
