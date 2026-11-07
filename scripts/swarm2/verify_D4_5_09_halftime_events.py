"""verify D4-5, step 9: halftime position from round_state announcement events (no swap convention used).

(a) Event list for two pattern-'9' matches and one pattern-'13' match, rounds 6..14: event_type, round, tick, scores.
(b) For every 2024 match: round label(s) of round_announce_last_round_half, round_announce_final,
    round_announce_match_start. Distribution of the halftime label versus the convention (12).
(c) Among complete days 12-25 matches, halftime label vs disputed pattern.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(14)
base = TOMES["2024"]
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key", "t_starters_score_final": "hT", "ct_starters_score_final": "hC"})
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")

# (a) sample matches: the two pattern-'9' examples and one pattern-'13' example from the residual step
ex = ["01afc05", "413a3d5", "ca2295e"]
keys = rs.select("match_key").unique().filter(pl.col("match_key").str.contains("|".join(ex)))["match_key"].to_list()
for k in keys[:3]:
    print("\n====", k[-44:-6])
    ev = rs.filter((pl.col("match_key") == k) & pl.col("round").is_between(6, 14) & (
        pl.col("event_type").str.contains("freeze_end|announce|round_end|officially|last_round|match_start")))
    ev = ev.select(["round", "tick", "event_type", "t_score", "ct_score"]).sort(["round", "tick"])
    print(ev)
    print("round_end winners 6..14:", re_.filter((pl.col("match_key") == k) & pl.col("round").is_between(6, 14))
          .sort("round").select(["round", "winner_team_code"]).rows())

# (b) halftime announcement label per match
half = rs.filter(pl.col("event_type") == "round_announce_last_round_half").group_by("match_key").agg(
    pl.col("round").min().alias("half_round_min"), pl.col("round").max().alias("half_round_max"),
    pl.len().alias("n_half_events"))
final = rs.filter(pl.col("event_type") == "round_announce_final").group_by("match_key").agg(
    pl.col("round").min().alias("final_round_min"), pl.col("round").max().alias("final_round_max"))
allm = rs.select("match_key").unique()
Mh = allm.join(half, on="match_key", how="left").join(final, on="match_key", how="left").join(H, on="match_key")
Mh = Mh.with_columns(day_expr().alias("day"))
print("\n(b) halftime announcement label distribution (all 2024 matches)")
print(Mh.group_by("half_round_min").len().sort("len", descending=True).head(12))
print("matches with no half announcement:", int(Mh["half_round_min"].is_null().sum()))
print("n_half_events distribution:")
print(Mh.group_by("n_half_events").len().sort("n_half_events"))

# (c) halftime label vs disputed pattern (complete days 12-25 matches)
re_n = re_.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label"))
Mc = Mh.join(re_n, on="match_key").with_columns((pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"))
print("\n(c) complete matches: halftime label distribution by day group")
Mc = Mc.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).when(pl.col("day").is_between(12, 25))
                     .then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("grp"))
print(Mc.group_by(["grp", "complete"]).agg(pl.len().alias("n")).sort(["grp", "complete"]))
print(Mc.filter(pl.col("complete")).group_by(["grp", "half_round_min"]).len().sort(["grp", "len"],
                                                                                   descending=[False, True]).head(20))

# max-wins check: first-to-9 finishes (max score 9, total <= 15) indicate an MR8-style format
Mc = Mc.with_columns(pl.max_horizontal("hT", "hC").alias("max_score"))
print("\nmax_score distribution among complete d12-25 matches (9 with total<=15 is an MR8-style finish):")
print(Mc.filter(pl.col("complete") & (pl.col("grp") == "d12-25")).group_by(["max_score"]).len().sort("max_score"))
