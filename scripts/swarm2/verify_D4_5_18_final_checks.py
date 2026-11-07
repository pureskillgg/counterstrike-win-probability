"""verify D4-5, step 18: last two numbers from the claim, plus the round-1 labelling x build x dispute table.

(a) Round-13 bomb rows, all 2024 days, valid unambiguous (no swap after 13): n, score side == bomb side,
    score side == round_end winner. Claim: 799 rows, 100% vs bomb, 46% vs round_end.
(b) 2024 days 12-25 matches: round-1 label present in round_end x build_num x disputed.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, swap_expected_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
base = TOMES["2024"]
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
bs = scan(base, "bomb_state", ["match_key", "round", "tick", "event_type"]).collect()
bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
      .sort("tick").group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("bev")))
H = header(base, ["key", "build_num"]).rename({"key": "match_key"})
fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "c1"})
D = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t1") - pl.col("t_score")).alias("dT"), (pl.col("c1") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"),
    swap_expected_expr(pl.lit(24)).alias("sw"))
J = D.join(re_.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner").join(
    bo, on=["match_key", "round"], how="left").with_columns(
    pl.when(pl.col("bev") == "bomb_exploded").then(pl.lit(T))
    .when(pl.col("bev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bside"),
    day_expr().alias("day"))
J13 = J.filter((pl.col("round") == 13) & (~pl.col("sw")) & (pl.col("ss") != 0) & (pl.col("bside") != 0))
print(f"(a) round-13 bomb rows, all 2024 days, valid unambiguous: n={J13.height}")
print(f"    score side == bomb side: {(J13['ss'] == J13['bside']).mean():.4f}")
print(f"    score side == round_end winner: {(J13['ss'] == J13['winner_team_code']).mean():.4f}")
print("    by day group:")
print(J13.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("g"))
      .group_by("g").agg(pl.len().alias("n"), (pl.col("ss") == pl.col("bside")).mean().alias("eq_bomb"),
                         (pl.col("ss") == pl.col("winner_team_code")).mean().alias("eq_round_end")).sort("g"))

# (b) round-1 label presence x build x disputed, days 12-25
first = re_.group_by("match_key").agg(pl.col("round").min().alias("min_label"))
Dv = J.filter((~pl.col("sw")) & (pl.col("ss") != 0))
disp = Dv.filter(pl.col("winner_team_code") != pl.col("ss")).select("match_key").unique().with_columns(
    pl.lit(True).alias("disputed"))
Mb = first.join(H, on="match_key", how="left").join(disp, on="match_key", how="left").with_columns(
    pl.col("disputed").fill_null(False), pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32)
    .alias("day"))
Mb = Mb.filter(pl.col("day").is_between(12, 25)).with_columns(
    pl.when(pl.col("min_label") == 1).then(pl.lit("label1_present")).otherwise(pl.lit("starts_at_2")).alias("lab"))
print("\n(b) days 12-25 matches: round-1 label x build x disputed")
print(Mb.group_by(["lab", "build_num", "disputed"]).len().sort(["lab", "build_num", "disputed"]))
