"""verify D4-5, step 17: is the dispute tied to demo build? 2024 by day group and 2026 control.

For every match: build_num, rushb_version, day group, and whether the match has any disputed valid round
(score side != round_end winner, unambiguous, no swap after the round by the match's convention).
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)


def per_match(era):
    base = TOMES[era]
    rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
    re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
        ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
    cols = ["key", "build_num", "rushb_version", "match_date"] + (["max_rounds"] if era == "2026" else [])
    Hf = header(base, cols).rename({"key": "match_key"})
    if era == "2024":
        Hf = Hf.with_columns(pl.lit(24).alias("max_rounds"))
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
          .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
    fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "c1"})
    D = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
        (pl.col("t1") - pl.col("t_score")).alias("dT"), (pl.col("c1") - pl.col("ct_score")).alias("dC")).with_columns(
        pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
        .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"))
    D = D.join(Hf.select(["match_key", "max_rounds"]), on="match_key", how="left").with_columns(
        (((pl.col("round") == pl.col("max_rounds") // 2) |
          ((pl.col("round") > pl.col("max_rounds")) & (((pl.col("round") - pl.col("max_rounds") - 3) % 6) == 0)))
         ).alias("sw"))
    J = D.filter((~pl.col("sw")) & (pl.col("ss") != 0)).join(
        re_.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner").with_columns(
        (pl.col("ss") != pl.col("winner_team_code")).alias("dis"))
    pm = J.group_by("match_key").agg(pl.col("dis").sum().alias("n_dis"), pl.len().alias("n_valid"))
    out = Hf.join(pm, on="match_key", how="left").with_columns(
        pl.col("n_dis").fill_null(0), pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32)
        .alias("day"))
    out = out.with_columns((pl.col("n_dis") > 0).alias("disputed"), pl.lit(era).alias("era"))
    return out


A = per_match("2024")
A = A.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).when(pl.col("day").is_between(12, 25))
                   .then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("grp"))
print("2024: build_num x day group x disputed (matches)")
print(A.group_by(["grp", "build_num", "disputed"]).len().sort(["grp", "build_num", "disputed"]))
print("\n2024: rushb_version x day group x disputed")
print(A.group_by(["grp", "rushb_version", "disputed"]).len().sort(["grp", "rushb_version", "disputed"]))
print("\n2024 days 12-25: dispute rate by build_num")
print(A.filter(pl.col("grp") == "d12-25").group_by("build_num").agg(
    pl.len().alias("matches"), pl.col("disputed").sum().alias("disputed"),
    pl.col("disputed").mean().alias("rate")).sort("build_num"))

B = per_match("2026")
print("\n2026 control: build_num, rushb_version, disputed matches (score side != round_end)")
print(B.group_by(["build_num", "rushb_version"]).agg(pl.len().alias("matches"),
                                                      pl.col("disputed").sum().alias("disputed")).sort("build_num"))
print("2026 total disputed matches:", int(B["disputed"].sum()), "of", B.height)
