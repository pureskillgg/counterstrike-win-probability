"""verify D1-3: round-by-round trace of two matches (read-only, 2024 tome).

Match A: first d01-07 match with a full 5v5 at round 13 (clean reference).
Match B: first d12-25 match with a round-13 winner vs S2 disagreement and a bomb mismatch at round 13.
For each round prints: winner_team_code, win_reason_message, S2 (score side), score pre/post,
kill class (from player_death + player_info), planted flag, bomb side, and (nT, nCT).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(34)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def rd(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message", "player_count"]).unique(["match_key", "round"], keep="first")
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"]).sort(["match_key", "round", "tick"])
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"]).unique(["match_key", "round", "player_id_fixed"], keep="first")
PD = rd("player_death", ["match_key", "round", "tick", "player_id_fixed"]).filter(pl.col("player_id_fixed").is_not_null()).with_columns(pl.col("player_id_fixed").cast(pl.Int64))

# choose matches
clean_keys = RE.filter(pl.col("match_key").str.contains(r"csds/2024/02/0[1-7]/")).select("match_key").unique().sort("match_key")
mA = clean_keys.row(0)[0]
oc = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"]).agg(
    pl.col("event_type").unique().alias("evs"))
oc = oc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
bad = (RE.filter((pl.col("round") == 13) & (pl.col("player_count") == 20))
       .join(oc.filter(pl.col("round") == 13), on=["match_key", "round"], how="inner")
       .filter(pl.col("bomb_side") != pl.col("winner_team_code")).select("match_key").unique().sort("match_key"))
mB = bad.row(0)[0]
print("MATCH A (clean, d01-07):", mA)
print("MATCH B (round-13 bomb mismatch, d12-25):", mB)


def side_table(mk):
    rs = RS.filter(pl.col("match_key") == mk)
    pre = rs.filter(pl.col("event_type") == "round_freeze_end").group_by("round").agg(
        pl.col("t_score").first().alias("t_pre"), pl.col("ct_score").first().alias("ct_pre"))
    pre0 = rs.group_by("round").agg(pl.col("t_score").first().alias("t_pre0"), pl.col("ct_score").first().alias("ct_pre0"))
    post = rs.filter(pl.col("event_type") == "round_officially_ended").group_by("round").agg(
        pl.col("t_score").last().alias("t_post"), pl.col("ct_score").last().alias("ct_post"))
    post0 = rs.group_by("round").agg(pl.col("t_score").last().alias("t_post0"), pl.col("ct_score").last().alias("ct_post0"))
    t = pre0.join(pre, on="round", how="left").join(post0, on="round").join(post, on="round", how="left")
    t = t.with_columns(pl.coalesce(["t_pre", "t_pre0"]).alias("t_pre"), pl.coalesce(["ct_pre", "ct_pre0"]).alias("ct_pre"),
                       pl.coalesce(["t_post", "t_post0"]).alias("t_post"), pl.coalesce(["ct_post", "ct_post0"]).alias("ct_post"))
    t = t.with_columns(
        pl.when(((pl.col("t_post") - pl.col("t_pre")) == 1) & ((pl.col("ct_post") - pl.col("ct_pre")) == 0)).then(2)
          .when(((pl.col("ct_post") - pl.col("ct_pre")) == 1) & ((pl.col("t_post") - pl.col("t_pre")) == 0)).then(3)
          .otherwise(None).alias("S2"))
    return t.select(["round", "t_pre", "ct_pre", "t_post", "ct_post", "S2"])


def trace(mk):
    print("\n" + "=" * 110 + f"\n{mk}\n" + "=" * 110)
    rt = side_table(mk)
    re_ = RE.filter(pl.col("match_key") == mk).select(["round", "winner_team_code", "win_reason_message", "player_count"])
    # players per side per round
    pi = PI.filter(pl.col("match_key") == mk)
    ns = pi.group_by(["round", "team_code"]).len().pivot(on="team_code", index="round", values="len").rename({"2": "nT", "3": "nCT"}).fill_null(0)
    pdm = PD.filter(pl.col("match_key") == mk).join(pi.select(["round", "player_id_fixed", "team_code"]), on=["round", "player_id_fixed"], how="left")
    dd = pdm.filter(pl.col("team_code").is_not_null()).group_by(["round", "team_code"]).agg(pl.col("player_id_fixed").n_unique().alias("nd")) \
        .pivot(on="team_code", index="round", values="nd").rename({"2": "dT", "3": "dCT"}).fill_null(0)
    pl_ = BS.filter((pl.col("match_key") == mk) & (pl.col("event_type") == "bomb_planted")).select("round").unique().with_columns(pl.lit(True).alias("planted"))
    ob = oc.filter(pl.col("match_key") == mk).select(["round", "bomb_side"])
    out = (re_.join(rt, on="round", how="full", coalesce=True).join(ns, on="round", how="left").join(dd, on="round", how="left")
           .join(pl_, on="round", how="left").join(ob, on="round", how="left").sort("round"))
    out = out.with_columns(
        pl.col("nT").fill_null(0), pl.col("nCT").fill_null(0), pl.col("dT").fill_null(0), pl.col("dCT").fill_null(0),
        (pl.col("winner_team_code") != pl.col("S2")).alias("W_ne_S2"),
        (pl.col("bomb_side") != pl.col("winner_team_code")).alias("bomb_mm"))
    print(out.select(["round", "winner_team_code", "S2", "W_ne_S2", "win_reason_message", "t_pre", "t_post", "ct_pre", "ct_post",
                      "nT", "nCT", "dT", "dCT", "planted", "bomb_side", "bomb_mm", "player_count"]))


trace(mA)
trace(mB)
