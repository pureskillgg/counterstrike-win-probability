"""D3-2 verifier follow-up 3: round 9 score-vs-round_end disagreements (elimination undefined), and the
round_end reason codes for the switch-round defect. Reads only the G: 2024 tome (column-projected).
ASSUMPTIONS (marked): 5v5 rosters; a side with 5 distinct deaths was eliminated.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_2_followup3.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(260)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
DAY_RE = r"2024[/\\]02[/\\](\d\d)"
ROSTER = 5  # ASSUMPTION


def g_read(ch, cols):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def side_from_delta(dt, dct):
    return (pl.when((pl.col(dct) == 1) & (pl.col(dt) == 0)).then(pl.lit(CT))
            .when((pl.col(dt) == 1) & (pl.col(dct) == 0)).then(pl.lit(T))
            .otherwise(pl.lit(None, dtype=pl.Int64)))


RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                          "win_reason_message", "player_count"]).filter(pl.col("player_count") == 20)
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
PD = g_read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
day = lambda c: pl.col(c).str.extract(DAY_RE, 1).cast(pl.Int64)

outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("ev")))
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tb"),
                                            pl.col("ct_score").first().alias("cb")))
fe_nx = fe.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("tb").alias("ta"),
                   pl.col("cb").alias("ca")])
sc = fe.join(fe_nx, on=["match_key", "round"], how="left").with_columns(
    (pl.col("ta") - pl.col("tb")).alias("dt"), (pl.col("ca") - pl.col("cb")).alias("dc"))
sc = sc.with_columns(side_from_delta("dt", "dc").alias("score_side"))
kills = (PD.filter(pl.col("player_team_code").is_in([T, CT]))
         .group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("nd")))
dT = kills.filter(pl.col("player_team_code") == T).select(["match_key", "round", pl.col("nd").alias("dT")])
dC = kills.filter(pl.col("player_team_code") == CT).select(["match_key", "round", pl.col("nd").alias("dC")])
elim = dT.join(dC, on=["match_key", "round"], how="full", coalesce=True).with_columns(
    pl.col("dT").fill_null(0), pl.col("dC").fill_null(0)).with_columns(
    pl.when((pl.col("dT") == ROSTER) & (pl.col("dC") < ROSTER)).then(pl.lit(CT))
    .when((pl.col("dC") == ROSTER) & (pl.col("dT") < ROSTER)).then(pl.lit(T))
    .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))

base = (RE.join(outc, on=["match_key", "round"], how="left")
        .join(sc.select(["match_key", "round", "dt", "dc", "score_side"]), on=["match_key", "round"], how="left")
        .join(elim.select(["match_key", "round", "dT", "dC", "elim_side"]), on=["match_key", "round"], how="left")
        .with_columns(day("match_key").alias("day"), pl.col("ev").is_not_null().alias("bomb")))

# ---------------------------------------------------------------- R9: rows where round_end and score disagree
r9 = base.filter((pl.col("round") == 9) & (pl.col("day") >= 12) & pl.col("score_side").is_not_null())
r9 = r9.with_columns((pl.col("winner_team_code") != pl.col("score_side")).alias("dis"))
print(f"[R9] round 9 days 12-25 score-defined rows: {r9.height}; RE != score: {int(r9['dis'].sum())}")
bad = r9.filter(pl.col("dis"))
print("[R9] disagreeing rows: bomb?  counts:", bad.group_by("bomb").len().rows())
print("[R9] disagreeing rows: RE win_reason_message counts:")
print(bad.group_by("win_reason_message").len().sort("len", descending=True).rows())
print("[R9] disagreeing rows: elim_side counts (None = neither side wiped):")
print(bad.group_by(bad["elim_side"].fill_null(-1).alias("el")).len().rows())
print("[R9] disagreeing rows: (dT, dC) top 8:")
print(bad.group_by(["dT", "dC"]).len().sort("len", descending=True).head(8).rows())
print("[R9] disagreeing rows: by day:", bad.group_by("day").len().sort("day").rows())
print("[R9] disagreeing rows: RE winner counts:", bad.group_by("winner_team_code").len().rows())
print("[R9] agreeing rows, same grouping (bomb?):", r9.filter(~pl.col("dis")).group_by("bomb").len().rows())
print("[R9] agreeing rows: (dT, dC) top 8:")
print(r9.filter(~pl.col("dis")).group_by(["dT", "dC"]).len().sort("len", descending=True).head(8).rows())

# ---------------------------------------------------------------- switch-round defect: reason codes for the RE winner
print("\n[S] round 13 days 12-25 non-bomb rows with elim defined: RE winner x RE reason x elim:")
s13 = base.filter((pl.col("round") == 13) & (pl.col("day") >= 12) & ~pl.col("bomb") & pl.col("elim_side").is_not_null())
print(s13.group_by(["winner_team_code", "win_reason_message", "elim_side"]).len().sort("len", descending=True).rows())
print("\n[S] round 13 days 12-25 bomb rows: RE winner x RE reason x bomb side (from outcome):")
bo = base.filter((pl.col("round") == 13) & (pl.col("day") >= 12) & pl.col("bomb")).with_columns(
    pl.when(pl.col("ev") == "bomb_exploded").then(pl.lit(T)).otherwise(pl.lit(CT)).alias("bomb_side"))
print(bo.group_by(["winner_team_code", "win_reason_message", "bomb_side"]).len().sort("len", descending=True).rows())

print("\nDONE")
