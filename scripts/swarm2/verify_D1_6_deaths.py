"""verify D1-6 death-based winner check (swarm2 verifier): an independent event stream.

Logic: if every CT-roster player died in the round, T won the round; if every T-roster player died, CT won.
Roster per (match, round) = distinct player_id_fixed with team_code 2 (T) or 3 (CT) in player_info for that round.
Victims per (match, round) = distinct player_id_fixed in player_death, split by player_team_code at death.
The implication is logical given complete death data, so any 'implied winner' that contradicts a source is a
contradiction in that source. Calibration: the implied winner should agree with winner_team_code on the
agreeing (ADJ == winner) rows of the same rounds.

ASSUMPTIONS: A1 days; A2 player_count == 20 for the 305 set; A3 2 = T, 3 = CT; A8 duplicate exclusion;
A12 ASSUMPTION: player_info roster for the round is complete (10 players); rounds with a roster below 5 on a side
    are reported as 'roster_incomplete' and excluded from the implied-winner test.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count",
                                            "win_reason_message"]) for f in fl("round_end")], how="vertical_relaxed")
dups = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
RE = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE = RE.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "t_score", "ct_score"])
                for f in fl("round_state")], how="vertical_relaxed").sort(["match_key", "round", "tick"])
S = RS.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
    pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"))
S = S.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
      .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"))

PI = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id_fixed", "team_code"])
                for f in fl("player_info")], how="vertical_relaxed")
PI = PI.filter(pl.col("player_id_fixed").is_not_null()).unique(["match_key", "round", "player_id_fixed", "team_code"])
roster = (PI.group_by(["match_key", "round"]).agg(
    (pl.col("team_code") == 2).sum().alias("rosterT"),
    (pl.col("team_code") == 3).sum().alias("rosterCT")))
print("player_info (match, round) groups:", roster.height)

PD = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id_fixed", "player_team_code"])
                for f in fl("player_death")], how="vertical_relaxed")
PD = PD.filter(pl.col("player_id_fixed").is_not_null()).with_columns(pl.col("player_id_fixed").cast(pl.Int64))
PD = PD.unique(["match_key", "round", "player_id_fixed"])
victims = (PD.group_by(["match_key", "round"]).agg(
    (pl.col("player_team_code") == 2).sum().alias("victT"),
    (pl.col("player_team_code") == 3).sum().alias("victCT")))
print("player_death (match, round) groups:", victims.height)

A = (RE.join(S.select(["match_key", "round", "adj_side"]), on=["match_key", "round"], how="left")
       .join(roster, on=["match_key", "round"], how="left")
       .join(victims, on=["match_key", "round"], how="left")
       .with_columns(pl.col("victT").fill_null(0), pl.col("victCT").fill_null(0)))
A = A.with_columns(
    (pl.col("adj_side").is_not_null() & (pl.col("adj_side") != pl.col("winner_team_code"))).alias("adj_mm"),
    ((pl.col("rosterT") >= 5) & (pl.col("rosterCT") >= 5)).alias("roster_ok"),
    ((pl.col("rosterCT") > 0) & (pl.col("victCT") >= pl.col("rosterCT"))).alias("allCT_dead"),
    ((pl.col("rosterT") > 0) & (pl.col("victT") >= pl.col("rosterT"))).alias("allT_dead"))
A = A.with_columns(
    pl.when(~pl.col("roster_ok")).then(pl.lit("roster_incomplete"))
      .when(pl.col("allCT_dead") & ~pl.col("allT_dead")).then(pl.lit("T_by_elim"))
      .when(pl.col("allT_dead") & ~pl.col("allCT_dead")).then(pl.lit("CT_by_elim"))
      .when(pl.col("allT_dead") & pl.col("allCT_dead")).then(pl.lit("both_dead"))
      .otherwise(pl.lit("no_elim")).alias("elim"))
A = A.with_columns(
    pl.when(pl.col("elim") == "T_by_elim").then(pl.lit(2))
      .when(pl.col("elim") == "CT_by_elim").then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))
A = A.with_columns(
    (pl.col("elim_side").is_not_null() & (pl.col("elim_side") != pl.col("winner_team_code"))).alias("elim_mm"))

hdr("E0 calibration: implied elimination winner vs winner_team_code, all rounds 1..30, d12-25 + d01-07, pc20")
cal = A.filter((pl.col("player_count") == 20) & pl.col("elim_side").is_not_null())
print("rounds with elimination evidence:", cal.height, " implied winner == winner_code:", int((~cal["elim_mm"]).sum()),
      " contradictions:", int(cal["elim_mm"].sum()))
print(cal.group_by("dg").agg(pl.len().alias("n_elim_evidence"), pl.col("elim_mm").sum().alias("contradictions")))

hdr("E1 round 9 (player_count == 20, d12-25): elimination evidence by ADJ status")
r9 = A.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20))
r9 = r9.with_columns(
    pl.when(pl.col("adj_side").is_null()).then(pl.lit("no_side"))
      .when(pl.col("adj_mm")).then(pl.lit("ADJ_disagree"))
      .otherwise(pl.lit("ADJ_agree")).alias("status"))
print(r9.group_by(["status", "elim", "elim_side", "winner_team_code", "adj_side"]).len()
      .sort(["status", "len"], descending=[False, True]))

hdr("E2 round 9 disagree rows: elimination evidence vs winner code vs score side (summary)")
d9 = r9.filter(pl.col("status") == "ADJ_disagree")
print("disagree rows:", d9.height)
print("elimination evidence present:", d9.filter(pl.col("elim_side").is_not_null()).height)
print("  implied == winner_team_code:", d9.filter(pl.col("elim_side") == pl.col("winner_team_code")).height)
print("  implied == ADJ score side:  ", d9.filter(pl.col("elim_side") == pl.col("adj_side")).height)
print("roster-incomplete or no-elim:", d9.filter(pl.col("elim_side").is_null()).height)

hdr("E3 round 9 agree rows (control): elimination evidence vs winner code")
ag9 = r9.filter(pl.col("status") == "ADJ_agree")
print("agree rows with elimination evidence:", ag9.filter(pl.col("elim_side").is_not_null()).height,
      " implied == winner:", ag9.filter(pl.col("elim_side") == pl.col("winner_team_code")).height,
      " contradictions:", ag9.filter(pl.col("elim_mm")).height)

hdr("E4 round 9 disagree rows: elimination flags (what the death data says)")
print(d9.group_by("elim").len().sort("len", descending=True))
