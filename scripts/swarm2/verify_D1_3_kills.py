"""verify D1-3: kill-based (elimination) side label, independent of round_end, round_state and bomb events.

Read-only, 2024 tome. Channels: player_info, player_death, bomb_state (planted only), round_end, round_state.

Elimination rule (game-logic, ASSUMPTION A1 below):
  K2 = all CT players dead (distinct victims on team_code 3, from player_info) and not all T dead -> T (2) won.
  K3 = all T players dead and not all CT dead and no bomb planted that round -> CT (3) won.
  KX = both sides fully dead -> ambiguous (not used).
  KN = neither side fully dead -> no elimination evidence (time-out or bomb round).
  KP = all T dead with bomb planted -> not decidable by elimination (explosion or defuse decides).

ASSUMPTIONS:
  A1 a round cannot end by elimination of one side while the other side has living players, except
     via bomb events (explosion/defuse); a planted bomb keeps the round running after all Ts die.
  A2 team_code in player_info is the side of the player in that round (2 = T, 3 = CT), as in the claim.
  A3 5 players per side per round; sides with fewer players in player_info are counted with their real n.
  A4 duplicates in player_info (3 matches, per the claim) are dropped by (match, round, player) keep first.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(20)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def rd(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


PI_raw = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
PI = PI_raw.unique(["match_key", "round", "player_id_fixed"], keep="first")
print("player_info rows", PI_raw.height, " after dedupe", PI.height)
PD = rd("player_death", ["match_key", "round", "tick", "player_id_fixed", "player_team_code"])
PD = PD.filter(pl.col("player_id_fixed").is_not_null()).with_columns(pl.col("player_id_fixed").cast(pl.Int64))
print("player_death rows (with player_id_fixed)", PD.height)
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS = rd("bomb_state", ["match_key", "round", "event_type"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).sort(["match_key", "round", "tick"])

# bomb
planted = BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("planted"))
oc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
oc = oc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))

# S2 score side (event-anchored, same definition as verify_D1_3_recompute.py)
pre = (RS.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
       .agg(pl.col("t_score").first().alias("tp"), pl.col("ct_score").first().alias("cp")))
pre0 = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tp0"), pl.col("ct_score").first().alias("cp0"))
post = (RS.filter(pl.col("event_type") == "round_officially_ended").group_by(["match_key", "round"])
        .agg(pl.col("t_score").last().alias("tq"), pl.col("ct_score").last().alias("cq")))
post0 = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").last().alias("tq0"), pl.col("ct_score").last().alias("cq0"))
sc = pre0.join(pre, on=["match_key", "round"], how="left").join(post0, on=["match_key", "round"], how="left").join(post, on=["match_key", "round"], how="left")
sc = sc.with_columns(pl.coalesce(["tp", "tp0"]).alias("tp"), pl.coalesce(["cp", "cp0"]).alias("cp"),
                     pl.coalesce(["tq", "tq0"]).alias("tq"), pl.coalesce(["cq", "cq0"]).alias("cq"))
sc = sc.with_columns(
    pl.when(((pl.col("tq") - pl.col("tp")) == 1) & ((pl.col("cq") - pl.col("cp")) == 0)).then(2)
      .when(((pl.col("cq") - pl.col("cp")) == 1) & ((pl.col("tq") - pl.col("tp")) == 0)).then(3)
      .otherwise(None).alias("S2")).select(["match_key", "round", "S2"])

# per-round side counts and dead counts
nside = PI.group_by(["match_key", "round", "team_code"]).len().pivot(on="team_code", index=["match_key", "round"], values="len").rename(
    {"2": "nT", "3": "nCT"}).fill_null(0)
dead = (PD.join(PI.select(["match_key", "round", "player_id_fixed", "team_code"]), on=["match_key", "round", "player_id_fixed"], how="left")
        .filter(pl.col("team_code").is_not_null())
        .group_by(["match_key", "round", "team_code"]).agg(pl.col("player_id_fixed").n_unique().alias("nd"))
        .pivot(on="team_code", index=["match_key", "round"], values="nd").rename({"2": "dT", "3": "dCT"}).fill_null(0))
# rounds with no death rows get zero dead
K = (RE.join(nside, on=["match_key", "round"], how="left").join(dead, on=["match_key", "round"], how="left")
     .join(planted, on=["match_key", "round"], how="left").join(oc, on=["match_key", "round"], how="left")
     .join(sc, on=["match_key", "round"], how="left")
     .with_columns(pl.col("nT").fill_null(0), pl.col("nCT").fill_null(0), pl.col("dT").fill_null(0), pl.col("dCT").fill_null(0),
                   pl.col("planted").fill_null(False),
                   pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
K = K.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
K = K.with_columns(
    (pl.col("dCT") >= pl.col("nCT")).alias("allCT_dead"),
    (pl.col("dT") >= pl.col("nT")).alias("allT_dead"),
).with_columns(
    pl.when((pl.col("nT") == 0) | (pl.col("nCT") == 0)).then(pl.lit("no_players"))
      .when(pl.col("allCT_dead") & pl.col("allT_dead")).then(pl.lit("KX_both"))
      .when(pl.col("allCT_dead")).then(pl.lit("K2_T"))
      .when(pl.col("allT_dead") & ~pl.col("planted")).then(pl.lit("K3_CT"))
      .when(pl.col("allT_dead") & pl.col("planted")).then(pl.lit("KP_planted"))
      .otherwise(pl.lit("KN_none")).alias("kill_class"))
K = K.with_columns(
    pl.when(pl.col("kill_class") == "K2_T").then(2).when(pl.col("kill_class") == "K3_CT").then(3).otherwise(None).alias("kill_side"),
    pl.when(pl.col("bomb_side").is_null()).then(pl.lit("no_bomb")).otherwise(pl.lit("bomb")).alias("bstat"),
    (pl.col("winner_team_code") != pl.col("S2")).alias("W_ne_S2"))

hdr("0. coverage: player counts per round and kill classes")
print("rounds:", K.height, " nT/nCT distribution (top):")
print(K.group_by(["nT", "nCT"]).len().sort("len", descending=True).head(8))
print("\nkill_class by day group:")
print(K.group_by(["dg", "kill_class"]).len().sort(["dg", "kill_class"]))

hdr("1. KILL-based side vs round_end winner (rounds where kill side is defined: K2 or K3)")
Kd = K.filter(pl.col("kill_side").is_not_null())
print("rounds with kill-defined side:", Kd.height)
print(Kd.group_by(["dg", "kill_side", "winner_team_code"]).len().sort(["dg", "kill_side", "winner_team_code"]))
print("\nkill-defined rounds: winner == kill side, by round number (d12-25 and d01-07 pooled), disagreements listed:")
dis = Kd.filter(pl.col("kill_side") != pl.col("winner_team_code"))
print("disagreements total:", dis.height, " by round:")
print(dis.group_by(["dg", "round"]).len().sort(["dg", "round"]))
print("\nagreement rate by round (d12-25, rounds 9/13/25/27/28 and a control round 14):")
print(Kd.filter((pl.col("dg") == "d12-25") & pl.col("round").is_in([9, 13, 14, 25, 27, 28]))
      .group_by("round").agg(pl.len().alias("n"), (pl.col("kill_side") == pl.col("winner_team_code")).sum().alias("agree_winner"),
                             (pl.col("kill_side") == pl.col("S2")).sum().alias("agree_S2"))
      .sort("round"))

hdr("2. ROUND 13 (d12-25): kill side vs winner vs S2 vs bomb")
r13 = K.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25"))
print("round-13 rows d12-25:", r13.height)
print(r13.group_by(["W_ne_S2", "kill_class", "bstat"]).len().sort(["W_ne_S2", "kill_class", "bstat"]))
print("\nround-13 d12-25 where winner != S2 (the 2080 rows): kill_class distribution")
print(r13.filter(pl.col("W_ne_S2")).group_by(["kill_class", "bstat"]).len().sort(["kill_class", "bstat"]))
print("\nround-13 d12-25 where winner == S2: kill_class distribution")
print(r13.filter(~pl.col("W_ne_S2")).group_by(["kill_class", "bstat"]).len().sort(["kill_class", "bstat"]))
print("\nround-13 d01-07 (control, all days clean per claim):")
r13c = K.filter((pl.col("round") == 13) & (pl.col("dg") == "d01-07"))
print(r13c.group_by(["W_ne_S2", "kill_class"]).len().sort(["W_ne_S2", "kill_class"]))

hdr("3. ROUND 9, 25, 27, 28 (d12-25): kill class vs winner vs S2")
for rr in [9, 25, 27, 28, 14]:
    q = K.filter((pl.col("round") == rr) & (pl.col("dg") == "d12-25"))
    print(f"\n-- round {rr}, d12-25, n={q.height}")
    print(q.group_by(["W_ne_S2", "kill_class"]).len().sort(["W_ne_S2", "kill_class"]))

hdr("4. ORACLE CHECK: kill side vs bomb side on bomb rounds (d12-25 and d01-07)")
B = K.filter(pl.col("bomb_side").is_not_null() & pl.col("kill_side").is_not_null())
print(B.group_by(["dg", "round"]).agg(pl.len().alias("n"), (pl.col("kill_side") == pl.col("bomb_side")).sum().alias("kill_eq_bomb"),
                                     (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("winner_eq_bomb"))
      .filter(pl.col("round").is_in([9, 13, 25, 27, 28]) | (pl.col("n") > 0)).sort(["dg", "round"]).filter(pl.col("round").is_in([9, 13, 25, 27, 28])))

hdr("5. kill class vs S2 for all rounds (agreement where both defined) by round number, d12-25")
Q = K.filter((pl.col("dg") == "d12-25") & pl.col("kill_side").is_not_null() & pl.col("S2").is_not_null())
print(Q.group_by("round").agg(pl.len().alias("n"), (pl.col("kill_side") != pl.col("S2")).sum().alias("kill_ne_S2"))
      .filter(pl.col("kill_ne_S2") > 0).sort("round"))
