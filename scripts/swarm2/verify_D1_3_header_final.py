"""verify D1-3: match-level arbitration with the demo header final scores (read-only, 2024 tome).

Header fields used: t_starters_score_final (final score of the team that started on T = team A),
ct_starters_score_final (team B). These come from the demo header, a channel separate from round_end
and round_state.

For each match:
  team A = the 5 players with team_code 2 (T) in round 1 (player_info, deduped). Team B = the rest.
  sideA(r) = side of team A in round r from player_info team codes (majority of A's present players).
  totA_src = number of rounds r where source_side(r) == sideA(r), for source in {winner, S2, kill, bomb}.
  totB_src = number of rounds where source_side(r) == 5 - sideA(r).
  match-level agreement: totA_src == header_A and totB_src == header_B.

ASSUMPTIONS:
  A1 'starters' = the team that started on T (team A), i.e. t_starters_score_final = team A final score.
  A2 a source's side for a round is used only where defined (null rounds are not counted).
  A3 a round where team A's side is undefined in player_info is not counted.
  A4 header key equals match_key (checked by the join coverage printed below).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def rd(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


HD = pl.concat([pl.read_parquet(f, columns=["key", "t_starters_score_final", "ct_starters_score_final"])
                for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))], how="vertical_relaxed")
HD = HD.unique("key", keep="first").rename({"key": "match_key", "t_starters_score_final": "hA", "ct_starters_score_final": "hB"})

PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"]).unique(["match_key", "round", "player_id_fixed"], keep="first")
RE = rd("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
BS = rd("bomb_state", ["match_key", "round", "event_type"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).sort(["match_key", "round", "tick"])
PD = rd("player_death", ["match_key", "round", "tick", "player_id_fixed"]).filter(pl.col("player_id_fixed").is_not_null()).with_columns(pl.col("player_id_fixed").cast(pl.Int64))

# team A membership from round 1
A_players = (PI.filter((pl.col("round") == 1) & (pl.col("team_code") == 2)).select(["match_key", "player_id_fixed"]).unique()
             .with_columns(pl.lit(True).alias("isA")))
PIA = PI.join(A_players, on=["match_key", "player_id_fixed"], how="left").with_columns(pl.col("isA").fill_null(False))
sideA = (PIA.filter(pl.col("isA")).group_by(["match_key", "round"])
         .agg((pl.col("team_code") == 2).sum().alias("aT"), (pl.col("team_code") == 3).sum().alias("aCT"))
         .with_columns(pl.when(pl.col("aT") > pl.col("aCT")).then(2).when(pl.col("aCT") > pl.col("aT")).then(3).otherwise(None).alias("sideA")))

# S2
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

# bomb outcome side
oc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"])
      .agg(pl.col("event_type").unique().alias("evs")))
oc = oc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("BOMB"))
oc = oc.select(["match_key", "round", "BOMB"])

# kill side (K2 = all CT dead -> 2 ; K3 = all T dead and no plant -> 3)
planted = BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("planted"))
nside = PI.group_by(["match_key", "round", "team_code"]).len().pivot(on="team_code", index=["match_key", "round"], values="len").rename(
    {"2": "nT", "3": "nCT"}).fill_null(0)
dead = (PD.join(PI.select(["match_key", "round", "player_id_fixed", "team_code"]), on=["match_key", "round", "player_id_fixed"], how="left")
        .filter(pl.col("team_code").is_not_null())
        .group_by(["match_key", "round", "team_code"]).agg(pl.col("player_id_fixed").n_unique().alias("nd"))
        .pivot(on="team_code", index=["match_key", "round"], values="nd").rename({"2": "dT", "3": "dCT"}).fill_null(0))
KL = (nside.join(dead, on=["match_key", "round"], how="left").join(planted, on=["match_key", "round"], how="left")
      .with_columns(pl.col("dT").fill_null(0), pl.col("dCT").fill_null(0), pl.col("planted").fill_null(False)))
KL = KL.with_columns(
    pl.when((pl.col("dCT") >= pl.col("nCT")) & ~(pl.col("dT") >= pl.col("nT"))).then(2)
      .when((pl.col("dT") >= pl.col("nT")) & ~(pl.col("dCT") >= pl.col("nCT")) & ~pl.col("planted")).then(3)
      .otherwise(None).alias("KILL")).select(["match_key", "round", "KILL"])

# per-round frame
R = (RE.join(sc, on=["match_key", "round"], how="left").join(oc, on=["match_key", "round"], how="left")
     .join(KL, on=["match_key", "round"], how="left").join(sideA.select(["match_key", "round", "sideA"]), on=["match_key", "round"], how="left")
     .with_columns(pl.col("winner_team_code").alias("WIN")))
R = R.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")).with_columns(
    pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

# per-match totals by source, team A and team B
srcs = ["WIN", "S2", "KILL", "BOMB"]
aggs = []
for s in srcs:
    aggs.append(((pl.col(s).is_not_null()) & (pl.col(s) == pl.col("sideA"))).sum().alias(f"A_{s}"))
    aggs.append(((pl.col(s).is_not_null()) & (pl.col(s) != pl.col("sideA")) & pl.col("sideA").is_not_null()).sum().alias(f"B_{s}"))
    aggs.append(pl.col(s).is_not_null().sum().alias(f"n_{s}"))
M = R.filter(pl.col("sideA").is_not_null()).group_by("match_key").agg(pl.col("dg").first().alias("dg"), *aggs)
M = M.join(HD, on="match_key", how="left")
print("matches with header:", M.join(HD, on="match_key", how="inner").height, " of", M.height)

hdr("1. match-level agreement of each source with header final scores (team A = T starters, team B = CT starters)")
for s in srcs:
    M = M.with_columns(((pl.col(f"A_{s}") == pl.col("hA")) & (pl.col(f"B_{s}") == pl.col("hB"))).alias(f"ok_{s}"))
rows = []
for dg in ["d01-07", "d12-25"]:
    q = M.filter(pl.col("dg") == dg)
    row = {"dg": dg, "n": q.height}
    for s in srcs:
        row[f"ok_{s}"] = int(q[f"ok_{s}"].sum())
    rows.append(row)
print(pl.DataFrame(rows))

hdr("2. per-match: d12-25 matches grouped by whether winner matches header (WIN ok) and whether S2 matches header (S2 ok)")
Q = M.filter(pl.col("dg") == "d12-25")
print(Q.group_by(["ok_WIN", "ok_S2"]).len().sort(["ok_WIN", "ok_S2"]))
print("\nd12-25 matches where WIN fails but S2 holds: examples (A_WIN, B_WIN, A_S2, B_S2, hA, hB, nWIN)")
print(Q.filter(~pl.col("ok_WIN") & pl.col("ok_S2")).select(["match_key", "A_WIN", "B_WIN", "A_S2", "B_S2", "hA", "hB", "n_WIN"]).head(8))
print("\nd12-25 matches where S2 fails but WIN holds: count", Q.filter(~pl.col("ok_S2") & pl.col("ok_WIN")).height)

hdr("3. raw totals: header vs sources, sum check (hA + hB vs n rounds per source)")
print(M.with_columns((pl.col("hA") + pl.col("hB")).alias("hsum"),
                     (pl.col("A_WIN") + pl.col("B_WIN")).alias("wsum"),
                     (pl.col("A_S2") + pl.col("B_S2")).alias("ssum"))
      .group_by("dg").agg(pl.len().alias("n"),
                          (pl.col("hsum") == pl.col("wsum")).sum().alias("hsum_eq_WINsum"),
                          (pl.col("hsum") == pl.col("ssum")).sum().alias("hsum_eq_S2sum"),
                          pl.col("hA").is_null().sum().alias("no_header")))

hdr("4. KILL and BOMB sources (defined rounds only) per match")
for s in ["KILL", "BOMB"]:
    q = M.filter(pl.col(f"n_{s}") > 0)
    print(s, "matches with >=1 defined round:", q.height, " ok (both teams):", int(q[f"ok_{s}"].sum()))
