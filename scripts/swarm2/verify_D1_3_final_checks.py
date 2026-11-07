"""verify D1-3: final checks (read-only, 2024 tome). Team-level S2T as in verify_D1_3_header_final2.py.

Reports:
  F1 affected matches by bomb evidence (claim: 480) and their overlap with S2T-disagreement matches.
  F2 S2T-disagreement matches by round (9, 13, 25, 28) and union; comparison with bomb-observable counts.
  F3 round-13 S2T disagreement rate by day (d12-25) to compare with the claimed bomb-based day rates.
  F4 header arbitration restricted to matches whose header sums equal the round count (header-consistent).
  F5 rounds 14-24 disagreement count under S2T (claim: no blanket second-half flip).
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
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).sort(["match_key", "round", "tick"])
BS = rd("bomb_state", ["match_key", "round", "event_type"])

first_r = PI.group_by("match_key").agg(pl.col("round").min().alias("r0"))
TA = (PI.join(first_r, on="match_key").filter((pl.col("round") == pl.col("r0")) & (pl.col("team_code") == 2))
      .select(["match_key", "player_id_fixed"]).unique().with_columns(pl.lit(True).alias("isA")))
PIA = PI.join(TA, on=["match_key", "player_id_fixed"], how="left").with_columns(pl.col("isA").fill_null(False))
sideA = (PIA.filter(pl.col("isA")).group_by(["match_key", "round"])
         .agg((pl.col("team_code") == 2).sum().alias("aT"), (pl.col("team_code") == 3).sum().alias("aCT"))
         .with_columns(pl.when(pl.col("aT") > pl.col("aCT")).then(2).when(pl.col("aCT") > pl.col("aT")).then(3).otherwise(None).alias("sideA"))
         .select(["match_key", "round", "sideA"]))
pre = (RS.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
       .agg(pl.col("t_score").first().alias("tp"), pl.col("ct_score").first().alias("cp")))
pre0 = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tp0"), pl.col("ct_score").first().alias("cp0"))
pv = pre0.join(pre, on=["match_key", "round"], how="left").with_columns(
    pl.coalesce(["tp", "tp0"]).alias("tpre"), pl.coalesce(["cp", "cp0"]).alias("cpre")).select(["match_key", "round", "tpre", "cpre"])
post_last = (RS.filter(pl.col("event_type") == "round_officially_ended").group_by(["match_key", "round"])
             .agg(pl.col("t_score").last().alias("tq"), pl.col("ct_score").last().alias("cq")))
post_last0 = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").last().alias("tq0"), pl.col("ct_score").last().alias("cq0"))
pl_ = post_last0.join(post_last, on=["match_key", "round"], how="left").with_columns(
    pl.coalesce(["tq", "tq0"]).alias("tqf"), pl.coalesce(["cq", "cq0"]).alias("cqf")).select(["match_key", "round", "tqf", "cqf"])
R = (RE.join(sideA, on=["match_key", "round"], how="left").join(pv, on=["match_key", "round"], how="left").join(pl_, on=["match_key", "round"], how="left"))
nxt = pv.with_columns((pl.col("round") - 1).alias("round")).rename({"tpre": "tnext", "cpre": "cnext"})
nsd = sideA.with_columns((pl.col("round") - 1).alias("round")).rename({"sideA": "sideA_next"})
R = R.join(nxt, on=["match_key", "round"], how="left").join(nsd, on=["match_key", "round"], how="left")
R = R.with_columns(
    pl.when(pl.col("sideA") == 2).then(pl.col("tpre")).otherwise(pl.col("cpre")).alias("A_pre"),
    pl.when(pl.col("sideA") == 2).then(pl.col("cpre")).otherwise(pl.col("tpre")).alias("B_pre"))
R = R.with_columns(
    pl.when(pl.col("tnext").is_not_null()).then(pl.when(pl.col("sideA_next") == 2).then(pl.col("tnext")).otherwise(pl.col("cnext")))
      .otherwise(pl.when(pl.col("sideA") == 2).then(pl.col("tqf")).otherwise(pl.col("cqf"))).alias("A_post"),
    pl.when(pl.col("tnext").is_not_null()).then(pl.when(pl.col("sideA_next") == 2).then(pl.col("cnext")).otherwise(pl.col("tnext")))
      .otherwise(pl.when(pl.col("sideA") == 2).then(pl.col("cqf")).otherwise(pl.col("tqf"))).alias("B_post"))
R = R.with_columns((pl.col("A_post") - pl.col("A_pre")).alias("gA"), (pl.col("B_post") - pl.col("B_pre")).alias("gB"))
R = R.with_columns(
    pl.when((pl.col("gA") == 1) & (pl.col("gB") == 0)).then(pl.col("sideA"))
      .when((pl.col("gB") == 1) & (pl.col("gA") == 0)).then(5 - pl.col("sideA"))
      .otherwise(None).alias("S2T"))
R = R.filter(pl.col("sideA").is_not_null()).with_columns(pl.col("winner_team_code").alias("WIN"))
R = R.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")).with_columns(
    pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

# bomb mismatch (claim definition)
oc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"])
      .agg(pl.col("event_type").unique().alias("evs")))
oc = oc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
R = R.join(oc.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
R = R.with_columns(((pl.col("player_count") == 20) & pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("WIN"))).alias("bomb_mm"))
R = R.with_columns((pl.col("S2T").is_not_null() & (pl.col("S2T") != pl.col("WIN"))).alias("S2T_dis"))

hdr("F1. affected matches by bomb evidence vs S2T disagreement")
bm = R.filter(pl.col("bomb_mm")).select("match_key").unique()
print("bomb-mismatch rounds:", int(R.filter(pl.col("bomb_mm")).height), "  matches with >=1 bomb mismatch:", bm.height, "  (claim: 480)")
sd = R.filter(pl.col("S2T_dis")).select("match_key").unique()
print("S2T-disagreement matches (any round):", sd.height, "  of which d12-25:", sd.join(R.select(["match_key", "dg"]).unique(), on="match_key").filter(pl.col("dg") == "d12-25").height)
print("bomb-mismatch matches that also have an S2T disagreement:", bm.join(sd, on="match_key", how="inner").height, "of", bm.height)

hdr("F2. S2T disagreement by round (d12-25): matches and rounds, vs bomb-observable rounds")
d12 = R.filter(pl.col("dg") == "d12-25")
rows = []
for rr in [9, 13, 25, 28]:
    q = d12.filter(pl.col("round") == rr)
    s2t_def = q.filter(pl.col("S2T").is_not_null())
    rows.append({"round": rr, "rounds_with_S2T": s2t_def.height, "S2T_disagree": int(s2t_def["S2T_dis"].sum()),
                 "rate": round(float(s2t_def["S2T_dis"].mean()), 4),
                 "bomb_rounds": int(q.filter((pl.col("player_count") == 20) & pl.col("bomb_side").is_not_null()).height),
                 "bomb_mm": int(q["bomb_mm"].sum())})
print(pl.DataFrame(rows))
print("union of S2T-disagreement rounds (d12-25) by round number (all rounds):")
print(d12.filter(pl.col("S2T_dis")).group_by("round").len().sort("round"))
print("S2T disagreement rounds in d01-07 (all rounds):", R.filter((pl.col("dg") == "d01-07") & pl.col("S2T_dis")).height)
r1424 = d12.filter(pl.col("round").is_between(14, 24) & pl.col("S2T").is_not_null())
print("F5 rounds 14-24 (d12-25) with S2T defined:", r1424.height, " disagreements:", int(r1424["S2T_dis"].sum()))
r1_8 = d12.filter(pl.col("round").is_between(1, 8) & pl.col("S2T").is_not_null())
r1012 = d12.filter(pl.col("round").is_in([10, 11, 12]) & pl.col("S2T").is_not_null())
print("rounds 1-8 disagreements:", int(r1_8["S2T_dis"].sum()), " rounds 10-12 disagreements:", int(r1012["S2T_dis"].sum()))

hdr("F3. round-13 S2T disagreement by day (d12-25) vs claimed bomb-based day rates")
r13 = d12.filter((pl.col("round") == 13) & pl.col("S2T").is_not_null())
print(r13.group_by("day").agg(pl.len().alias("n"), pl.col("S2T_dis").sum().alias("S2T_dis"),
                              pl.col("S2T_dis").mean().round(3).alias("rate")).sort("day"))

hdr("F4. header arbitration restricted to header-consistent matches (hA+hB == rounds with a team winner)")
M = (R.group_by("match_key").agg(pl.col("dg").first().alias("dg"), pl.len().alias("n_rounds"),
                                 (pl.col("WIN") == pl.col("sideA")).sum().alias("A_WIN"),
                                 (pl.col("WIN") == 5 - pl.col("sideA")).sum().alias("B_WIN"),
                                 pl.col("S2T").is_not_null().sum().alias("n_S2T"),
                                 (pl.col("S2T") == pl.col("sideA")).sum().alias("A_S2T"),
                                 (pl.col("S2T") == 5 - pl.col("sideA")).sum().alias("B_S2T"),
                                 pl.col("S2T_dis").sum().alias("n_dis"))
     .join(HD, on="match_key", how="left"))
M = M.with_columns(((pl.col("hA") + pl.col("hB")) == pl.col("n_rounds")).alias("hsum_ok"),
                   ((pl.col("A_WIN") == pl.col("hA")) & (pl.col("B_WIN") == pl.col("hB"))).alias("ok_WIN"),
                   ((pl.col("A_S2T") == pl.col("hA")) & (pl.col("B_S2T") == pl.col("hB"))).alias("ok_S2T"),
                   (pl.col("n_S2T") == pl.col("n_rounds")).alias("complete"))
print(M.group_by(["dg", "hsum_ok"]).len().sort(["dg", "hsum_ok"]))
Hc = M.filter((pl.col("dg") == "d12-25") & pl.col("hsum_ok") & pl.col("complete"))
print("d12-25 header-consistent AND complete:", Hc.height,
      " WIN ok:", int(Hc["ok_WIN"].sum()), " S2T ok:", int(Hc["ok_S2T"].sum()))
print("  of header-consistent complete matches with WIN failing: n_dis>=1 ->", Hc.filter(~pl.col("ok_WIN")).height,
      "; of these S2T ok ->", Hc.filter(~pl.col("ok_WIN") & pl.col("ok_S2T")).height)
Hd = M.filter((pl.col("dg") == "d01-07") & pl.col("hsum_ok"))
print("d01-07 header-consistent matches:", Hd.height, " of", M.filter(pl.col("dg") == "d01-07").height)
