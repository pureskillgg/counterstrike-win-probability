"""verify_D7_8_main: independent check of the D7-8 claim (format mix explains the 2024 round-13 side defect).

Read-only on the 2024 tome. Output goes to stdout; the caller redirects it to logs/swarm2/verify_D7_8.log.

Independent sources (not the D7 ledger):
  - player_info.team_code per (match, round, player): per-round side of every player.
  - round_state round_freeze_end t_score / ct_score: score deltas (round winner).
  - round_end.winner_team_code: the field under test.
  - bomb_state bomb_defused / bomb_exploded: outcome truth used by the defect definition.
  - header.build_num, header.match_date, match_key day: build and date.
The D7 min-swap ledger (logs/swarm2/D7_ledger_dp.parquet) is used only for a comparison at the end.

ASSUMPTIONS (marked where used):
  A1 team A = players on T (team_code 2) in the first round present in player_info for the match.
  A2 a round's side for a group = majority of its players' team_code; ties -> unknown.
  A3 round winner = the group whose score rose by exactly 1 between freeze_end r and freeze_end r+1, the other unchanged.
  A4 'clean' rounds = round_end.player_count == 20 (same filter as the D7 scripts).
  A5 bomb side: bomb_defused -> 3 (CT), bomb_exploded -> 2 (T), as in scripts/swarm/e2_side_check_r13.py.
"""
import bisect
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(ch):
    return pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_{ch}{SUF}")])


print("polars", pl.__version__)

# ---------- header ----------
H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name"]) for f in files(BASE / "header")])
H = H.rename({"key": "match_key"}).with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
)
print("header rows:", H.height, "| unique keys:", H["match_key"].n_unique(),
      "| null mdt:", H["mdt"].null_count(), "| null day:", H["day"].null_count())
print("build_num distinct:", H["build_num"].n_unique())
print("build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
NMATCH = H.height

# ---------- freeze_end scores ----------
FE_raw = (scan("round_state").filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "t_score", "ct_score"]).collect())
dup_fe = FE_raw.height - FE_raw.unique(subset=["match_key", "round"]).height
FE = FE_raw.unique(subset=["match_key", "round"], keep="first")
print("\nfreeze_end rows:", FE_raw.height, "| duplicate (match,round):", dup_fe, "| kept:", FE.height)
print("matches with freeze_end:", FE["match_key"].n_unique(), "| max round:", FE["round"].max())

# ---------- round_end ----------
RE_raw = scan("round_end").select(["match_key", "round", "winner_team_code", "player_count"]).collect()
RE_g = RE_raw.group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").n_unique().alias("nw"),
    pl.col("winner_team_code").max().alias("winner_team_code"),
    pl.col("player_count").max().alias("player_count"),
)
print("\nround_end rows:", RE_raw.height, "| (match,round) keys:", RE_g.height,
      "| keys with conflicting winners:", RE_g.filter(pl.col("nw") > 1).height)
RE = RE_g.select(["match_key", "round", "winner_team_code", "player_count"])
print("round_end player_count counts:", RE.group_by("player_count").len().sort("len", descending=True).head(5).rows())
cov = FE.join(RE, on=["match_key", "round"], how="left")
print("freeze_end rounds with NO round_end row:", cov.filter(pl.col("winner_team_code").is_null()).height)
cov2 = RE.join(FE, on=["match_key", "round"], how="left")
print("round_end rows with NO freeze_end row:", cov2.filter(pl.col("t_score").is_null()).height)

# ---------- bomb outcome and planter ----------
BS = (scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "event_type", "player_team_code"]).collect())
BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"])
      .agg(pl.col("event_type").n_unique().alias("n_out"),
           pl.col("event_type").unique().alias("evs")))
print("\nbomb outcome rounds:", BO.height, "| with BOTH defused and exploded:", BO.filter(pl.col("n_out") > 1).height)
BO = BO.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
PL = (BS.filter(pl.col("event_type") == "bomb_planted")
      .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side")))
print("planted rounds:", PL.height, "| planter_side counts:", PL.group_by("planter_side").len().sort("planter_side").rows())

# ---------- player_info per-round team codes ----------
PI = (scan("player_info").select(["match_key", "round", "player_id", "team_code"])
      .filter(pl.col("team_code").is_in([2, 3])).collect())
dup_pi = PI.height - PI.unique(subset=["match_key", "round", "player_id"]).height
print("\nplayer_info rows (team 2/3):", PI.height, "| duplicate (match,round,player):", dup_pi,
      "| matches:", PI["match_key"].n_unique())
print("rounds per match in player_info (min/max):", PI.group_by("match_key").agg(pl.col("round").min().alias("mn"), pl.col("round").max().alias("mx")).select(pl.col("mn").min().alias("min_r0"), pl.col("mx").max().alias("max_r")).rows())

# A1: team A = players on T (2) in the first round present for the match
r0 = PI.group_by("match_key").agg(pl.col("round").min().alias("r0"))
REF = (PI.join(r0, on="match_key").filter(pl.col("round") == pl.col("r0"))
       .select(["match_key", "player_id", "team_code"]).rename({"team_code": "ref_code"}).unique(subset=["match_key", "player_id"]))
PIg = PI.join(REF, on=["match_key", "player_id"], how="inner")
G = (PIg.group_by(["match_key", "round", "ref_code"])
     .agg((pl.col("team_code") == 2).sum().alias("n_t"),
          (pl.col("team_code") == 3).sum().alias("n_ct"),
          pl.len().alias("n")))
# A2: side of each group by majority
G = G.with_columns(
    pl.when(pl.col("n_t") > pl.col("n_ct")).then(pl.lit(2))
      .when(pl.col("n_ct") > pl.col("n_t")).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("gside"),
    (pl.max_horizontal(["n_t", "n_ct"]) / pl.col("n")).alias("purity"))
GA = G.filter(pl.col("ref_code") == 2).select(["match_key", "round", pl.col("gside").alias("sideA"), pl.col("n").alias("nA"), pl.col("purity").alias("purA")])
GB = G.filter(pl.col("ref_code") == 3).select(["match_key", "round", pl.col("gside").alias("sideB_obs")])
SA = GA.join(GB, on=["match_key", "round"], how="full", coalesce=True).with_columns(
    pl.col("sideA").fill_null(pl.lit(None, dtype=pl.Int64)))
# A consistency: group B should be on the opposite side of group A
cons = SA.filter(pl.col("sideA").is_not_null() & pl.col("sideB_obs").is_not_null())
print("orientation consistency (B == 5 - A) over rounds with both:", cons.height,
      "| consistent:", cons.filter(pl.col("sideB_obs") == 5 - pl.col("sideA")).height)
print("rounds with unknown sideA (tie):", SA.filter(pl.col("sideA").is_null()).height,
      "| of", SA.height)
print("median min purity A:", SA.select(pl.col("purA").median()).item())
SA = SA.with_columns((5 - pl.col("sideA")).alias("sideB"))

# ---------- score-based round winner under player-derived orientation ----------
U = FE.join(SA.select(["match_key", "round", "sideA", "sideB"]), on=["match_key", "round"], how="inner")
N = U.select(["match_key", (pl.col("round") - 1).alias("round"),
              pl.col("t_score").alias("t_next"), pl.col("ct_score").alias("ct_next"),
              pl.col("sideA").alias("sideA_next")])
P = U.join(N, on=["match_key", "round"], how="inner")
P = P.with_columns(
    pl.when(pl.col("sideA") == 2).then(pl.col("t_score")).otherwise(pl.col("ct_score")).alias("scA0"),
    pl.when(pl.col("sideA") == 2).then(pl.col("ct_score")).otherwise(pl.col("t_score")).alias("scB0"),
    pl.when(pl.col("sideA_next") == 2).then(pl.col("t_next")).otherwise(pl.col("ct_next")).alias("scA1"),
    pl.when(pl.col("sideA_next") == 2).then(pl.col("ct_next")).otherwise(pl.col("t_next")).alias("scB1"),
)
P = P.with_columns((pl.col("scA1") - pl.col("scA0")).alias("dA"), (pl.col("scB1") - pl.col("scB0")).alias("dB"))
P = P.with_columns(
    pl.when((pl.col("dA") == 1) & (pl.col("dB") == 0)).then(pl.lit("A"))
      .when((pl.col("dB") == 1) & (pl.col("dA") == 0)).then(pl.lit("B"))
      .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("won"))
P = P.with_columns(
    pl.when(pl.col("won") == "A").then(pl.col("sideA"))
      .when(pl.col("won") == "B").then(pl.col("sideB"))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("wside_p"))
print("\nscore-winner rows with orientation at r and r+1:", P.height,
      "| unresolved score delta (no unique winner):", P.filter(pl.col("won").is_null()).height)
print("score-delta patterns (dA,dB) top:", P.group_by(["dA", "dB"]).len().sort("len", descending=True).head(6).rows())

X = (P.join(RE, on=["match_key", "round"], how="left")
       .join(BO.select(["match_key", "round", "bomb_side", "n_out"]), on=["match_key", "round"], how="left")
       .join(PL, on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "day", "build_num", "map_name", "mdt"]), on="match_key", how="left"))
X = X.with_columns(
    (pl.col("winner_team_code") == pl.col("wside_p")).alias("w_p_ok"),
    (pl.col("bomb_side") == pl.col("wside_p")).alias("b_p_ok"),
    (pl.col("bomb_side") == pl.col("winner_team_code")).alias("w_b_ok"),
)

# D7 ledger comparison (a comparison only, not an input)
L = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet"))
X = X.join(L.select(["match_key", "round", "ledger_side"]), on=["match_key", "round"], how="left")
both = X.filter(pl.col("ledger_side").is_not_null() & pl.col("wside_p").is_not_null())
print("D7 ledger side vs player-derived winner side: compared", both.height,
      "| agree", both.filter(pl.col("ledger_side") == pl.col("wside_p")).height)

# ---------- swap starts from player orientation ----------
ST = (SA.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("sideA").alias("prev")])
        .join(SA.select(["match_key", "round", "sideA"]), on=["match_key", "round"], how="inner")
        .filter(pl.col("prev").is_not_null() & pl.col("sideA").is_not_null() & (pl.col("prev") != pl.col("sideA")))
        .select(["match_key", "round"]))
starts = {}
for k, r in ST.iter_rows():
    starts.setdefault(k, []).append(r)
for k in starts:
    starts[k].sort()

ALLK = FE.select("match_key").unique()
dp_sw = L.filter(pl.col("swap") == True).group_by("match_key").agg(pl.col("round").sort().alias("dp_after"))
dp_map = {k: sorted(v) for k, v in dp_sw.iter_rows()}


def fmt_of(s):
    if not s:
        return "none"
    s1 = s[0]
    return {9: "f8", 13: "f12"}.get(s1, f"other{s1}")


def pat_str(lst):
    return ",".join(str(x) for x in lst)


pat_rows = []
for k in ALLK["match_key"].to_list():
    s = starts.get(k, [])
    pl_after = [x - 1 for x in s]
    dp_after = dp_map.get(k, [])
    pat_rows.append((k, pat_str(pl_after), pat_str(dp_after), fmt_of(s)))
PM = pl.DataFrame(pat_rows, schema=["match_key", "pl_pat", "dp_pat", "fmt"], orient="row")
PM = PM.join(H.select(["match_key", "build_num", "day", "mdt"]), on="match_key", how="left")

print("\n== A. swap structure: player-derived (pl) vs D7 ledger DP (dp), matches:", PM.height)
print("pl pattern counts (top 12):", PM.group_by("pl_pat").len().sort("len", descending=True).head(12).rows())
print("dp pattern counts (top 12):", PM.group_by("dp_pat").len().sort("len", descending=True).head(12).rows())
print("exact pattern agreement (pl == dp):", PM.filter(pl.col("pl_pat") == pl.col("dp_pat")).height)
print("crosstab pl x dp (top 20):", PM.group_by(["pl_pat", "dp_pat"]).len().sort("len", descending=True).head(20).rows())
print("format (by first swap start, pl):", PM.group_by("fmt").len().sort("len", descending=True).rows())
PM8 = PM.filter(pl.col("dp_pat") == "8")
print("DP '8' matches:", PM8.height, "| of those, pl pattern:", PM8.group_by("pl_pat").len().sort("len", descending=True).rows())
PM12 = PM.filter(pl.col("dp_pat") == "12")
print("DP '12' matches:", PM12.height, "| of those, pl pattern:", PM12.group_by("pl_pat").len().sort("len", descending=True).head(6).rows())
PMN = PM.filter(pl.col("dp_pat") == "")
print("DP '' (no swap) matches:", PMN.height, "| pl pattern:", PMN.group_by("pl_pat").len().sort("len", descending=True).head(6).rows())
PMN2 = PM.filter(pl.col("pl_pat") == "")
print("pl '' (no swap) matches:", PMN2.height, "| DP pattern:", PMN2.group_by("dp_pat").len().sort("len", descending=True).head(6).rows())

# coverage of player orientation over freeze_end rounds
cov_r = FE.join(SA.select(["match_key", "round", "sideA"]), on=["match_key", "round"], how="left")
print("\nfreeze_end rounds without player orientation:", cov_r.filter(pl.col("sideA").is_null()).height,
      "of", cov_r.height, "| matches with any missing round:",
      cov_r.filter(pl.col("sideA").is_null())["match_key"].n_unique())

# ---------- per-round flags: is this round the first of a new side block? ----------
ST_set = {(k, r) for k, r in ST.iter_rows()}
rows_all = X.select(["match_key", "round", "wside_p", "winner_team_code", "bomb_side", "w_p_ok", "w_b_ok",
                     "player_count", "day", "build_num", "map_name"]).to_dicts()
out = []
for d in rows_all:
    s = starts.get(d["match_key"], [])
    r = d["round"]
    last = max([x for x in s if x <= r], default=None)
    d["is_start"] = (d["match_key"], r) in ST_set
    d["fmt"] = fmt_of(s)
    d["offset"] = (r - last + 1) if last is not None else r
    out.append(d)
XR = pl.DataFrame(out, infer_schema_length=None)
Q = XR.filter((pl.col("player_count") == 20) & pl.col("winner_team_code").is_not_null() & pl.col("wside_p").is_not_null())
print("\n== B. winner_team_code vs player-derived winner side (clean rounds, player_count==20)")
print("rows:", Q.height, "| mismatches (w_p_ok False):", Q.filter(pl.col("w_p_ok") == False).height)
print("by is_start (first round of a new block):")
print(Q.group_by("is_start").agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm"),
                                 (pl.col("w_p_ok") == False).mean().alias("rate")).sort("is_start").rows())
print("by fmt x is_start:")
print(Q.group_by(["fmt", "is_start"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm"),
                                          (pl.col("w_p_ok") == False).mean().alias("rate")).sort(["fmt", "is_start"]).rows())
print("by build_num x is_start:")
print(Q.group_by(["build_num", "is_start"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm"),
                                                (pl.col("w_p_ok") == False).mean().alias("rate")).sort(["build_num", "is_start"]).rows())
print("is_start rounds only, build_num x fmt:")
print(Q.filter(pl.col("is_start")).group_by(["build_num", "fmt"]).agg(
    pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm"),
    (pl.col("w_p_ok") == False).mean().alias("rate")).sort(["build_num", "fmt"]).rows())
print("non-start mismatches by round (top 15):")
print(Q.filter((pl.col("w_p_ok") == False) & (~pl.col("is_start"))).group_by("round").len().sort("len", descending=True).head(15).rows())
print("non-start mismatches by fmt x build:")
print(Q.filter((pl.col("w_p_ok") == False) & (~pl.col("is_start"))).group_by(["build_num", "fmt"]).len().sort("len", descending=True).rows())
print("mismatches with offset (rounds into block) distribution (top 15):")
print(Q.filter(pl.col("w_p_ok") == False).group_by("offset").len().sort("len", descending=True).head(15).rows())

print("\n== C. build_num x day (matches) and fmt x build_num (matches)")
print("build_num by day:", H.group_by(["day", "build_num"]).len().sort(["day", "build_num"]).rows())
print("fmt x build_num (matches):", PM.group_by(["fmt", "build_num"]).len().sort(["fmt", "build_num"]).rows())
print("fmt by day (matches):", PM.group_by(["day", "fmt"]).len().sort(["day", "fmt"]).rows())

print("\n== D. day view: is_start rounds, winner mismatch rate (clean), by day and fmt")
print(Q.filter(pl.col("is_start")).group_by(["day", "fmt"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm"),
                                                               (pl.col("w_p_ok") == False).mean().alias("rate")).sort(["day", "fmt"]).rows())
print("non-start mismatches by day (count):", Q.filter((pl.col("w_p_ok") == False) & (~pl.col("is_start"))).group_by("day").len().sort("day").rows())

# ---------- bomb-based check, exactly as the defect is defined (no swap rule involved) ----------
BR = (RE.filter(pl.col("player_count") == 20)
        .join(BO.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="inner")
        .join(H.select(["match_key", "day", "build_num"]), on="match_key", how="left")
        .join(ST.with_columns(pl.lit(True).alias("is_start_b")), on=["match_key", "round"], how="left")
        .with_columns(pl.col("is_start_b").fill_null(False)))
BR = BR.with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
BR = BR.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25")).otherwise(pl.lit("other")).alias("rbin"))
BR = BR.with_columns(pl.when(pl.col("day") >= "12").then(pl.lit("02-12..25")).otherwise(pl.lit("02-01..07")).alias("win"))
print("\nbomb-defect rows (player_count 20, bomb outcome present):", BR.height, "| mismatches:", BR.filter(pl.col("mm")).height)
print("by round bin x window (n, mismatches):")
print(BR.group_by(["win", "rbin"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["win", "rbin"]).rows())
print("round-13 rows by day (n, mm):", BR.filter(pl.col("round") == 13).group_by("day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("day").rows())
print("round-13 mismatch by is_start (first round of new block):")
print(BR.filter(pl.col("round") == 13).group_by("is_start_b").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("is_start_b").rows())
print("round-13 mismatch by build_num x is_start:")
print(BR.filter(pl.col("round") == 13).group_by(["build_num", "is_start_b"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["build_num", "is_start_b"]).rows())
print("all-round mismatches by is_start (n, mm):", BR.group_by("is_start_b").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("is_start_b").rows())
print("all-round mismatches by build_num x is_start:", BR.group_by(["build_num", "is_start_b"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["build_num", "is_start_b"]).rows())
print("mismatches (all rounds) by round number (top 15):", BR.filter(pl.col("mm")).group_by("round").len().sort("len", descending=True).head(15).rows())
