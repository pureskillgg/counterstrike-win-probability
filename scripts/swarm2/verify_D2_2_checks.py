"""Verifier D2-2, second pass: truth = side-attached score + bomb only (no death-based reference).

Same raw-tome build as verify_D2_2_recompute.py, then:
  T_SB  = team-tracked score winner S, else in-window bomb outcome B_win (no death reference)
  checks: claim counts with T_SB; static vs lag-1 identities; boundary-only flips;
          round-label shift alternative; swap-transition team tracking; death reference contamination;
          match-level flags; map and period confound.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_2_checks.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(36)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
PAD = 40  # ASSUMPTION: bomb events allowed up to 40 ticks past round_end (as D2_truth.py)


def files(ch):
    if ch == "header":
        return sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
    fd = sorted(glob.glob(str(BASE / f"basic_{ch}.*")))[0]
    return sorted(glob.glob(fd + "/dataframe_*"))


def scan(ch, cols):
    return pl.scan_parquet(files(ch)).select(cols)


# ------------------------------------------------------------------ build (same logic as recompute script)
re_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
RE = (re_raw.filter(pl.col("player_count") == 20).sort("tick").unique(["match_key", "round"], keep="first")
      .select(["match_key", "round", pl.col("tick").alias("t_end"), pl.col("winner_team_code").alias("R")]))
hdr = pl.scan_parquet(files("header")).select(["key", "map_name", "match_date"]).collect()
MAP = hdr.select([pl.col("key").alias("match_key"), "map_name"]).unique("match_key", keep="first")

rs = (scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").collect()
      .sort("tick").unique(["match_key", "round"], keep="first")
      .select(["match_key", "round", pl.col("tick").alias("t_fe"),
               pl.col("t_score").alias("tS"), pl.col("ct_score").alias("cS")]))

pi = (scan("player_info", ["match_key", "round", "player_id", "team_code"]).collect()
      .unique(["match_key", "round", "player_id"], keep="first"))
cnt = pi.group_by(["match_key", "round"]).agg(
    pl.len().alias("n"), (pl.col("team_code") == 2).sum().alias("n2"), (pl.col("team_code") == 3).sum().alias("n3"))
valid = cnt.filter((pl.col("n") == 10) & (pl.col("n2") == 5) & (pl.col("n3") == 5))
first_valid = valid.group_by("match_key").agg(pl.col("round").min().alias("F"))
xmem = (pi.join(first_valid, on="match_key").filter(pl.col("round") == pl.col("F"))
        .filter(pl.col("team_code") == 2).select(["match_key", "player_id"]).with_columns(pl.lit(True).alias("isX")))
pi2 = pi.join(xmem, on=["match_key", "player_id"], how="left").with_columns(pl.col("isX").fill_null(False))
sxg = pi2.group_by(["match_key", "round"]).agg(
    (pl.col("isX") & (pl.col("team_code") == 2)).sum().alias("x2"),
    (pl.col("isX") & (pl.col("team_code") == 3)).sum().alias("x3"))
sxg = sxg.with_columns(
    pl.when((pl.col("x2") >= 3) & (pl.col("x3") == 0)).then(pl.lit(2))
    .when((pl.col("x3") >= 3) & (pl.col("x2") == 0)).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("sx"))
sx_tbl = sxg.select(["match_key", "round", "sx"])
sx_prev_tbl = sx_tbl.with_columns((pl.col("round") + 1).alias("round")).rename({"sx": "sx_prev"})
sx_nx_tbl = sx_tbl.with_columns((pl.col("round") - 1).alias("round")).rename({"sx": "sx_nx"})

dth = (scan("player_death", ["match_key", "round", "player_id", "player_team_code"]).collect()
       .unique(["match_key", "round", "player_id"], keep="first"))
dsum = (dth.filter(pl.col("player_team_code").is_in([2, 3])).group_by(["match_key", "round"]).agg(
    (pl.col("player_team_code") == 2).sum().alias("t_dead"),
    (pl.col("player_team_code") == 3).sum().alias("ct_dead")))
dsum = dsum.with_columns(
    pl.when((pl.col("ct_dead") == 5) & (pl.col("t_dead") < 5)).then(pl.lit(2))
    .when((pl.col("t_dead") == 5) & (pl.col("ct_dead") < 5)).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("D")).select(["match_key", "round", "D"])

bs = (scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
      .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_exploded", "bomb_defused"])).collect())
bo = bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
bw = (bo.join(RE.select(["match_key", "round", "t_end"]), on=["match_key", "round"], how="inner")
      .join(rs.select(["match_key", "round", "t_fe"]), on=["match_key", "round"], how="inner")
      .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + PAD)))
bwin = bw.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("dfs"))
bwin = bwin.with_columns(
    pl.when(pl.col("exp") & ~pl.col("dfs")).then(pl.lit(2))
    .when(pl.col("dfs") & ~pl.col("exp")).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("B_win")).select(["match_key", "round", "B_win"])
planter = (bs.filter(pl.col("event_type") == "bomb_planted").sort("tick")
           .unique(["match_key", "round"], keep="first").select(["match_key", "round", pl.col("player_team_code").alias("P")]))

fe_nx = rs.select(["match_key", (pl.col("round") - 1).alias("round"),
                   pl.col("tS").alias("tS_nx"), pl.col("cS").alias("cS_nx")])
m = (RE.join(rs.select(["match_key", "round", "tS", "cS"]), on=["match_key", "round"], how="left")
     .join(fe_nx, on=["match_key", "round"], how="left")
     .join(sx_prev_tbl, on=["match_key", "round"], how="left")
     .join(sx_nx_tbl, on=["match_key", "round"], how="left")
     .join(sxg.select(["match_key", "round", "sx"]), on=["match_key", "round"], how="left")
     .join(dsum, on=["match_key", "round"], how="left")
     .join(bwin, on=["match_key", "round"], how="left")
     .join(planter, on=["match_key", "round"], how="left")
     .join(MAP, on="match_key", how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
m = m.with_columns(
    pl.when(pl.col("sx") == 2).then(pl.col("tS")).otherwise(pl.col("cS")).alias("x0"),
    pl.when(pl.col("sx") == 3).then(pl.col("tS")).otherwise(pl.col("cS")).alias("y0"),
    pl.when(pl.col("sx_nx") == 2).then(pl.col("tS_nx")).otherwise(pl.col("cS_nx")).alias("x1"),
    pl.when(pl.col("sx_nx") == 3).then(pl.col("tS_nx")).otherwise(pl.col("cS_nx")).alias("y1"))
m = m.with_columns(
    (pl.col("x1") - pl.col("x0")).alias("dX"), (pl.col("y1") - pl.col("y0")).alias("dY"))
m = m.with_columns(
    pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
    .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("S"),
    pl.when((pl.col("tS_nx") - pl.col("tS")) == 1).then(pl.lit(2))
    .when((pl.col("cS_nx") - pl.col("cS")) == 1).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("S_naive"),
    (pl.col("sx") != pl.col("sx_prev")).alias("swap"),
    (pl.col("sx_nx") != pl.col("sx")).alias("swap_next"),
    pl.when(pl.col("round").is_between(1, 12)).then(pl.lit("r01_12"))
    .when(pl.col("round") == 13).then(pl.lit("r13"))
    .when(pl.col("round").is_between(14, 24)).then(pl.lit("r14_24"))
    .when(pl.col("round") == 25).then(pl.lit("r25"))
    .when(pl.col("round") == 28).then(pl.lit("r28"))
    .otherwise(pl.lit("r_other")).alias("rbin"))
m = m.with_columns(
    pl.coalesce(["S", "B_win"]).alias("T_SB"),
    pl.coalesce(["S", "B_win", "D"]).alias("T_D"))
m = m.with_columns(
    (pl.col("R") != pl.col("T_SB")).alias("fSB"),
    (pl.col("R") != pl.col("T_D")).alias("fD_all"),
    (pl.col("R") != pl.col("B_win")).alias("fB"),
    (pl.col("R") != pl.col("S")).alias("fS"))
print(f"rows {m.height}; T_SB known {m.filter(pl.col('T_SB').is_not_null()).height}; "
      f"S known {m.filter(pl.col('S').is_not_null()).height}; B_win known {m.filter(pl.col('B_win').is_not_null()).height}")

# shifted truth tables (for round-label alternatives)
TT = m.select(["match_key", "round", "T_SB"])
T_prev = TT.with_columns((pl.col("round") + 1).alias("round")).rename({"T_SB": "T_prev"})  # key r, value T(r-1)
T_next = TT.with_columns((pl.col("round") - 1).alias("round")).rename({"T_SB": "T_next"})  # key r, value T(r+1)
m = m.join(T_prev, on=["match_key", "round"], how="left").join(T_next, on=["match_key", "round"], how="left")

# ------------------------------------------------------------------ A. claim counts with T_SB
print("\n== A. recorded R vs T_SB (score then bomb; no death reference), by round group")
print(m.group_by("rbin").agg(
    pl.col("T_SB").is_not_null().sum().alias("n_T"), pl.col("fSB").sum().alias("R!=T_SB"),
    pl.col("fSB").mean().round(4).alias("rate")).sort("rbin"))
r13 = m.filter((pl.col("round") == 13) & pl.col("T_SB").is_not_null())
print("round 13 by day (T_SB):")
print(r13.group_by("day").agg(pl.len().alias("n"), pl.col("fSB").sum().alias("flip"),
                              pl.col("fSB").mean().round(3).alias("rate")).sort("day").rows())

# ------------------------------------------------------------------ B. structure with T_SB (claim readings)
print("\n== B. static identity vs lag-1 side on boundary rounds (truth T_SB)")
for rnd in [13, 25, 28]:
    sub = m.filter((pl.col("round") == rnd) & pl.col("sx").is_not_null() & pl.col("sx_prev").is_not_null()
                   & pl.col("T_SB").is_not_null())
    sub = sub.with_columns(
        pl.when(pl.col("T_SB") == pl.col("sx")).then(pl.lit(2)).otherwise(pl.lit(3)).alias("static_pred"),
        pl.when(pl.col("T_SB") == pl.col("sx")).then(pl.col("sx_prev")).otherwise(5 - pl.col("sx_prev")).alias("lag_pred"))
    sub = sub.with_columns(
        (pl.col("R") == pl.col("static_pred")).alias("eqS"),
        (pl.col("R") == pl.col("lag_pred")).alias("eqL"))
    ft = sub.filter(pl.col("fSB"))
    un = sub.filter(~pl.col("fSB"))
    print(f"-- round {rnd}: rows {sub.height}; (sx_prev,sx) {sub.group_by(['sx_prev','sx']).len().sort(['sx_prev','sx']).rows()}")
    print(f"   flipped {ft.height}: R==static {ft.filter(pl.col('eqS')).height}, R==lag1 {ft.filter(pl.col('eqL')).height}")
    print(f"   unflipped {un.height}: R==static {un.filter(pl.col('eqS')).height}, R==lag1 {un.filter(pl.col('eqL')).height}")
    print(f"   static_pred == 5-lag_pred in {sub.filter(pl.col('static_pred') == 5 - pl.col('lag_pred')).height} rows;"
          f" static_pred == lag_pred in {sub.filter(pl.col('static_pred') == pl.col('lag_pred')).height} rows")
    print(f"   flip == (R==static) mismatches {sub.filter(pl.col('eqS') != pl.col('fSB')).height}; "
          f"flip == (R==lag1) mismatches {sub.filter(pl.col('eqL') != pl.col('fSB')).height}")
    print(f"   R == 5 - T_SB (complement) in {sub.filter(pl.col('R') == 5 - pl.col('T_SB')).height} of {sub.height}")

# ------------------------------------------------------------------ C. round-label shift alternative
print("\n== C. round-label shift alternative: is R(r) explained by T(r-1) or T(r+1) instead of 5-T(r)?")
for rnd in [13, 25, 28]:
    sub = m.filter((pl.col("round") == rnd) & pl.col("T_SB").is_not_null())
    print(f"round {rnd}: rows {sub.height}; R==T_prev(r-1) {sub.filter(pl.col('R') == pl.col('T_prev')).height}; "
          f"R==T_next(r+1) {sub.filter(pl.col('R') == pl.col('T_next')).height}; "
          f"R==5-T(r) {sub.filter(pl.col('R') == 5 - pl.col('T_SB')).height}; R==T(r) {sub.filter(pl.col('R') == pl.col('T_SB')).height}")
sub = m.filter((pl.col("round") == 13) & pl.col("T_SB").is_not_null() & pl.col("fSB") & pl.col("T_prev").is_not_null())
print(f"flipped round-13 rows with T(12) known: {sub.height}; R==T(12) {sub.filter(pl.col('R') == pl.col('T_prev')).height}")

# ------------------------------------------------------------------ D. swap transitions: team tracking vs naive
print("\n== D. transition r -> r+1 with a team swap (sx_nx != sx): S (team-tracked) vs naive side delta vs bomb")
tr = m.filter(pl.col("swap_next") & pl.col("S").is_not_null())
print("swap_next rows with S:", tr.height, "by round:", tr.group_by("round").len().sort("round").rows())
trb = tr.filter(pl.col("B_win").is_not_null())
print(f"  with B_win: {trb.height}; S == B_win {trb.filter(pl.col('S') == pl.col('B_win')).height}; "
      f"S_naive == B_win {trb.filter(pl.col('S_naive') == pl.col('B_win')).height}; "
      f"S_naive != S {tr.filter(pl.col('S_naive') != pl.col('S')).height}")
print("  R == S at swap_next rows:", tr.filter(pl.col("R") == pl.col("S")).height, "of", tr.height)

# ------------------------------------------------------------------ E. death reference contamination
print("\n== E. death-based winner (D) vs bomb outcome, split by whether a bomb was planted")
dd = m.filter(pl.col("D").is_not_null() & pl.col("B_win").is_not_null())
print("  rows with D and B_win:", dd.height, " D == B_win:", dd.filter(pl.col("D") == pl.col("B_win")).height)
dp = dd.filter(pl.col("P").is_not_null())
print("  planted rounds:", dp.height, " D == B_win:", dp.filter(pl.col("D") == pl.col("B_win")).height)
dn = m.filter(pl.col("D").is_not_null() & pl.col("S").is_not_null() & pl.col("P").is_null())
print("  unplanted rounds with D and S:", dn.height, " D == S:", dn.filter(pl.col("D") == pl.col("S")).height)
print("  flips counted only by D (T_SB unknown, D known), by round:",
      m.filter(pl.col("T_SB").is_null() & pl.col("D").is_not_null() & (pl.col("R") != pl.col("D")))
      .group_by("rbin").len().sort("rbin").rows())

# ------------------------------------------------------------------ F. match-level flags
print("\n== F. match-level flag at round 13 (T_SB), and flips by round inside flagged vs unflagged matches")
mf = (m.filter((pl.col("round") == 13) & pl.col("T_SB").is_not_null())
      .group_by("match_key").agg(pl.col("fSB").any().alias("mflag13"), pl.col("map_name").first().alias("map_name"),
                                 pl.col("day").first().alias("day")))
print("matches with T_SB at 13:", mf.height, " flagged:", mf.filter(pl.col("mflag13")).height)
mm = m.join(mf.select(["match_key", "mflag13"]), on="match_key", how="inner").filter(pl.col("T_SB").is_not_null())
print("flips per round, flagged matches (rows, flips):")
print(mm.filter(pl.col("mflag13")).group_by("round").agg(pl.len().alias("n"), pl.col("fSB").sum().alias("flips"))
      .sort("round").rows())
print("flips per round, unflagged matches (rows, flips):")
print(mm.filter(~pl.col("mflag13")).group_by("round").agg(pl.len().alias("n"), pl.col("fSB").sum().alias("flips"))
      .sort("round").rows())

pv = (m.filter(pl.col("round").is_in([13, 25, 28]) & pl.col("T_SB").is_not_null())
      .select(["match_key", "round", "fSB"]).pivot(on="round", index="match_key", values="fSB", aggregate_function="first"))
print("per-match flags at 13/25/28 (T_SB):")
print(pv.group_by(["13", "25", "28"]).len().sort(["13", "25", "28"]))
odd = pv.filter((pl.col("13") != True) & (pl.col("25") != True) & (pl.col("28") == True))
print("matches flagged at 28 only (not 13 or 25):", odd.height)
if odd.height:
    ids = odd["match_key"].to_list()
    det = m.filter(pl.col("match_key").is_in(ids) & pl.col("round").is_in([12, 13, 24, 25, 27, 28])).select(
        ["match_key", "round", "sx", "sx_prev", "R", "B_win", "S", "D", "T_SB"]).sort(["match_key", "round"])
    print(det)

# ------------------------------------------------------------------ G. map and period confound
print("\n== G. round-13 match flag by map_name and period (T_SB)")
mf = mf.with_columns(pl.when(pl.col("day").cast(pl.Int64) <= 7).then(pl.lit("early_01-07")).otherwise(pl.lit("late_12-25")).alias("period"))
print(mf.group_by("map_name").agg(pl.len().alias("n"), pl.col("mflag13").sum().alias("flagged"),
                                  pl.col("mflag13").mean().round(3).alias("rate")).sort("n", descending=True))
print(mf.group_by(["period", "map_name"]).agg(pl.len().alias("n"), pl.col("mflag13").sum().alias("flagged"),
                                              pl.col("mflag13").mean().round(3).alias("rate")).sort(["period", "map_name"]))
