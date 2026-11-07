# verify_A2_1 independent check 2 (verifier "ind"): phase-transition alignment and win_reason adjudication.
# ASSUMPTION F: win_reason_message is side-specific (e.g. a bomb-explosion reason implies T). Its implied side is
#   calibrated only on rounds outside KS where score delta and round_end winner agree 100% (both builds).
# ASSUMPTION G: phase transition round k = first freeze_end round whose phase differs from the previous freeze_end round.
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
HDR = os.path.join(BASE, "header", "dataframe_00000")
KS = [9, 13, 25, 27, 28]
KG = pl.when(pl.col("round").is_in(KS)).then(pl.col("round").cast(pl.Utf8)).otherwise(pl.lit("other"))


def files(ch):
    fs = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}{SUF}", "dataframe_*")))
    if not fs:
        raise SystemExit("no files for " + ch)
    return fs


def section(t):
    print("\n==== " + t, flush=True)


hdr = pl.read_parquet(HDR, columns=["key", "build_num"]).rename({"key": "match_key"}).with_columns(
    pl.col("match_key").str.split("/").list.get(3).alias("day"))
hdr_s = hdr.select(["match_key", "build_num", "day"])

fe_raw = (pl.scan_parquet(files("round_state"))
          .filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "t_score", "ct_score", "phase"])
          .with_columns(pl.col("round").cast(pl.Int64))
          .collect())
fe = fe_raw.unique(["match_key", "round"], keep="first", maintain_order=True).sort(["match_key", "round"])
fe = fe.with_columns(pl.col("phase").shift(1).over("match_key").alias("phase_prev"))
nx = fe.with_columns(pl.col("round").shift(-1).over("match_key").alias("r_next"),
                     pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                     pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"))
pr = (nx.filter(pl.col("r_next") == pl.col("round") + 1)
      .with_columns((pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT"))
      .with_columns(pl.when(pl.col("dT") == 1).then(pl.lit(2)).when(pl.col("dCT") == 1).then(pl.lit(3))
                    .otherwise(pl.lit(0)).alias("w_score")))

re_raw = (pl.scan_parquet(files("round_end"))
          .select(["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message"])
          .with_columns(pl.col("round").cast(pl.Int64))
          .collect())
W = (re_raw.unique(["match_key", "round"], keep="first", maintain_order=True)
     .with_columns(pl.col("win_reason_message").cast(pl.Utf8).fill_null("NULL"))
     .select(["match_key", "round", pl.col("winner_team_code").alias("w_end"), "win_reason_code", "win_reason_message"]))

# ---------------------------------------------------------------- 1. phase transitions by build
section("1. phase transitions (first freeze_end round whose phase differs from previous) by build")
tr = (fe.filter(pl.col("phase_prev").is_not_null() & (pl.col("phase_prev") != pl.col("phase")))
      .join(hdr_s, on="match_key", how="left")
      .group_by(["build_num", "round"]).agg(pl.len().alias("matches_with_transition"))
      .filter(pl.col("round") <= 30).sort(["build_num", "round"]))
print("(build, transition round, matches):", tr.rows(), flush=True)
print("phase values at rounds 24-30 by build:",
      fe.join(hdr_s, on="match_key", how="left").filter(pl.col("round").is_between(24, 30))
      .group_by(["build_num", "round", "phase"]).len().sort(["build_num", "round", "phase"]).rows(), flush=True)

pairs = (pr.join(W, on=["match_key", "round"], how="left")
         .join(hdr_s, on="match_key", how="left")
         .with_columns((pl.col("w_end") != pl.col("w_score")).cast(pl.Int64).alias("dis"),
                       (pl.col("phase") != pl.col("phase_prev")).fill_null(False).alias("phase_change"),
                       KG.alias("kg"))
         .filter(pl.col("w_score") != 0))
print("pairs total", pairs.height, flush=True)
section("1b. disagreement (score vs round_end) by build x phase_change for pairs")
print("(build, phase_change, n, dis):", pairs.group_by(["build_num", "phase_change"])
      .agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort(["build_num", "phase_change"]).rows(), flush=True)
print("(build, k in KS, phase_change, n, dis):", pairs.filter(pl.col("kg") != "other")
      .group_by(["build_num", "kg", "phase_change"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"))
      .sort(["build_num", "kg", "phase_change"]).rows(), flush=True)
print("(build, phase_change, n, dis) for k not in KS (other):", pairs.filter(pl.col("kg") == "other")
      .group_by(["build_num", "phase_change"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"))
      .sort(["build_num", "phase_change"]).rows(), flush=True)

# ---------------------------------------------------------------- 2. win_reason calibration
section("2. win_reason_message calibration (rounds k not in KS where score delta and round_end agree)")
calib_src = pairs.filter((pl.col("kg") == "other") & (pl.col("dis") == 0))
cal = (calib_src.group_by("win_reason_message")
       .agg((pl.col("w_end") == 2).sum().alias("n2"), (pl.col("w_end") == 3).sum().alias("n3"))
       .sort("win_reason_message").rows())
mapping = {}
for m, n2, n3 in cal:
    tot = n2 + n3
    side = 2 if n2 >= n3 else 3
    purity = max(n2, n3) / tot if tot else 0.0
    mapping[m] = side
    print(f"   reason={m!r:40s} n_T(2)={n2:6d} n_CT(3)={n3:6d} implied_side={side} purity={purity:.4f}", flush=True)
mdf = pl.DataFrame({"win_reason_message": list(mapping.keys()), "implied_side": list(mapping.values())},
                   schema={"win_reason_message": pl.Utf8, "implied_side": pl.Int64})

# ---------------------------------------------------------------- 3. adjudicate with reason
section("3. implied side from win_reason vs round_end winner vs score delta")
allr = (W.join(hdr_s, on="match_key", how="left")
        .join(mdf, on="win_reason_message", how="left")
        .with_columns(KG.alias("kg"),
                      (pl.col("implied_side") != pl.col("w_end")).cast(pl.Int64).alias("reason_vs_end_mis"))
        .filter(pl.col("implied_side").is_not_null() & pl.col("build_num").is_not_null()))
print("all round_end rows with implied side (build, kg, n, reason_vs_roundend_mismatch):",
      allr.group_by(["build_num", "kg"]).agg(pl.len().alias("n"), pl.col("reason_vs_end_mis").sum().alias("mis"))
      .sort(["build_num", "kg"]).rows(), flush=True)
pa = (pairs.join(mdf, on="win_reason_message", how="left")
      .filter(pl.col("implied_side").is_not_null())
      .with_columns((pl.col("implied_side") == pl.col("w_score")).cast(pl.Int64).alias("reason_agrees_score"),
                    (pl.col("implied_side") == pl.col("w_end")).cast(pl.Int64).alias("reason_agrees_end")))
print("score-pairs with implied side (build, kg, n, reason=score, reason=roundend):",
      pa.group_by(["build_num", "kg"]).agg(pl.len().alias("n"), pl.col("reason_agrees_score").sum().alias("rs"),
                                           pl.col("reason_agrees_end").sum().alias("re")).sort(["build_num", "kg"]).rows(),
      flush=True)
print("reason messages on 9961 k=9/13/25/27/28 disagreement rounds (build, kg, message, w_score, w_end, implied, n):",
      pairs.filter((pl.col("build_num") == 9961) & (pl.col("dis") == 1) & (pl.col("kg") != "other"))
      .group_by(["kg", "win_reason_message", "w_score", "w_end"]).len()
      .sort(["kg", "win_reason_message"]).rows(), flush=True)
print("reason messages on 9842 k=27 disagreement rounds (kg, message, w_score, w_end, n):",
      pairs.filter((pl.col("build_num") == 9842) & (pl.col("dis") == 1))
      .group_by(["kg", "win_reason_message", "w_score", "w_end"]).len().sort(["kg", "win_reason_message"]).rows(), flush=True)

# ---------------------------------------------------------------- 4. k=9 detail
section("4. 9961 k=9 disagreements: reason vs score vs round_end")
k9 = pairs.filter((pl.col("build_num") == 9961) & (pl.col("round") == 9)).join(mdf, on="win_reason_message", how="left")
print("(dis, phase_change, implied, n):", k9.group_by(["dis", "phase_change", "implied_side"]).len().sort(["dis", "phase_change", "implied_side"]).rows(), flush=True)
print("DONE", flush=True)
