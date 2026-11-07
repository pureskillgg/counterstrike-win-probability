# verify_A2_1 independent check 3 (verifier "ind"): do bomb outcome events fall inside the round window they are attributed to?
# ASSUMPTION H: a bomb outcome belongs to round k if its tick lies in [freeze_end(k).tick, round_end(k).tick].
# ASSUMPTION I: bomb_exploded -> T (2); bomb_defused -> CT (3); only rounds with exactly one outcome row are used.
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


section("0. bomb_state and round_state schemas")
print("bomb_state:", list(pl.read_parquet_schema(files("bomb_state")[0]).items()), flush=True)
print("round_state:", list(pl.read_parquet_schema(files("round_state")[0]).items()), flush=True)

hdr = pl.read_parquet(HDR, columns=["key", "build_num"]).rename({"key": "match_key"})

bo = (pl.scan_parquet(files("bomb_state"))
      .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "tick", "event_type"])
      .with_columns(pl.col("round").cast(pl.Int64))
      .collect())
cnt = bo.group_by(["match_key", "round"]).len()
print("bomb outcome rows", bo.height, " (match,round) groups", cnt.height, " groups with >1 row", cnt.filter(pl.col("len") > 1).height, flush=True)
bo1 = (bo.unique(["match_key", "round"], keep="first", maintain_order=True)
       .with_columns(pl.when(pl.col("event_type") == "bomb_defused").then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
       .select(["match_key", "round", "tick", "bomb_side", "event_type"]))

fe_raw = (pl.scan_parquet(files("round_state"))
          .filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "tick"])
          .with_columns(pl.col("round").cast(pl.Int64))
          .collect())
fe = fe_raw.unique(["match_key", "round"], keep="first", maintain_order=True).rename({"tick": "fe_tick"})

re_raw = (pl.scan_parquet(files("round_end"))
          .select(["match_key", "round", "tick", "winner_team_code", "win_reason_message"])
          .with_columns(pl.col("round").cast(pl.Int64))
          .collect())
re_u = (re_raw.unique(["match_key", "round"], keep="first", maintain_order=True)
        .with_columns(pl.col("win_reason_message").cast(pl.Utf8).fill_null("NULL"))
        .rename({"tick": "re_tick", "winner_team_code": "w_end"}))

fe_prev = fe.with_columns((pl.col("round") + 1).alias("round")).rename({"fe_tick": "fe_prev_tick"})
re_prev = re_u.select(["match_key", "round", "re_tick"]).with_columns((pl.col("round") + 1).alias("round")).rename({"re_tick": "re_prev_tick"})

j = (bo1.join(fe, on=["match_key", "round"], how="left")
     .join(re_u, on=["match_key", "round"], how="left")
     .join(fe_prev, on=["match_key", "round"], how="left")
     .join(re_prev, on=["match_key", "round"], how="left")
     .join(hdr, on="match_key", how="left")
     .with_columns(KG.alias("kg"),
                   ((pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick"))).alias("in_own_window"),
                   ((pl.col("tick") >= pl.col("fe_prev_tick")) & (pl.col("tick") <= pl.col("re_prev_tick"))).alias("in_prev_window"),
                   (pl.col("bomb_side") != pl.col("w_end")).cast(pl.Int64).alias("bomb_vs_end_mis")))
j = j.filter(pl.col("w_end").is_not_null() & pl.col("fe_tick").is_not_null() & pl.col("re_tick").is_not_null())

section("1. bomb event tick vs own round window and previous round window (by build x kg)")
print("(build, kg, n, in_own_window, in_prev_window, bomb_vs_roundend_mismatch):",
      j.group_by(["build_num", "kg"]).agg(pl.len().alias("n"), pl.col("in_own_window").sum().alias("in_own"),
                                          pl.col("in_prev_window").sum().alias("in_prev"),
                                          pl.col("bomb_vs_end_mis").sum().alias("mis"))
      .sort(["build_num", "kg"]).rows(), flush=True)

section("2. bomb side x win_reason_message x round_end winner (by build, k=13 and k=other)")
print("(build, kg, bomb_side, event_type, win_reason_message, w_end, n):",
      j.filter(pl.col("kg").is_in(["13", "other"]))
      .group_by(["build_num", "kg", "bomb_side", "event_type", "win_reason_message", "w_end"]).len()
      .sort(["build_num", "kg", "bomb_side", "win_reason_message"]).rows(), flush=True)

section("3. 9961 k=13 bomb-attributed rounds: tick offsets relative to own window")
x = j.filter((pl.col("build_num") == 9961) & (pl.col("kg") == "13"))
print("n:", x.height, " median (tick - fe_tick):", x.select((pl.col("tick") - pl.col("fe_tick")).median()).item(),
      " median (re_tick - tick):", x.select((pl.col("re_tick") - pl.col("tick")).median()).item(), flush=True)
print("min/max tick - fe_tick:", x.select((pl.col("tick") - pl.col("fe_tick")).min()).item(),
      x.select((pl.col("tick") - pl.col("fe_tick")).max()).item(), flush=True)
print("DONE", flush=True)
