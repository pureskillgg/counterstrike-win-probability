"""verify_D7_1_main.py -- independent recomputation of the D7-1 claim from the raw 2024 tome channels.

Reads (column-projected, lazy filter only): header, round_end, bomb_state, and the 2026 header (control).
Writes nothing. Output goes to logs\\swarm2\\verify_D7_1.log via the shell.

ASSUMPTIONS (printed in the log as ASSUMPTION where used):
  A1 match_date is UTC as recorded (trailing 'Z'); no timezone conversion is applied.
  A2 player_count == 20 (full-match convention, as in scripts/swarm/e2_side_check_r13.py and D7_1).
  A3 outcome side: bomb_defused -> 3 (CT), bomb_exploded -> 2 (T); mismatch = outcome side != winner_team_code.
  A4 round 13 = first round of the second half (MR12 halftime side swap); round 25 = first OT round.
  A5 the claim's cut 2024-02-07 02:39:55 UTC is used for the pre/post split; it is also re-derived from the data.
"""
import glob
import sys
import datetime as dt
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
B26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
CUT_CLAIM = dt.datetime(2024, 2, 7, 2, 39, 55)
PATH_CUT = dt.date(2024, 2, 12)
ONSET_LO = dt.datetime(2024, 2, 7, 0, 24, 22)
ONSET_HI = dt.datetime(2024, 2, 7, 1, 1, 18)


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def chan(ch):
    return files(BASE / f"basic_{ch}{SUF}")


def hdr(title):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)


print("polars", pl.__version__)
print("ASSUMPTION A1: match_date is UTC as recorded (trailing Z).")
print("ASSUMPTION A2: player_count == 20 for outcome rows.")
print("ASSUMPTION A3: bomb_defused -> side 3 (CT), bomb_exploded -> side 2 (T); mismatch = side != winner_team_code.")
print("ASSUMPTION A4: round 13 = first round of second half; round 25 = first OT round.")
print("ASSUMPTION A5: claim cut 2024-02-07 02:39:55 UTC used for pre/post; also re-derived below.")

# ---------------------------------------------------------------- A. header
hdr("A. header (one row per match): match_date vs csds path day")
H = (pl.read_parquet(files(BASE / "header"),
                     columns=["key", "match_id", "match_date", "build_num", "map_name", "tick_rate"])
     .with_columns(
         pl.col("match_date").str.ends_with("Z").alias("has_z"),
         pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("md_ts"),
         pl.col("key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1).str.replace_all("/", "-")
           .str.to_date("%Y-%m-%d").alias("path_date"),
     ))
H = H.with_columns(pl.col("md_ts").dt.date().alias("md_date"))
print("rows", H.height, "| unique key", H["key"].n_unique(), "| unique match_id", H["match_id"].n_unique())
print("all match_date end with Z:", bool(H["has_z"].all()), "| null md_ts:", H["md_ts"].null_count(),
      "| null path_date:", H["path_date"].null_count())
print("match_date range (as recorded):", H["md_ts"].min(), "..", H["md_ts"].max())
print("path_date range:", H["path_date"].min(), "..", H["path_date"].max())
print("build_num counts:", H["build_num"].value_counts().sort("build_num").rows())
print("tick_rate:", H["tick_rate"].unique().to_list(), "| map counts:", H["map_name"].value_counts().sort("count", descending=True).rows()[:9])
DIFF = H.with_columns((pl.col("path_date").cast(pl.Int32) - pl.col("md_date").cast(pl.Int32)).alias("diff_days"))
nd = DIFF.filter(pl.col("diff_days") != 0).height
print(f"matches where path day != match_date UTC day: {nd} of {H.height}   (claim: 2,943 of 4,435)")
print("diff_days (path - match day), value counts sorted by diff:")
print(DIFF.group_by("diff_days").len().sort("diff_days").rows())
print("path day later than match day:", DIFF.filter(pl.col("diff_days") > 0).height,
      "| earlier:", DIFF.filter(pl.col("diff_days") < 0).height)

hdr("A2. control: 2026 tome, same comparison (header only)")
try:
    H26 = pl.read_parquet(files(B26 / "header"), columns=["key", "match_date"]).with_columns(
        pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).dt.date().alias("md_date"),
        pl.col("key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1).str.replace_all("/", "-")
          .str.to_date("%Y-%m-%d").alias("path_date"))
    print("2026 header rows", H26.height, "| path day != match day:",
          H26.filter(pl.col("path_date") != pl.col("md_date")).height)
    print("2026 key sample:", H26["key"].head(3).to_list())
except Exception as e:  # report, do not hide
    print("2026 control not run:", repr(e))

# ---------------------------------------------------------------- B. round_end, bomb outcomes
hdr("B. round_end (winner) and bomb outcome events (2024 tome)")
RE0 = (pl.scan_parquet(chan("round_end"))
       .select(["match_key", "round", "tick", "winner_team_code", "win_reason_message", "player_count"])
       .collect())
print("round_end rows", RE0.height, "| unique (match,round)", RE0.select(["match_key", "round"]).unique().height)
dup = RE0.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("duplicated (match,round) keys:", dup.height)
if dup.height:
    print(RE0.join(dup.select(["match_key", "round"]), on=["match_key", "round"]))
RE = (RE0.unique(subset=["match_key", "round"], keep="first", maintain_order=True)
      .rename({"tick": "end_tick"}))
print("player_count dist (round_end):", RE.group_by("player_count").len().sort("player_count").rows())
print("winner_team_code dist:", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())

BS = (pl.scan_parquet(chan("bomb_state"))
      .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "tick", "event_type", "player_team_code"])
      .collect())
print("bomb event counts:", BS.group_by("event_type").len().sort("event_type").rows())
OUT0 = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg(pl.col("event_type").eq("bomb_defused").any().alias("has_def"),
             pl.col("event_type").eq("bomb_exploded").any().alias("has_exp"),
             pl.len().alias("n_out"),
             pl.col("tick").min().alias("out_tick")))
print("rounds with an outcome event:", OUT0.height, "| >1 outcome event:", OUT0.filter(pl.col("n_out") > 1).height,
      "| both defused and exploded:", OUT0.filter(pl.col("has_def") & pl.col("has_exp")).height)
OUT = (OUT0.filter(~(pl.col("has_def") & pl.col("has_exp")))
       .with_columns(pl.when(pl.col("has_def")).then(pl.lit(3)).otherwise(pl.lit(2)).cast(pl.Int64).alias("bomb_side")))

J = (RE.join(OUT.select(["match_key", "round", "out_tick", "bomb_side"]), on=["match_key", "round"], how="left")
     .join(H.select(["key", "match_id", "md_ts", "md_date", "path_date", "build_num", "map_name"])
           .rename({"key": "match_key"}), on="match_key", how="left"))
print("round_end rows with no header match:", J.filter(pl.col("md_ts").is_null()).height)

# ---------------------------------------------------------------- C. totals and round-13 onset
hdr("C. bomb-outcome rounds, player_count == 20: totals, round 13 by match_date, onset bracket")
P = (J.filter((pl.col("player_count") == 20) & pl.col("bomb_side").is_not_null())
     .with_columns(
         (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
         pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
           .otherwise(pl.lit("other")).alias("rbin"),
         (pl.col("md_ts") < CUT_CLAIM).alias("pre"),
         (pl.col("path_date") >= PATH_CUT).alias("pg"),
         (pl.col("out_tick") - pl.col("end_tick")).alias("lag"),
     ))
print("bomb-outcome rounds (pc20):", P.height, "| mismatches:", int(P["mm"].sum()), "  (D7_1: 15436 and 499)")
print(P.group_by("rbin").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("rbin"))
NP = J.filter((pl.col("player_count") != 20) & pl.col("bomb_side").is_not_null())
print("bomb-outcome rounds with player_count != 20 (n, mismatches):",
      NP.height, int((NP["bomb_side"] != NP["winner_team_code"]).sum()), "  (D7_1: 484, 0)")

R13 = P.filter(pl.col("round") == 13)
print("\nround 13 bomb rows:", R13.height, "| mismatches:", int(R13["mm"].sum()), "  (claim: 785 and 434)")
print("round 13 by match_date day (UTC as recorded):")
print(R13.with_columns(pl.col("md_ts").dt.strftime("%Y-%m-%d").alias("md_day"))
      .group_by("md_day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("md_day").rows())

rows13 = R13.sort("md_ts").select(["md_ts", "mm", "match_key", "build_num", "path_date"]).rows()
first_mm = next(r for r in rows13 if r[1])
clean_before = [r for r in rows13 if (not r[1]) and r[0] < first_mm[0]]
last_clean_before = max(clean_before, key=lambda r: r[0])
print("\nfirst round-13 mismatch (match_date):", first_mm[0], "| match", first_mm[2], "| build", first_mm[3],
      "| path", first_mm[4])
print("last clean round-13 row before it:", last_clean_before[0], "| match", last_clean_before[2])
print("round-13 rows with md_ts strictly between those two:",
      sum(1 for r in rows13 if last_clean_before[0] < r[0] < first_mm[0]))
print("round-13 rows with md_ts inside ledger window [00:24:22, 01:01:18] on 2024-02-07:",
      sum(1 for r in rows13 if ONSET_LO <= r[0] <= ONSET_HI))
# best single cut re-derived from the data (errors = mismatches before cut + clean rows after cut)
n13 = len(rows13)
mm_pre = [0] * (n13 + 1)
ok_pre = [0] * (n13 + 1)
for i, r in enumerate(rows13):
    mm_pre[i + 1] = mm_pre[i] + int(r[1])
    ok_pre[i + 1] = ok_pre[i] + (0 if r[1] else 1)
errs = [mm_pre[c] + (ok_pre[n13] - ok_pre[c]) for c in range(n13 + 1)]
best_c = min(range(n13 + 1), key=lambda c: errs[c])
print("best single cut over round-13 rows: errors =", errs[best_c], "| cut index", best_c, "of", n13,
      "| last before cut:", rows13[best_c - 1][0] if best_c else None,
      "| first at/after cut:", rows13[best_c][0] if best_c < n13 else None)
print("errors at claim cut (pre=md_ts < 02:39:55):", errs[sum(1 for r in rows13 if r[0] < CUT_CLAIM)])

pp = R13.group_by("pre").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("pre")
print("\nround 13 split at claim cut (pre = before 2024-02-07 02:39:55 UTC):")
print(pp)
print("ALL bomb rows split at claim cut:")
print(P.group_by(["rbin", "pre"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["rbin", "pre"]))

# ---------------------------------------------------------------- D. path day vs match time
hdr("D. path day (csds key) versus match time: which predicts the mismatch?")
print("pre = match_date before claim cut; pg = csds path day >= 2024-02-12")
print("\nround 13 only:")
print(R13.group_by(["pre", "pg"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["pre", "pg"]))
print("\nall bomb-outcome rounds (pc20):")
print(P.group_by(["pre", "pg"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["pre", "pg"]))
print("\nDefect's path-day grouping reproduced, round 13 (pg = path >= 02/12):")
print(R13.group_by("pg").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("pg"))
print("round-13 mismatch csds path days: min", R13.filter(pl.col("mm"))["path_date"].min(),
      "| max", R13.filter(pl.col("mm"))["path_date"].max())
clean_pre_pg = R13.filter(pl.col("pre") & pl.col("pg") & (~pl.col("mm")))
print("round-13 CLEAN rows with match_date before the cut but path day >= 02/12:", clean_pre_pg.height)
print("  sample:", clean_pre_pg.select(["md_ts", "path_date", "match_id", "build_num"]).head(5).rows())
post_pre_path = P.filter((~P["pre"]) & (~P["pg"]))
print("all bomb rows after cut with path day < 02/12 (n, mismatches):", post_pre_path.height,
      int(post_pre_path["mm"].sum()))

# ---------------------------------------------------------------- E. build number
hdr("E. build_num (header) versus match time")
print("bomb-outcome rounds (pc20) by build_num x pre-cut:")
print(P.group_by(["build_num", "pre"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["build_num", "pre"]))
print("round 13 by build_num x pre-cut:")
print(R13.group_by(["build_num", "pre"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["build_num", "pre"]))
first9961 = H.filter(pl.col("build_num") == 9961)["md_ts"].min()
print("first build 9961 match_date:", first9961, "| last build 9842 match_date:",
      H.filter(pl.col("build_num") == 9842)["md_ts"].max())
print("build 9961 matches before 00:24:22 on 02-07:",
      H.filter((pl.col("build_num") == 9961) & (pl.col("md_ts") < ONSET_LO)).height)
late9842 = H.filter((pl.col("build_num") == 9842) & (pl.col("md_ts") > first9961))
print("build 9842 matches AFTER first build-9961 match:", late9842.height, "of", H.filter(pl.col("md_ts") > first9961).height,
      "matches after that time")
print("  their bomb-outcome rounds (pc20) by rbin (n, mm):")
print(P.filter((pl.col("build_num") == 9842) & (pl.col("md_ts") > first9961))
      .group_by("rbin").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("rbin"))
print("  late build-9842 match_dates (up to 20):", late9842["md_ts"].sort().head(20).to_list())

# ---------------------------------------------------------------- F. lag signature and win reason
hdr("F. secondary signatures (lag = bomb-outcome tick - round_end tick; win reason text)")
P = P.with_columns((pl.col("lag") == -429).alias("lag429"))
print("round 13 by lag==-429 x mismatch:")
print(P.filter(pl.col("rbin") == "r13").group_by(["lag429", "mm"]).agg(pl.len().alias("n")).sort(["lag429", "mm"]))
print("all bomb rows by lag==-429 x mismatch:")
print(P.group_by(["lag429", "mm"]).agg(pl.len().alias("n")).sort(["lag429", "mm"]))
print("win_reason_message x bomb_side x winner_team_code x mismatch (all bomb rows, pc20):")
print(P.group_by(["win_reason_message", "bomb_side", "winner_team_code", "mm"]).agg(pl.len().alias("n")).sort(
    ["win_reason_message", "bomb_side", "winner_team_code", "mm"]))

print("\nDONE main")
