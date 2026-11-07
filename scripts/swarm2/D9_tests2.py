"""D9 follow-up: is the build (9961) or player_count effect separable, and are the bomb events
attributed to the right round? Read-only on the tome; reads the D9 scratchpad tables.

Assumptions (ASSUMPTION):
  - population P_all = bomb-outcome rounds (any player_count); P_main = player_count == 20
  - bomb outcome row = first bomb_defused / bomb_exploded row of a round (ticks, team codes)
  - round window = [round_start tick, round_end tick] of the same round (round_start from
    basic_round_start; one row per round assumed, first taken)
  - adjacency: winner of round r-1 and r+1 taken from round_end rows of the same match
"""
import collections
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
R = pl.read_parquet(SCR / "D9_round.parquet")
M = pl.read_parquet(SCR / "D9_match.parquet")


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


RJ = R.join(M.select(["match_key", "build_num", "rushb_version", "map_name", "frac_pc20",
                     "is_dup_group", "is_lowest_key_in_group"]), on="match_key", how="left").with_columns(
    (pl.col("player_count") == 20).alias("pc20"))
P_all = RJ.filter(pl.col("has_outcome"))
P_main = P_all.filter(pl.col("pc20"))


def mh_or_strat(df, exp_mask_col, y_col="mismatch"):
    """MH odds ratio for a boolean exposure column, strata = day."""
    num = den = 0.0
    used = 0
    for s in df["day"].unique().to_list():
        g = df.filter(pl.col("day") == s)
        e = g[exp_mask_col].to_numpy().astype(bool)
        y = g[y_col].cast(pl.Int64).to_numpy()
        a = int((e & (y == 1)).sum())
        b = int((e & (y == 0)).sum())
        c = int((~e & (y == 1)).sum())
        d = int((~e & (y == 0)).sum())
        n = a + b + c + d
        if n == 0 or a + b == 0 or c + d == 0:
            continue
        used += 1
        num += a * d / n
        den += b * c / n
    return (num / den if den > 0 else float("nan")), used


print("== A. Round 13, build x pc20 (all bomb-outcome rounds)")
t = (P_all.filter(pl.col("is_r13")).group_by(["build_num", "pc20"])
     .agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort(["build_num", "pc20"]))
print(t)

print("\n== B. Round 13, day x build, pc20 rounds only (defect window and clean days)")
cnt = collections.defaultdict(lambda: [0, 0])
for s, b, mmv in P_main.filter(pl.col("is_r13")).select(["day", "build_num", "mismatch"]).iter_rows():
    cnt[(s, str(b))][0] += 1
    cnt[(s, str(b))][1] += int(mmv or 0)
for s in sorted({k[0] for k in cnt}):
    print(f"   day {s}: " + "  ".join(f"build {b}: n={cnt[(s, b)][0]} mm={cnt[(s, b)][1]}"
                                   for b in ["9842", "9961"] if (s, b) in cnt))

print("\n== C. Build effect on round 13 (pc20), MH odds ratio stratified by day")
r13m = P_main.filter(pl.col("is_r13")).with_columns((pl.col("build_num") == 9961).alias("b9961"))
orr, used = mh_or_strat(r13m, "b9961")
print(f"   build 9961 vs 9842: MH OR={orr:.2f}, strata used={used}")
allm = P_main.with_columns((pl.col("build_num") == 9961).alias("b9961"))
orr2, used2 = mh_or_strat(allm, "b9961")
print(f"   build 9961 vs 9842, all rounds with outcome (pc20): MH OR={orr2:.2f}, strata used={used2}")
w = P_main.filter((pl.col("build_num") == 9961) & pl.col("is_r13"))
print(f"   within build 9961, round 13 (pc20): n={w.height}, mismatch={int(w['mismatch'].sum())}")
w2 = P_main.filter((pl.col("build_num") == 9842) & pl.col("is_r13"))
print(f"   within build 9842, round 13 (pc20): n={w2.height}, mismatch={int(w2['mismatch'].sum())}")

print("\n== D. Within build 9961 and round 13: pc20 status and map")
z = P_all.filter((pl.col("build_num") == 9961) & pl.col("is_r13"))
print("   pc20 within 9961 r13 (n, mm):", z.group_by("pc20").agg(pl.len().alias("n"),
                                                               pl.col("mismatch").sum().alias("mm")).sort("pc20").rows())
zm = P_main.filter((pl.col("build_num") == 9961) & pl.col("is_r13"))
print("   map within 9961 r13 pc20 (map, n, mm):",
      zm.group_by("map_name").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
      .sort("map_name").rows())
zm_all = P_main.filter((pl.col("build_num") == 9961) & ~pl.col("is_r13"))
print("   non-r13 rounds in build 9961 (pc20): n=", zm_all.height, "mm=", int(zm_all["mismatch"].sum()))
print("   non-r13 mismatches by round (build 9961):",
      zm_all.filter(pl.col("mismatch") == 1).group_by("round").len().sort("round").rows())
print("   non-r13 mismatches by round (build 9842):",
      P_main.filter((pl.col("build_num") == 9842) & (pl.col("mismatch") == 1)).group_by("round").len().rows())
mn = P_main.filter((pl.col("build_num") == 9961) & pl.col("is_r13") & (pl.col("mismatch") == 0))
print("   r13 pc20 build-9961 rounds with NO mismatch (n=%d):" % mn.height)
print(mn.select(["match_key", "day", "winner_team_code", "win_reason_message", "bomb_side", "n_outcome_rows",
                 "planter_side", "dc_rows"]))

print("\n== E. Bomb-event attribution check (round 13 and r25, all bomb-outcome rounds)")
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RST = scan("round_start", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("start_tick"))
RE = R.select(["match_key", "round", "tick", "winner_team_code"]).rename({"tick": "end_tick"})
out_rows = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
            .sort("tick")
            .group_by(["match_key", "round"])
            .agg(pl.col("tick").first().alias("bomb_tick"),
                 pl.col("event_type").first().alias("bomb_event"),
                 pl.col("player_team_code").first().alias("bomb_event_team"),
                 pl.len().alias("n_outcome_rows")))
print("   outcome rows per round (value counts):", out_rows.group_by("n_outcome_rows").len().rows())
E = (out_rows.join(RE, on=["match_key", "round"], how="inner")
     .join(RST, on=["match_key", "round"], how="left")
     .with_columns(pl.when(pl.col("bomb_event") == "bomb_defused").then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                   (pl.col("bomb_tick") >= pl.col("start_tick")).alias("after_start"),
                   (pl.col("bomb_tick") <= pl.col("end_tick")).alias("before_end")))
E = E.with_columns(
    pl.when(pl.col("after_start") & pl.col("before_end")).then(pl.lit("inside"))
      .when(pl.col("bomb_tick") > pl.col("end_tick")).then(pl.lit("after_end"))
      .when(pl.col("bomb_tick") < pl.col("start_tick")).then(pl.lit("before_start"))
      .otherwise(pl.lit("unknown")).alias("tick_window"))
E = E.with_columns(pl.col("mismatch").fill_null(False))
print("   tick window x mismatch (round 13 rows):")
print(E.filter(pl.col("round") == 13).group_by(["mismatch", "tick_window"]).len().sort(["mismatch", "tick_window"]))
print("   tick window x mismatch (all rounds):")
print(E.group_by(["mismatch", "tick_window"]).len().sort(["mismatch", "tick_window"]))
print("   bomb event team code (r13 mismatch vs match):")
print(E.filter(pl.col("round") == 13).group_by(["mismatch", "bomb_event", "bomb_event_team"]).len()
      .sort(["mismatch", "bomb_event", "bomb_event_team"]))

print("\n   Adjacent-round winners for round 13 mismatch rows: does the bomb outcome match round 12 or 14?")
wprev = RE.select(["match_key", "round", "winner_team_code"]).with_columns(
    (pl.col("round") + 1).alias("round")).rename({"winner_team_code": "w_prev"})
wnext = RE.select(["match_key", "round", "winner_team_code"]).with_columns(
    (pl.col("round") - 1).alias("round")).rename({"winner_team_code": "w_next"})
A = (E.filter(pl.col("round") == 13)
     .join(wprev, on=["match_key", "round"], how="left")
     .join(wnext, on=["match_key", "round"], how="left"))
A = A.with_columns(
    (pl.col("bomb_side") == pl.col("w_prev")).alias("bomb_eq_prev_winner"),
    (pl.col("bomb_side") == pl.col("w_next")).alias("bomb_eq_next_winner"),
)
print(A.group_by(["mismatch", "bomb_eq_prev_winner", "bomb_eq_next_winner"]).len()
      .sort(["mismatch", "bomb_eq_prev_winner", "bomb_eq_next_winner"]))

print("\n== F. Match-level clustering: r13 mismatch by match (dup copies share status?)")
mm_by_match = (P_main.filter(pl.col("is_r13")).group_by("match_key")
               .agg(pl.col("mismatch").sum().alias("mm"), pl.len().alias("n")))
print("   r13 pc20 matches:", mm_by_match.height,
      "| matches with mismatch:", mm_by_match.filter(pl.col("mm") > 0).height)
np.random.seed(0)
