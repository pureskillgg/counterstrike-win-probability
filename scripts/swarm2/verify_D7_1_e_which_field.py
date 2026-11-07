"""D7-1 verifier, part E: which field is physically consistent in round-13 mismatch rows?
E1: bomb-event player_team_code (defuser / explosion actor side) for defused and exploded events, by mismatch.
E2: timing of the outcome event relative to round_end (out_tick - tick), by event type and mismatch.
E3: which round numbers are missing per match, by build (round-number sequence check).
ASSUMPTION: a bomb_defused event is performed by a CT-side player (team code 3); bomb_exploded is a T-side win.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
R = pl.read_parquet(str(SCRATCH / "verify_D7_1_rounds.parquet"))
H = pl.read_parquet(str(SCRATCH / "verify_D7_1_header.parquet"))
BS = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type", "player_team_code"])
                .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])) for f in files(BASE / f"basic_bomb_state{SUF}")]).collect()
ev = (BS.group_by(["match_key", "round", "event_type"]).agg(pl.col("player_team_code").first().alias("ev_team"),
      pl.len().alias("n_ev")))
B = R.filter(pl.col("bomb_side").is_not_null() & (pl.col("player_count") == 20) & (pl.col("round") == 13))
J = B.join(ev, on=["match_key", "round"], how="left")
print("E1 round-13 bomb rows with an outcome event row (event_type, ev_team) by mismatch:")
print(J.group_by(["mm", "event_type", "ev_team"]).len().sort(["mm", "event_type", "ev_team"]).rows())
print("E1 event_type counts by mismatch (rows):", J.group_by(["mm", "event_type"]).len().sort(["mm", "event_type"]).rows())
print("E1 round-13 mismatch rows with planted bomb (planter_side) by event_type:",
      J.filter(pl.col("mm")).group_by(["event_type", "planter_side"]).len().sort(["event_type", "planter_side"]).rows())

# E2 timing: outcome tick minus round_end tick (ticks at 64/s)
E2 = B.with_columns(pl.when(pl.col("bomb_side") == 2).then(pl.lit("exploded")).otherwise(pl.lit("defused")).alias("ev_side"),
                    (pl.col("out_tick") - pl.col("tick")).alias("dt"))
print("\nE2 outcome tick - round_end tick (ticks), by bomb side and mismatch:")
print(E2.group_by(["ev_side", "mm"]).agg(pl.len().alias("n"), pl.col("dt").min().alias("min"), pl.col("dt").median().alias("median"),
      pl.col("dt").max().alias("max")).sort(["ev_side", "mm"]))

# E3: missing round numbers per match, by build
RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round"]) for f in files(BASE / f"basic_round_end{SUF}")]).collect()
M = RE.group_by("match_key").agg(pl.col("round").max().alias("maxr"), pl.col("round").unique().alias("rs"))
M = M.join(H.select(["key", "build_num"]).rename({"key": "match_key"}), on="match_key")
M = M.with_columns(pl.struct(["rs", "maxr"]).map_elements(
    lambda s: [i for i in range(1, s["maxr"] + 1) if i not in set(s["rs"])], return_dtype=pl.List(pl.Int64)).alias("missing"))
ex = M.explode("missing").filter(pl.col("missing").is_not_null())
print("\nE3 missing round numbers within 1..maxr, by build:")
print(ex.group_by(["build_num", "missing"]).len().sort(["build_num", "missing"]).rows())
print("E3 matches with no missing round, by build:", M.filter(pl.col("missing").list.len() == 0).group_by("build_num").len().rows())
print("E3 min round number by build:", M.group_by("build_num").agg(pl.col("rs").list.min().min().alias("min_round")).rows())
