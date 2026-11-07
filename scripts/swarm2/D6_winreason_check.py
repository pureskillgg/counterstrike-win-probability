"""D6 third source: win_reason_code (inside the same round_end row) gives the winning side.
T-win reasons: 1 Target_Bombed, 9 Terrorists_Win, 13 Hostages_Not_Rescued, 17 Terrorists_Surrender.
CT-win reasons: 7 Bomb_Defused, 8 CTs_Win, 11 All_Hostages_Rescued, 12 Target_Saved, 18 CTs_Surrender.
Compares side-from-reason with winner_team_code for 2024 and 2026, by round, using the same
first-post-swap positions as D6_swap_detect.py (round 9, 13, 28)."""
import sys, glob, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
T_REASONS = {1, 9, 13, 17}
CT_REASONS = {7, 8, 11, 12, 18}

def read(base, ch, cols):
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")

for era, base in [("2024", ROOT/"tomes"/"2024-02-g"/"tome"/"csds"), ("2026", ROOT/"tomes"/"2026-10-01_to_08"/"tome"/"csds")]:
    re_ = read(base, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code"]).unique(["match_key","round"], keep="first")
    side = re_.with_columns(
        pl.when(pl.col("win_reason_code").is_in(list(T_REASONS))).then(pl.lit(2))
          .when(pl.col("win_reason_code").is_in(list(CT_REASONS))).then(pl.lit(3))
          .otherwise(pl.lit(None)).alias("reason_side"))
    print(f"== {era}: win_reason_code values: {sorted(set(side['win_reason_code'].drop_nulls().to_list()))}; "
          f"unmapped rows: {side.filter(pl.col('reason_side').is_null()).height} of {side.height}")
    agree = side.filter(pl.col("reason_side").is_not_null())
    tot = agree.height
    dis = agree.filter(pl.col("reason_side") != pl.col("winner_team_code")).height
    print(f"  reason-side vs winner_team_code, all rounds: {tot - dis} agree, {dis} disagree")
    for r in (9, 12, 13, 14, 28):
        s = agree.filter(pl.col("round") == r)
        d = s.filter(pl.col("reason_side") != pl.col("winner_team_code")).height
        print(f"  round {r:>2}: rows={s.height} disagree={d}")
    dr = agree.filter(pl.col("reason_side") != pl.col("winner_team_code"))["round"].value_counts().sort("count", descending=True).head(6)
    print(f"  disagreements by round (top 6): {dr.rows()}")
