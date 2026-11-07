"""A13: winner-label status, 2026 canonical vs 2024-02-g canonical (read-only, stdout only).

Label test: a round with a bomb outcome must have winner_team_code 2 (T) if exploded, 3 (CT) if defused.
Also checks win_reason_code 1 (Target_Bombed) -> 2 and 7 (Bomb_Defused) -> 3 on every round.
Day = the DD folder of the match key (csds/<yyyy>/<mm>/<DD>/<id>) for both tomes.
Round bins: r13 and r25 (the 2024 defect rounds in the open issue), other.
Output: logs/swarm2/A13_label_check.log (captured by the caller).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
TOMES = {
    "2026 canonical": (ROOT / "tomes/2026-10-01_to_08/tome/csds", ".2026-09-18,2026-10-08"),
    "2024 canonical": (ROOT / "tomes/2024-02-g/tome/csds", ".2023-12-28,2024-02-25"),
}


def read(base, suf, ch, cols):
    fd = base / f"basic_{ch}{suf}"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


for name, (base, suf) in TOMES.items():
    print(f"\n== {name} ({base}) ==")
    RE = read(base, suf, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code"])
    BS = read(base, suf, "bomb_state", ["match_key", "round", "event_type"])
    print(f"round_end rows {RE.height}; null winner {RE.filter(pl.col('winner_team_code').is_null()).height}")
    wr = (RE.filter(pl.col("win_reason_code").is_in([1, 7]))
          .with_columns(pl.when(pl.col("win_reason_code") == 1).then(2).otherwise(3).alias("expect")))
    wr_bad = wr.filter(pl.col("winner_team_code") != pl.col("expect")).height
    print(f"win_reason 1/7 rows: {wr.height}; winner disagrees with reason: {wr_bad}")
    outcome = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
               .group_by(["match_key", "round"])
               .agg(pl.col("event_type").unique().alias("evs")))
    multi = outcome.filter(pl.col("evs").list.len() > 1).height
    j = (RE.join(outcome, on=["match_key", "round"], how="inner")
         .with_columns(
             pl.when(pl.col("evs").list.contains("bomb_exploded")).then(2).otherwise(3).alias("side_from_bomb"),
             pl.col("match_key").str.split("/").list.get(3).alias("day"),
             pl.when(pl.col("round") == 13).then(pl.lit("r13"))
               .when(pl.col("round") == 25).then(pl.lit("r25"))
               .otherwise(pl.lit("other")).alias("rbin"))
         .with_columns((pl.col("side_from_bomb") != pl.col("winner_team_code")).alias("mismatch")))
    print(f"rounds with a bomb outcome: {j.height}; rounds with both outcome types: {multi}")
    print(f"mismatches (winner vs bomb outcome): {j.filter(pl.col('mismatch')).height}")
    by_bin = (j.group_by("rbin").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
              .with_columns((pl.col("mm") / pl.col("n")).round(4).alias("rate")).sort("rbin"))
    print("by round bin:")
    print(by_bin)
    r13 = j.filter(pl.col("round") == 13)
    byday = (r13.group_by("day").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
             .with_columns((pl.col("mm") / pl.col("n")).round(3).alias("rate")).sort("day"))
    print("round 13 by day:")
    print(byday)
    if name.startswith("2024"):
        early = r13.filter(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"]))
        late = r13.filter(~pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"]))
        print(f"round 13 days 01-07: n={early.height} mm={early.filter(pl.col('mismatch')).height}; "
              f"days 12+: n={late.height} mm={late.filter(pl.col('mismatch')).height}")
    else:
        print(f"round 13 total n={r13.height} mismatches={r13.filter(pl.col('mismatch')).height}")
