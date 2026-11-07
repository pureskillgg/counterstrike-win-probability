"""D4 extra check: inside bomb rounds, does round_end.win_reason_code agree with the bomb event or with winner_team_code?

For each full round (2024: player_count 20; 2026: player_count 10) with a bomb outcome row (label-based,
as in e2_side_check_r13.py), tabulate:
  bomb_event_side (defused -> 3 CT, exploded -> 2 T) x winner_team_code x win_reason_code x round bin.
Mismatch = bomb_event_side != winner_team_code. Also reports win_reason_message values, and the planter side
from bomb_planted rows in the same round (should be 2 = T).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_reason_check.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", 20),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", 10),
}
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read(base, ch, cols):
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


for era, (base, pc) in TOMES.items():
    print(f"\n==================== ERA {era} (full round = player_count {pc}) ====================")
    re_ = (read(base, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code",
                                    "win_reason_message", "player_count"])
           .unique(["match_key", "round"], keep="first")
           .filter(pl.col("player_count") == pc))
    bs = read(base, "bomb_state", ["match_key", "round", "event_type", "player_team_code"])
    outs = (bs.filter(pl.col("event_type").is_in(OUTCOMES))
            .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    planter = (bs.filter(pl.col("event_type") == "bomb_planted")
               .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side")))
    J = (outs.join(re_, on=["match_key", "round"], how="inner")
         .join(planter, on=["match_key", "round"], how="left")
         .with_columns(
             pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"),
             pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
             .otherwise(pl.lit("other")).alias("rbin"),
             pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
         )
         .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
    print(f"outcome rounds with full-round round_end: {J.height}; mismatches: {int(J['mismatch'].sum())}")
    print("\nbomb_side x winner_team_code x win_reason_code, by round bin (counts):")
    print(J.group_by(["rbin", "evs", "winner_team_code", "win_reason_code", "mismatch"]).len()
          .sort(["rbin", "mismatch", "evs", "winner_team_code", "win_reason_code"]))
    print("\nwin_reason_message for mismatched rounds:")
    print(J.filter(pl.col("mismatch")).group_by(["win_reason_code", "win_reason_message"]).len()
          .sort("len", descending=True).head(10))
    print("\nplanter side in outcome rounds (2 = T expected), by mismatch:")
    print(J.group_by(["mismatch", "planter_side"]).len().sort(["mismatch", "planter_side"]))
    if era == "2024":
        print("\n2024 mismatch by day and round bin:")
        print(J.group_by(["day", "rbin"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
              .sort(["day", "rbin"]))
