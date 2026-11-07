"""Feasibility probe (Task A): distinct values needed for the decision-window logic.

Read-only. Prints distinct event_type values for round_state, bomb_state, bomb_action and bomb_defuse,
the win_reason_code / winner_team_code / player_count counts from round_end, and null counts for
player_death.player_team_code. Nothing is written except the log.
Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\feas_probe_values.py
"""

import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SUFFIX = ".2026-09-18,2026-10-08"


def read(ch):
    folder = os.path.join(TOME, f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def main():
    for ch in ["round_state", "bomb_state", "bomb_action", "bomb_defuse"]:
        df = read(ch)
        print(f"== {ch} event_type counts (rows={df.height}) ==")
        print(df.group_by("event_type").len().sort("event_type").to_pandas().to_string(index=False))
        print()

    re_ = read("round_end")
    print(f"== round_end rows={re_.height} ==")
    print(re_.group_by("win_reason_code").agg(pl.len().alias("n"),
                                             pl.col("win_reason_message").first().alias("msg"))
          .sort("win_reason_code").to_pandas().to_string(index=False))
    print(re_.group_by("winner_team_code").len().sort("winner_team_code").to_pandas().to_string(index=False))
    print(re_.group_by("player_count").len().sort("player_count").to_pandas().to_string(index=False))
    print(f"round_end null counts: {re_.null_count().to_dicts()}")
    print()

    de = read("player_death")
    print(f"== player_death rows={de.height}; null player_team_code={de['player_team_code'].null_count()}; "
          f"null player_id={de['player_id'].null_count()}; null tick={de['tick'].null_count()} ==")
    print(de.group_by("player_team_code").len().sort("player_team_code", nulls_last=True)
          .to_pandas().to_string(index=False))
    print()

    bd = read("bomb_defuse")
    print(f"== bomb_defuse rows={bd.height}; rounds with >1 row: "
          f"{bd.group_by(['match_key', 'round']).len().filter(pl.col('len') > 1).height} ==")
    bs = read("bomb_state")
    print(f"bomb_state rows={bs.height}")


if __name__ == "__main__":
    main()
