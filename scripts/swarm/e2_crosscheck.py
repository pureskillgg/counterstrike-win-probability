"""Independent cross-check for E2 map profile (no dedupe, no kill logic).

Recomputes per map, for each era, from G: tomes only:
  - matches (header rows), clean 5v5 rounds (round_end.player_count == pc),
  - CT win count (winner_team_code == 3),
  - live-start count (round_state round_freeze_end), planted rounds (bomb_state bomb_planted in live window).
Prints a compact table so it can be compared against scripts\\swarm\\out\\e2_map_profile.json.
Run: pixi run python -I -X utf8 scripts\\swarm\\e2_crosscheck.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
ERAS = {
    "2026": {"base": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
             "suffix": ".2026-09-18,2026-10-08", "pc": 10},
    "2024": {"base": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
             "suffix": ".2023-12-28,2024-02-25", "pc": 20},
}


def read(base, suffix, ch, cols):
    fd = base / ("header" if ch == "header" else f"basic_{ch}{suffix}")
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    if not files:
        raise SystemExit(f"no pages for {ch} in {fd}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


for era, cfg in ERAS.items():
    base, suffix, pc = cfg["base"], cfg["suffix"], cfg["pc"]
    hdr = read(base, suffix, "header", ["key", "map_name"]).rename({"key": "match_key"})
    RE = read(base, suffix, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    RS = read(base, suffix, "round_state", ["match_key", "round", "tick", "event_type"])
    BS = read(base, suffix, "bomb_state", ["match_key", "round", "tick", "event_type"])

    clean = RE.filter(pl.col("player_count") == pc).join(hdr, on="match_key", how="left")
    per_map = clean.group_by("map_name").agg(
        pl.col("match_key").n_unique().alias("matches_raw_clean"),
        pl.len().alias("rounds_raw_clean"),
        (pl.col("winner_team_code") == 3).sum().alias("ct_wins_raw"),
    ).sort("rounds_raw_clean", descending=True)

    live = (RS.filter(pl.col("event_type") == "round_freeze_end")
            .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("live_tick")))
    cr = clean.select(["match_key", "round", "tick", "map_name"]).join(live, on=["match_key", "round"], how="left")
    live_per_map = cr.group_by("map_name").agg(pl.col("live_tick").is_not_null().sum().alias("rounds_with_live"))

    plants = (BS.filter(pl.col("event_type") == "bomb_planted")
              .select(["match_key", "round", "tick"]).rename({"tick": "plant_tick"}))
    pl_round = (cr.join(plants, on=["match_key", "round"], how="inner")
                .filter(pl.col("live_tick").is_not_null() & (pl.col("plant_tick") >= pl.col("live_tick"))
                        & (pl.col("plant_tick") <= pl.col("tick")))
                .group_by(["match_key", "round"]).agg(pl.col("map_name").first()))
    planted_per_map = pl_round.group_by("map_name").agg(pl.len().alias("planted_rounds_raw"))

    out = (per_map.join(live_per_map, on="map_name", how="left")
           .join(planted_per_map, on="map_name", how="left")
           .with_columns((pl.col("ct_wins_raw") / pl.col("rounds_raw_clean")).alias("ct_rate_raw")))
    print(f"\n### {era} cross-check (no dedupe; clean pc={pc})")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60):
        print(out)
    print(f"total clean rounds {clean.height}, total CT wins {int((clean['winner_team_code'] == 3).sum())}, "
          f"header rows {hdr.height}")
