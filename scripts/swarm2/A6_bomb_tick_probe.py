"""A6 probe: is the bomb event before or after the round_end row, by win code (both eras)?

For every round with a defuse or explode event in bomb_state:
  d = tick(bomb defuse/explode event) - tick(round_end row of the same match and round)
  d < 0: bomb event inside the round (before its round_end row); d > 0: after it.
Also the tick of the last player death in the round relative to round_end (elimination codes),
and whether the plant precedes round_end.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_bomb_tick_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

ROOT = Path(r"G:\pureskill\csds-cs2")
ERAS = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}


def files(base, ch):
    return sorted(glob.glob(str(sorted(base.glob(f"basic_{ch}.*"))[0] / "dataframe_*")))


def load(base, ch, cols):
    return pl.scan_parquet(files(base, ch)).select(cols).collect()


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def main():
    for era, base in ERAS.items():
        section(f"ERA {era}")
        RE = load(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).unique(
            ["match_key", "round"], keep="first").rename({"tick": "re_tick"})
        BS = load(base, "bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
            pl.col("event_type").is_in(["bomb_defused", "bomb_exploded", "bomb_planted"]))
        bend = BS.filter(pl.col("event_type") != "bomb_planted").group_by(["match_key", "round"]).agg(
            pl.col("tick").min().alias("end_tick"), pl.col("event_type").first().alias("end_type"))
        plant = BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(
            pl.col("tick").min().alias("plant_tick"))
        PD = load(base, "player_death", ["match_key", "round", "tick", "player_id"]).filter(
            pl.col("player_id").is_not_null())
        lastd = PD.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("last_death_tick"))
        j = (RE.join(bend, on=["match_key", "round"], how="left")
             .join(plant, on=["match_key", "round"], how="left")
             .join(lastd, on=["match_key", "round"], how="left"))
        bj = j.filter(pl.col("end_tick").is_not_null()).with_columns(
            (pl.col("end_tick") - pl.col("re_tick")).alias("d_end"),
            pl.when(pl.col("end_tick") - pl.col("re_tick") < 0).then(pl.lit("before_re"))
            .when(pl.col("end_tick") - pl.col("re_tick") == 0).then(pl.lit("same_tick"))
            .otherwise(pl.lit("after_re")).alias("pos"))
        print("bomb end event position vs round_end tick, by event type and win code:")
        print(bj.group_by(["end_type", "win_reason_code", "pos"]).len().sort(["end_type", "win_reason_code", "pos"]))
        print("median d_end (ticks) by end_type x code:")
        print(bj.group_by(["end_type", "win_reason_code"]).agg(pl.len().alias("n"),
                                                               pl.col("d_end").median().alias("median_d_end"),
                                                               pl.col("d_end").min().alias("min_d_end"),
                                                               pl.col("d_end").max().alias("max_d_end"))
              .sort(["end_type", "win_reason_code"]))
        pj = j.filter(pl.col("plant_tick").is_not_null()).with_columns(
            (pl.col("plant_tick") < pl.col("re_tick")).alias("plant_before_re"))
        print("plant before round_end, by code:", pj.group_by(["win_reason_code", "plant_before_re"]).len().sort(
            ["win_reason_code", "plant_before_re"]).rows())
        # last death relative to round_end for codes 8/9
        ej = j.filter(pl.col("win_reason_code").is_in([8, 9]) & pl.col("last_death_tick").is_not_null()).with_columns(
            (pl.col("re_tick") - pl.col("last_death_tick")).alias("gap_re_minus_lastdeath"))
        print("code 8/9: gap (round_end tick - last death tick), quantiles (ticks):")
        print(ej.group_by("win_reason_code").agg(
            pl.len().alias("n"),
            pl.col("gap_re_minus_lastdeath").quantile(0.05).alias("p05"),
            pl.col("gap_re_minus_lastdeath").quantile(0.5).alias("p50"),
            pl.col("gap_re_minus_lastdeath").quantile(0.95).alias("p95"),
            (pl.col("gap_re_minus_lastdeath") < 0).sum().alias("death_after_re")).sort("win_reason_code"))


if __name__ == "__main__":
    main()
