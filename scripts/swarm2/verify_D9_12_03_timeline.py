"""verify D9-12 step 3: physical timeline of one 2024 and one 2026 match from round_state, round_start,
round_end, round_mvp-free tables. Shows event order, round labels and cumulative score around round boundaries.

Read-only. Prints only a few matches to keep output small.
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
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def scan(base, suf, ch, cols):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    lf = pl.scan_parquet(fs)
    have = set(lf.collect_schema().names())
    return lf.select([c for c in cols if c in have]).collect()


for year, (base, suf) in TOMES.items():
    rs = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                                          "t_score_raw", "ct_score_raw", "phase"])
    rst = scan(base, suf, "round_start", ["match_key", "round", "tick"]).with_columns(pl.lit("BASIC_round_start").alias("event_type"))
    ren = scan(base, suf, "round_end", ["match_key", "round", "tick", "winner_team_code"]).with_columns(
        pl.lit("BASIC_round_end").alias("event_type"), pl.lit(None).cast(pl.Int64).alias("t_score"))
    keys = (rs.filter(pl.col("match_key").is_not_null()).select("match_key").unique().sort("match_key")
            ["match_key"].head(60).to_list())
    # pick a match with a complete 2026 or 2024 structure: first match alphabetically with >= 16 rounds
    cnt = rst.group_by("match_key").len().filter(pl.col("len") >= 16).sort("match_key")
    mk = cnt["match_key"][0]
    print(f"\n######## {year} example match {mk}")
    ev = pl.concat([
        rs.filter((pl.col("match_key") == mk) & pl.col("event_type").is_in(
            ["round_prestart", "round_start", "round_freeze_end", "round_end", "round_officially_ended",
             "round_poststart", "bomb_planted", "bomb_defused", "bomb_exploded", "cs_win_panel_round",
             "round_announce_last_round_half", "round_announce_match_start", "cs_win_panel_match", "begin_new_match"]))
        .with_columns(pl.col("winner_team_code") if False else pl.lit(None).cast(pl.Int64).alias("winner_team_code")),
        rst.filter(pl.col("match_key") == mk).with_columns(pl.lit(None).cast(pl.Int64).alias("t_score"),
                                                          pl.lit(None).cast(pl.Int64).alias("ct_score"),
                                                          pl.lit(None).cast(pl.Int64).alias("t_score_raw"),
                                                          pl.lit(None).cast(pl.Int64).alias("ct_score_raw"),
                                                          pl.lit(None).cast(pl.String).alias("phase"),
                                                          pl.lit(None).cast(pl.Int64).alias("winner_team_code")),
        ren.filter(pl.col("match_key") == mk).with_columns(pl.lit(None).cast(pl.Int64).alias("ct_score"),
                                                          pl.lit(None).cast(pl.Int64).alias("t_score_raw"),
                                                          pl.lit(None).cast(pl.Int64).alias("ct_score_raw"),
                                                          pl.lit(None).cast(pl.String).alias("phase")),
    ], how="diagonal_relaxed").sort("tick")
    ev = ev.select(["tick", "round", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "winner_team_code", "phase"])
    print(ev.head(40))
    print("...rows shown: 40 of", ev.height)
