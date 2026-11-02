"""Trace the grid's 116-second cap: the rows with no bomb planted at or after 115 seconds of live play, and whether their round is
planted later (the cap is applied only to rounds with no plant tick, so a round planted after 116 s keeps its pre-plant rows).

  ...python.exe -I -X utf8 scripts\\model\\exp_cap_trace.py
Writes data/combined/exp_cap_trace.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    d = os.path.join(W.ROOT, "data", "combined")
    assign = json.load(open(os.path.join(d, "split_manifest.json"), encoding="utf-8"))["assign"]
    g = pl.read_parquet(os.path.join(d, "grid.parquet"),
                        columns=["match_key", "round", "bomb_planted", "seconds_since_round_live", "label_ct_win"])
    planted_rounds = g.filter(pl.col("bomb_planted") == 1).select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("plant_later_or_now"))
    g = g.join(planted_rounds, on=["match_key", "round"], how="left").with_columns(pl.col("plant_later_or_now").fill_null(False))
    test_keys = [k for k, v in assign.items() if v == "test"]
    out = {}
    for name, cond in [("all_splits", pl.lit(True)), ("test_only", pl.col("match_key").is_in(test_keys))]:
        sub = g.filter(cond)
        for band_name, band in [("115_to_116", (pl.col("seconds_since_round_live") >= 115) & (pl.col("seconds_since_round_live") < 116)),
                                ("at_or_past_116", pl.col("seconds_since_round_live") >= 116)]:
            rows = sub.filter((pl.col("bomb_planted") == 0) & band)
            planted_later = rows.filter(pl.col("plant_later_or_now"))
            never = rows.filter(~pl.col("plant_later_or_now"))
            out[f"{name}_{band_name}"] = {
                "unplanted_rows": rows.height,
                "rounds_planted_later": planted_later.select(["match_key", "round"]).unique().height,
                "rows_in_rounds_planted_later": planted_later.height,
                "rows_in_rounds_never_planted": never.height,
                "ct_win_rate_never_planted": float(never["label_ct_win"].mean()) if never.height else None,
                "ct_win_rate_planted_later": float(planted_later["label_ct_win"].mean()) if planted_later.height else None,
            }
            print(name, band_name, out[f"{name}_{band_name}"], flush=True)
    W.dump_json(out, os.path.join(d, "exp_cap_trace.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
