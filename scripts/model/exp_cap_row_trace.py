"""The one unplanted test row past 116 seconds that sits in a round recorded as never planted: which match and round it is, and what the
round record says about its plant and its end. Writes data/combined/exp_cap_row_trace.json.

  ...python.exe -I -X utf8 scripts\\model\\exp_cap_row_trace.py
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
    test_keys = {k for k, v in assign.items() if v == "test"}
    g = pl.read_parquet(os.path.join(d, "grid.parquet"),
                        columns=["match_key", "round", "tick", "bomb_planted", "seconds_since_round_live", "label_ct_win", "era"])
    g = g.filter(pl.col("match_key").is_in(list(test_keys)))
    planted = g.filter(pl.col("bomb_planted") == 1).select(["match_key", "round"]).unique()
    rows = (g.filter((pl.col("bomb_planted") == 0) & (pl.col("seconds_since_round_live") >= 116))
            .join(planted.with_columns(pl.lit(True).alias("planted_later")), on=["match_key", "round"], how="left")
            .filter(pl.col("planted_later").is_null()))
    print(f"unplanted test rows past 116 s in never-planted rounds: {rows.height}", flush=True)
    out = {"rows": []}
    for r in rows.iter_rows(named=True):
        k, rd = r["match_key"], r["round"]
        rr = pl.read_parquet(os.path.join(d, "rounds.parquet")) if "rounds" not in out else out["rounds"]
        out["rounds"] = rr
        rec = rr.filter((pl.col("match_key") == k) & (pl.col("round") == rd))
        rec_d = rec.to_dicts()[0] if rec.height else {}
        grid_round = g.filter((pl.col("match_key") == k) & (pl.col("round") == rd)).sort("tick")
        info = {"match_key": k, "round": rd, "tick": r["tick"], "seconds_since_round_live": r["seconds_since_round_live"],
                "label_ct_win": r["label_ct_win"], "grid_rows_in_round": grid_round.height,
                "grid_last_seconds": float(grid_round["seconds_since_round_live"].max()) if grid_round.height else None,
                "rounds_record": {c: (str(v) if v is not None else None) for c, v in rec_d.items()
                                  if c in ("plant_tick", "live_start_tick", "freeze", "round_end_tick", "end", "end_eff", "final", "label_source", "era")}}
        out["rows"].append(info)
        print(info, flush=True)
    out.pop("rounds", None)
    W.dump_json(out, os.path.join(d, "exp_cap_row_trace.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
