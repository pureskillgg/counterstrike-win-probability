"""Can the served function compute score lead from the DSDK tome? For a few 2026 test matches in the tome, compare the freeze-end score
difference (ct_score minus t_score, read from the round_state channel at the freeze end) with the grid's score_lead_diag for the same round.
Reads the tome pages directly (no project code).

  ...python.exe -I -X utf8 scripts\\serving\\check_score_lead_tome.py [--matches 5]
"""
import argparse
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SUFFIX = ".2026-09-18,2026-10-08"


def read_channel(name):
    folder = os.path.join(TOME, "header" if name == "header" else f"basic_{name}{SUFFIX}")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed") if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=5)
    a = ap.parse_args()
    rs = read_channel("round_state")
    rstart = read_channel("round_start")
    print("round_state columns:", rs.columns, flush=True)
    grid = pl.read_parquet(os.path.join(ROOT, "data", "combined", "grid.parquet"),
                           columns=["match_key", "round", "tick", "score_lead_diag", "era"]).filter(pl.col("era") == "2026")
    # first live second of each round in the grid, with its score lead
    g_first = (grid.sort("tick").group_by(["match_key", "round"]).agg(pl.col("score_lead_diag").first().alias("grid_lead")))
    in_tome = set(rs["match_key"].unique().to_list()) & set(g_first["match_key"].unique().to_list())
    keys = sorted(in_tome)[: a.matches]
    print(f"tome matches also in the 2026 grid: {len(in_tome)}; checking {len(keys)}", flush=True)
    for key in keys:
        r = rs.filter(pl.col("match_key") == key)
        st = rstart.filter(pl.col("match_key") == key) if rstart is not None else None
        # freeze-end scores: the round_state row at or before each round's first event; use the last row before round start when available
        rounds = sorted(r["round"].unique().to_list())
        n_match = 0
        n_eq = 0
        for rd in rounds:
            rr = r.filter(pl.col("round") == rd).sort("tick")
            if rr.height == 0 or "ct_score" not in rr.columns:
                continue
            first = rr.row(0, named=True)
            tome_lead = (first["ct_score"] or 0) - (first["t_score"] or 0)
            gl = g_first.filter((pl.col("match_key") == key) & (pl.col("round") == rd))
            if gl.height == 0:
                continue
            n_match += 1
            n_eq += int(tome_lead == gl["grid_lead"][0])
        print(f"{key}: rounds compared {n_match}, tome lead equals grid lead in {n_eq}", flush=True)


if __name__ == "__main__":
    main()
