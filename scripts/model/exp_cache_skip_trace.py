"""Why the 2026 snapshot cache lacks 10 matches that the grid holds. The cache build drops (1) roster-defect rounds and (2) rounds whose player
table and event-derived alive counts disagree (features_status.inconsistent_rounds). This script applies the same checks to the grid rows of
the 10 matches and reports how many rounds each one loses, and whether every round goes.

  ...python.exe -I -X utf8 scripts\\model\\exp_cache_skip_trace.py
Writes data/combined/exp_cache_skip_trace.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl

import features_status as FS
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    d = os.path.join(W.ROOT, "data", "combined")
    cache_keys = set(pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select("match_key").unique().collect()["match_key"].to_list())
    grid = pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026").collect()
    missing = sorted(set(grid["match_key"].unique().to_list()) - cache_keys)
    print(f"grid 2026 matches missing from the cache: {len(missing)}", flush=True)
    G = grid.filter(pl.col("match_key").is_in(missing))
    rounds_all = G.select(["match_key", "round"]).unique()
    G_rd = W.drop_roster_defect_rounds(G, "combined")
    rounds_after_roster = G_rd.select(["match_key", "round"]).unique()
    incons = FS.inconsistent_rounds(G_rd)
    res = []
    for k in missing:
        r_all = rounds_all.filter(pl.col("match_key") == k).height
        r_roster = rounds_after_roster.filter(pl.col("match_key") == k).height
        r_incons = incons.filter(pl.col("match_key") == k).height if incons.height else 0
        res.append({"match_key": k, "rounds_in_grid": r_all, "rounds_after_roster_drop": r_roster, "rounds_inconsistent": r_incons,
                    "rounds_left_after_both_drops": r_roster - r_incons})
        print(res[-1], flush=True)
    out = {"missing_matches": len(missing), "matches": res,
           "all_rounds_removed_by_both_drops": sum(1 for r in res if r["rounds_left_after_both_drops"] == 0)}
    W.dump_json(out, os.path.join(d, "exp_cache_skip_trace.json"))
    print(f"matches with every round removed by the two drops: {out['all_rounds_removed_by_both_drops']} of {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
