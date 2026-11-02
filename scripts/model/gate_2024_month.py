"""The 2024 alive-count gate over the whole month (every 5-versus-5 match in positions_1hz_2024), with and without the two exclusions
found by gate_2024_side_recovery.py:

  roster-defect rounds        rounds.missing_side is set (the existing pinned rule)
  unknown-side death rounds   a death in the round has a null victim side in deaths.parquet (the gate cannot count it)

Pinned threshold (alive_gate.py): at least 99.9% of (snapshot, side) pairs agree. The full-month result is the decision input for
whether 2024 positions can be used.
Writes data/diagnostics/gate_2024_month.json.
Run in the cs2gpu environment (polars):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/gate_2024_month.py
"""
import glob
import json
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from gate_2024_residual import checked_table  # noqa: E402

ROOT = "G:/pureskill/csds-cs2"
OUT = os.path.join(ROOT, "data", "diagnostics", "gate_2024_month.json")
THRESHOLD = 0.999


def main():
    t0 = time.time()
    days = sorted(glob.glob(ROOT + "/data/positions_1hz_2024/2024-*.parquet"))
    pos = pl.concat([pl.read_parquet(d, columns=["match_key", "round", "tick", "second", "player_id", "side"]) for d in days])
    per_match = pos.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n"))
    five = per_match.filter(pl.col("n") == 10)["match_key"].to_list()
    print(f"2024 positions: {pos.height:,} rows, {per_match.height:,} matches, {len(five):,} five-versus-five ({time.time() - t0:.0f} s)",
          flush=True)
    P = pos.filter(pl.col("match_key").is_in(five))
    fe = (P.with_columns((pl.col("tick") - 64 * pl.col("second")).alias("fe"))
            .group_by(["match_key", "round"]).agg(pl.col("fe").min().alias("fe")))
    D = pl.read_parquet(ROOT + "/data/combined/deaths.parquet").filter(pl.col("match_key").is_in(five))
    D = D.join(fe, on=["match_key", "round"], how="inner").with_columns(
        ((pl.col("tick") - pl.col("fe") + 63) // 64).clip(lower_bound=0).cast(pl.Int64).alias("second"),
        pl.col("side").cast(pl.Int64).alias("victim_side"))
    snap = P.select(["match_key", "round", "second"]).unique()
    positions = P.select(["match_key", "round", "second", pl.col("side").cast(pl.Int64), "player_id"])
    chk = checked_table(positions, D.select(["match_key", "round", "second", "victim_side"]), snap, "ceil")
    print(f"gate pairs {chk.height:,} ({time.time() - t0:.0f} s)", flush=True)

    R = pl.read_parquet(ROOT + "/data/combined/rounds.parquet", columns=["match_key", "round", "missing_side"]).with_columns(
        pl.col("round").cast(pl.Int64)).filter(pl.col("match_key").is_in(five))
    null_rounds = (D.filter(pl.col("victim_side").is_null()).select(["match_key", pl.col("round").cast(pl.Int64)]).unique())
    defect_rounds = R.filter(pl.col("missing_side").is_not_null()).select(["match_key", "round"])
    all_rounds = R.select(["match_key", "round"]).unique()
    excluded = pl.concat([null_rounds, defect_rounds]).unique()

    chk = chk.with_columns(pl.col("round").cast(pl.Int64))
    agree_all = 1 - chk.filter(pl.col("diff") != 0).height / chk.height
    kept = chk.join(excluded.with_columns(pl.lit(True).alias("_x")), on=["match_key", "round"], how="left").filter(pl.col("_x").is_null())
    agree_kept = 1 - kept.filter(pl.col("diff") != 0).height / kept.height
    defect_only = chk.join(defect_rounds.with_columns(pl.lit(True).alias("_x")), on=["match_key", "round"], how="left").filter(pl.col("_x").is_null())
    agree_no_defect = 1 - defect_only.filter(pl.col("diff") != 0).height / defect_only.height
    res = {
        "matches_five_v_five": len(five),
        "rounds_all": all_rounds.height,
        "rounds_roster_defect": defect_rounds.height,
        "rounds_null_side_death": null_rounds.height,
        "rounds_excluded_union": excluded.height,
        "rounds_excluded_share": excluded.height / all_rounds.height,
        "agree_all_pairs": agree_all,
        "agree_without_roster_defect": agree_no_defect,
        "agree_kept_rounds": agree_kept,
        "kept_pairs": kept.height,
        "mismatch_pairs_kept_rounds": kept.filter(pl.col("diff") != 0).height,
        "threshold": THRESHOLD,
        "pass_all_pairs": bool(agree_all >= THRESHOLD),
        "pass_kept_rounds": bool(agree_kept >= THRESHOLD),
        "seconds": round(time.time() - t0),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
