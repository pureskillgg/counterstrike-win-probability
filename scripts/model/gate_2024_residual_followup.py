"""Follow-up to gate_2024_residual.py: are the missing living players absent from positions for a whole round (a player who never gets rows),
or do their rows carry a side code the gate does not count (not 2 or 3)? Same seeded 2024 sample as the residual diagnostic.

For each (match, round, side) with a persistent shortfall, the script counts:
  players_seen_in_round      distinct player_id with any row in positions for that round (5 expected on a full side)
  rows_side_not_2_or_3       positions rows in the round whose side is neither 2 nor 3 (the gate ignores them)
  onset_at_round_start       the shortfall starts at the round's first positions second
Writes data/diagnostics/gate_2024_residual_followup.json.
Run in the cs2gpu environment (polars):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/gate_2024_residual_followup.py
"""
import glob
import json
import os
import random
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from gate_2024_residual import checked_table  # noqa: E402

ROOT = "G:/pureskill/csds-cs2"
OUT = os.path.join(ROOT, "data", "diagnostics", "gate_2024_residual_followup.json")


def main():
    t0 = time.time()
    days = sorted(glob.glob(ROOT + "/data/positions_1hz_2024/2024-*.parquet"))
    pos = pl.concat([pl.read_parquet(d, columns=["match_key", "round", "tick", "second", "player_id", "side"]) for d in days])
    per_match = pos.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n"))
    five = sorted(per_match.filter(pl.col("n") == 10)["match_key"].to_list())
    random.seed(20241)
    sample = random.sample(five, 100)
    P = pos.filter(pl.col("match_key").is_in(sample))
    fe = (P.with_columns((pl.col("tick") - 64 * pl.col("second")).alias("fe"))
            .group_by(["match_key", "round"]).agg(pl.col("fe").min().alias("fe")))
    D = pl.read_parquet(ROOT + "/data/combined/deaths.parquet").filter(pl.col("match_key").is_in(sample))
    D = D.join(fe, on=["match_key", "round"], how="inner").with_columns(
        ((pl.col("tick") - pl.col("fe") + 63) // 64).clip(lower_bound=0).cast(pl.Int64).alias("second"),
        pl.col("side").cast(pl.Int64).alias("victim_side"))
    deaths = D.select(["match_key", "round", "second", "victim_side"])
    snap = P.select(["match_key", "round", "second"]).unique()
    positions = P.select(["match_key", "round", "second", pl.col("side").cast(pl.Int64), "player_id"])
    chk = checked_table(positions, deaths, snap, "ceil")
    bad = chk.filter(pl.col("diff") != 0).filter(pl.col("side").is_in([2, 3]))
    cs = chk.sort(["match_key", "round", "side", "second"])
    first = (bad.group_by(["match_key", "round", "side"]).agg(pl.col("second").min().alias("first_bad"), pl.len().alias("bad_seconds")))
    sides = first.join(cs.group_by(["match_key", "round", "side"]).agg(pl.col("second").min().alias("round_first_second")),
                       on=["match_key", "round", "side"], how="left")
    onset_start = (sides.filter(pl.col("first_bad") == pl.col("round_first_second")).height / sides.height) if sides.height else None

    seen = P.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("players_seen_in_round"))
    not23 = (P.filter(~pl.col("side").is_in([2, 3])).group_by(["match_key", "round"]).agg(pl.len().alias("rows_side_not_2_or_3")))
    per_round = (sides.select(["match_key", "round"]).unique()
                 .join(seen, on=["match_key", "round"], how="left").join(not23, on=["match_key", "round"], how="left")
                 .with_columns(pl.col("rows_side_not_2_or_3").fill_null(0)))
    all_rounds = seen.join(not23, on=["match_key", "round"], how="left").with_columns(pl.col("rows_side_not_2_or_3").fill_null(0))

    rounds_flags = pl.read_parquet(ROOT + "/data/combined/rounds.parquet", columns=["match_key", "round", "missing_side"]).with_columns(
        pl.col("round").cast(pl.Int64))
    pr = per_round.with_columns(pl.col("round").cast(pl.Int64)).join(rounds_flags, on=["match_key", "round"], how="left")
    pr = pr.with_columns(pl.col("missing_side").is_not_null().alias("roster_defect"))
    overlap = {f"players_{n}_roster_defect_{d}": int(pr.filter((pl.col("players_seen_in_round") == n) & (pl.col("roster_defect") == d)).height)
               for n in [9, 10] for d in [True, False]}
    all_r = all_rounds.with_columns(pl.col("round").cast(pl.Int64)).join(rounds_flags, on=["match_key", "round"], how="left").with_columns(
        pl.col("missing_side").is_not_null().alias("roster_defect"))
    all_overlap = {f"all_players_{n}_roster_defect_{d}": int(all_r.filter((pl.col("players_seen_in_round") == n) & (pl.col("roster_defect") == d)).height)
                   for n in [9, 10] for d in [True, False]}

    res = {
        "overlap_with_roster_defect_bad_rounds": overlap,
        "overlap_with_roster_defect_all_sampled_rounds": all_overlap,
        "sample_matches": len(sample),
        "bad_sides": sides.height,
        "onset_at_round_first_second_share": onset_start,
        "players_seen_in_round_distribution_bad_rounds": {str(k): int(v) for k, v in per_round.group_by("players_seen_in_round").agg(pl.len()).rows()},
        "players_seen_in_round_distribution_all_rounds": {str(k): int(v) for k, v in all_rounds.group_by("players_seen_in_round").agg(pl.len()).rows()},
        "bad_rounds_with_side_not_2_or_3_rows": int(per_round.filter(pl.col("rows_side_not_2_or_3") > 0).height),
        "all_rounds_with_side_not_2_or_3_rows": int(all_rounds.filter(pl.col("rows_side_not_2_or_3") > 0).height),
        "all_rounds": all_rounds.height,
        "bad_rounds": per_round.height,
        "seconds": round(time.time() - t0),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
