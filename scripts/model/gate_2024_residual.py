"""Diagnose the 2024 alive-count gate residual: every disagreeing (snapshot, side) pair, classified.

The gate (alive_gate.py) passes when 99.9% of (snapshot, side) pairs agree: observed living players (positions_1hz rows for that side at
that second) equals 5 minus that side's cumulative deaths. The seeded sample (100 five-versus-five 2024 matches, random.seed(20241),
the same sample as gate_2024_diag.py) gave 95.1% agreement overall, 98.4% after dropping roster-defect rounds.

This script keeps every mismatch (the gate keeps 20) and classifies it by:
  direction         observed minus expected (positive: more players observed than expected)
  death timing      deaths of that side at second s-1, s, s+1 (is the mismatch next to a death?)
  duplicate rows    the same player appearing twice at one second in positions_1hz
  round flags       missing_side (roster defect), swap boundary, duplicate-event group, recorded vs final winner disagreement
  death convention  agreement with death seconds mapped by ceil (the gate) versus floor
Writes data/diagnostics/gate_2024_residual.json.
Run in the cs2gpu environment (polars):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/gate_2024_residual.py
"""
import glob
import json
import os
import random
import sys
import time

import polars as pl

ROOT = "G:/pureskill/csds-cs2"
OUT = os.path.join(ROOT, "data", "diagnostics", "gate_2024_residual.json")
SIDES = [2, 3]


def checked_table(positions, deaths, snaps, mode="ceil"):
    """The gate's (snapshot, side) table, with the extra columns the diagnosis needs. mode selects the death-second convention."""
    grid = (snaps.select("match_key", "round", "second").unique()
            .with_columns(pl.lit(1).alias("_k"))
            .join(pl.DataFrame({"side": SIDES}).with_columns(pl.lit(1).alias("_k")), on="_k").drop("_k"))
    obs = positions.group_by(["match_key", "round", "second", "side"]).agg(pl.len().alias("observed"))
    dup = (positions.group_by(["match_key", "round", "second", "player_id"]).agg(pl.len().alias("rows"))
           .filter(pl.col("rows") > 1).group_by(["match_key", "round", "second"]).agg(pl.len().alias("dup_players")))
    dead = (deaths.group_by(["match_key", "round", "second", "victim_side"]).agg(pl.len().alias("n_deaths"))
            .rename({"victim_side": "side"}))
    pieces = []
    for side in SIDES:
        d = dead.filter(pl.col("side") == side).select("match_key", "round", "second", "n_deaths")
        g = grid.filter(pl.col("side") == side).select("match_key", "round", "second")
        g = g.join(d, on=["match_key", "round", "second"], how="left").with_columns(pl.col("n_deaths").fill_null(0))
        g = g.sort(["match_key", "round", "second"]).with_columns(
            pl.col("n_deaths").cum_sum().over(["match_key", "round"]).alias("cum_deaths"),
            pl.lit(side).alias("side"))
        for lag, name in [(-1, "deaths_prev"), (0, "deaths_now"), (1, "deaths_next")]:
            e = d.with_columns((pl.col("second") - lag).alias("second")).rename({"n_deaths": name})
            g = g.join(e.select(["match_key", "round", "second", name]), on=["match_key", "round", "second"], how="left") \
                 .with_columns(pl.col(name).fill_null(0))
        pieces.append(g)
    cum = pl.concat(pieces)
    chk = (grid.join(cum, on=["match_key", "round", "second", "side"], how="left")
               .join(obs, on=["match_key", "round", "second", "side"], how="left")
               .join(dup, on=["match_key", "round", "second"], how="left")
               .with_columns(pl.col("observed").fill_null(0), pl.col("dup_players").fill_null(0),
                             (5 - pl.col("cum_deaths").fill_null(0)).alias("expected")))
    chk = chk.with_columns(pl.col("observed").cast(pl.Int64), pl.col("expected").cast(pl.Int64))
    chk = chk.with_columns((pl.col("observed") - pl.col("expected")).alias("diff"))
    return chk


def main():
    t0 = time.time()
    days = sorted(glob.glob(ROOT + "/data/positions_1hz_2024/2024-*.parquet"))
    pos = pl.concat([pl.read_parquet(d, columns=["match_key", "round", "tick", "second", "player_id", "side"]) for d in days])
    per_match = pos.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n"))
    five = sorted(per_match.filter(pl.col("n") == 10)["match_key"].to_list())  # sorted: the sample must not depend on group-by order
    random.seed(20241)
    sample = random.sample(five, 100)
    P = pos.filter(pl.col("match_key").is_in(sample))
    fe = (P.with_columns((pl.col("tick") - 64 * pl.col("second")).alias("fe"))
            .group_by(["match_key", "round"]).agg(pl.col("fe").min().alias("fe")))
    D = pl.read_parquet(ROOT + "/data/combined/deaths.parquet").filter(pl.col("match_key").is_in(sample))
    D = D.join(fe, on=["match_key", "round"], how="inner").with_columns(pl.col("side").cast(pl.Int64).alias("victim_side"))
    deaths_ceil = D.with_columns(((pl.col("tick") - pl.col("fe") + 63) // 64).clip(lower_bound=0).cast(pl.Int64).alias("second"))
    deaths_floor = D.with_columns(((pl.col("tick") - pl.col("fe")) // 64).clip(lower_bound=0).cast(pl.Int64).alias("second"))
    snap = P.select(["match_key", "round", "second"]).unique()
    positions = P.select(["match_key", "round", "second", pl.col("side").cast(pl.Int64), "player_id"])

    chk = checked_table(positions, deaths_ceil.select(["match_key", "round", "second", "victim_side"]), snap, "ceil")
    n = chk.height
    bad = chk.filter(pl.col("diff") != 0)
    print(f"pairs {n:,}; disagreeing {bad.height:,} ({1 - bad.height / n:.4f} agree); {time.time() - t0:.0f} s", flush=True)

    R = pl.read_parquet(ROOT + "/data/combined/rounds.parquet",
                        columns=["match_key", "round", "missing_side", "swap_boundary", "after_swap", "dup_group_size", "final_differs_from_recorded"])
    bad = bad.join(R, on=["match_key", "round"], how="left")
    summary = {"pairs": n, "disagreeing": bad.height, "agree_share": 1 - bad.height / n}
    summary["direction"] = {"observed_more": int(bad.filter(pl.col("diff") > 0).height), "observed_fewer": int(bad.filter(pl.col("diff") < 0).height),
                            "observed_zero": int(bad.filter(pl.col("observed") == 0).height)}

    # Swap hypothesis: at half-time the two sides trade places. If the mismatch is a side swap, the observed count on each side equals
    # the expected count on the other side. Test this on the mismatched (round, second) pairs.
    both = chk.select(["match_key", "round", "second", "side", "observed", "expected"])
    pair = (both.filter(pl.col("side") == 2).select(["match_key", "round", "second", pl.col("observed").alias("obs2"), pl.col("expected").alias("exp2")])
            .join(both.filter(pl.col("side") == 3).select(["match_key", "round", "second", pl.col("observed").alias("obs3"), pl.col("expected").alias("exp3")]),
                  on=["match_key", "round", "second"], how="inner"))
    mis = pair.filter((pl.col("obs2") != pl.col("exp2")) | (pl.col("obs3") != pl.col("exp3")))
    swapped = mis.filter((pl.col("obs2") == pl.col("exp3")) & (pl.col("obs3") == pl.col("exp2")))
    summary["swap_test"] = {"mismatch_seconds": mis.height, "explained_by_side_swap": swapped.height,
                            "share_explained": swapped.height / mis.height if mis.height else None}
    rounds_all = pl.read_parquet(ROOT + "/data/combined/rounds.parquet", columns=["match_key", "round", "swap_boundary"]).filter(
        pl.col("match_key").is_in(sample))
    summary["swap_boundary_true_share_all_rounds"] = float(rounds_all["swap_boundary"].cast(pl.Float64).mean())
    bad_rounds = bad.select(["match_key", "round", "swap_boundary"]).unique()
    summary["swap_boundary_true_share_mismatch_rounds"] = float(bad_rounds["swap_boundary"].cast(pl.Float64).mean())

    # Persistence: once a player is missing from positions with no death, does the shortfall stay for the rest of the round?
    cs = chk.filter(pl.col("side").is_in(SIDES)).select(["match_key", "round", "second", "side", "diff", "observed", "expected"])
    cs = cs.sort(["match_key", "round", "side", "second"]).with_columns(
        pl.col("observed").shift(1).over(["match_key", "round", "side"]).alias("obs_prev"),
        pl.col("expected").shift(1).over(["match_key", "round", "side"]).alias("exp_prev"))
    first = (cs.filter(pl.col("diff") != 0).group_by(["match_key", "round", "side"])
               .agg(pl.col("second").min().alias("first_bad"), pl.len().alias("bad_seconds")))
    last = cs.group_by(["match_key", "round", "side"]).agg(pl.col("second").max().alias("last_second"))
    span = first.join(last, on=["match_key", "round", "side"], how="left")
    summary["bad_sides"] = span.height
    summary["bad_sides_bad_until_round_end_share"] = float(
        span.with_columns((pl.col("bad_seconds") >= (pl.col("last_second") - pl.col("first_bad") + 1)).cast(pl.Float64))["bad_seconds"].mean()) \
        if span.height else None
    onset = (cs.join(first, on=["match_key", "round", "side"]).filter(pl.col("second") == pl.col("first_bad")))
    onset = onset.with_columns((pl.col("observed") - pl.col("obs_prev")).alias("obs_step"),
                               (pl.col("expected") - pl.col("exp_prev")).alias("exp_step"))
    summary["onset_observed_dropped_without_expected_drop_share"] = float(
        onset.filter((pl.col("obs_step") < pl.col("exp_step")).fill_null(False)).height / onset.height) if onset.height else None
    death_at = (deaths_ceil.group_by(["match_key", "round", "second", "victim_side"]).agg(pl.len().alias("n_deaths"))
                .rename({"victim_side": "side"}).with_columns(pl.col("round").cast(chk.schema["round"]),
                                                              pl.col("second").cast(chk.schema["second"]),
                                                              pl.col("side").cast(chk.schema["side"])))
    summary["onset_second_equals_death_second_share"] = float(
        onset.join(death_at, on=["match_key", "round", "second", "side"], how="left").filter(pl.col("n_deaths").is_not_null()).height
        / onset.height) if onset.height else None

    # Round-level death counts: deaths.parquet against the rounds table, for mismatch rounds and for all sampled rounds.
    rd = pl.read_parquet(ROOT + "/data/combined/rounds.parquet", columns=["match_key", "round", "n_deaths", "deaths_unresolved",
                                                                          "deaths_t", "deaths_ct"]).filter(pl.col("match_key").is_in(sample))
    rd = rd.with_columns(pl.col("round").cast(pl.Int64))
    dcount = deaths_ceil.with_columns(pl.col("round").cast(pl.Int64)).group_by(["match_key", "round"]).agg(pl.len().alias("deaths_in_events"))
    rd = rd.join(dcount, on=["match_key", "round"], how="left").with_columns(pl.col("deaths_in_events").fill_null(0))
    mis_keys = bad_rounds.select(["match_key", pl.col("round").cast(pl.Int64)])
    rd = rd.join(mis_keys.with_columns(pl.lit(True).alias("mismatch")), on=["match_key", "round"], how="left").with_columns(
        pl.col("mismatch").fill_null(False))
    summary["rounds_with_unresolved_deaths_share"] = {
        "mismatch_rounds": float(rd.filter(pl.col("mismatch"))["deaths_unresolved"].fill_null(0).gt(0).mean()) if rd.filter(pl.col("mismatch")).height else None,
        "all_sampled_rounds": float(rd["deaths_unresolved"].fill_null(0).gt(0).mean())}
    summary["events_minus_rounds_death_count_mismatch_rounds"] = float(
        (rd.filter(pl.col("mismatch")).with_columns((pl.col("deaths_in_events") - pl.col("n_deaths")).alias("d"))["d"] != 0).mean()) \
        if rd.filter(pl.col("mismatch")).height else None
    summary["events_minus_rounds_death_count_all_rounds"] = float(
        (rd.with_columns((pl.col("deaths_in_events") - pl.col("n_deaths")).alias("d"))["d"] != 0).mean())
    summary["near_death"] = {
        "death_now": int(bad.filter(pl.col("deaths_now") > 0).height),
        "death_prev_second": int(bad.filter(pl.col("deaths_prev") > 0).height),
        "death_next_second": int(bad.filter(pl.col("deaths_next") > 0).height),
        "no_death_nearby": int(bad.filter((pl.col("deaths_now") + pl.col("deaths_prev") + pl.col("deaths_next")) == 0).height)}
    summary["duplicate_player_rows_at_mismatch"] = int(bad.filter(pl.col("dup_players") > 0).height)
    summary["roster_defect_round"] = int(bad.filter(pl.col("missing_side").is_not_null()).height)
    summary["swap_boundary_round"] = int(bad.filter(pl.col("swap_boundary").is_not_null()).height)
    summary["duplicate_event_group_round"] = int(bad.filter(pl.col("dup_group_size") > 1).height)
    summary["final_differs_from_recorded_round"] = int(bad.filter(pl.col("final_differs_from_recorded").fill_null(False)).height)
    summary["mismatch_rounds"] = bad.select(["match_key", "round"]).unique().height
    summary["mismatch_matches"] = bad["match_key"].n_unique()
    dist = bad.group_by(["diff"]).agg(pl.len().alias("pairs")).sort("pairs", descending=True).head(10).to_dicts()
    summary["diff_distribution"] = dist

    # Death convention: the same table with death seconds floored instead of ceiled.
    chk_f = checked_table(positions, deaths_floor.select(["match_key", "round", "second", "victim_side"]), snap, "floor")
    summary["agree_share_floor_convention"] = 1 - chk_f.filter(pl.col("diff") != 0).height / chk_f.height

    # Rounds with a roster defect and rounds whose recorded and final winner disagree, excluded in turn.
    keep = chk.join(R, on=["match_key", "round"], how="left").filter(pl.col("missing_side").is_null())
    summary["agree_share_without_roster_defect"] = 1 - keep.filter(pl.col("diff") != 0).height / keep.height
    # The residual after the three exclusions the gate report uses (roster defect, duplicate-event group, recorded vs final winner).
    clean = keep.filter((pl.col("dup_group_size").fill_null(1) <= 1) & (~pl.col("final_differs_from_recorded").fill_null(False)))
    summary["agree_share_clean_rounds"] = 1 - clean.filter(pl.col("diff") != 0).height / clean.height
    summary["clean_rounds_mismatch_rounds"] = clean.filter(pl.col("diff") != 0).select(["match_key", "round"]).unique().height

    res = {"sample_matches": len(sample), "summary": summary, "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
