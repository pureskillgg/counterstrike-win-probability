"""Alive-count gate for 1 Hz extraction: living players per side from positions_1hz must equal 5 minus the side's cumulative deaths.

The gate runs on 5-versus-5 matches only (player_count 10). Per snapshot (match_key, round, second) and side:

  expected alive = 5 - number of that side's player deaths with death second <= this second
  observed alive = number of that side's rows in positions_1hz at this second

Pass: at least 99.9% of (snapshot, side) pairs agree. The gate stops the 2024 run before any full extraction if it fails.
The deaths come from the tome death events (basic_player_death), mapped to the 1 Hz second of the grid.

  from alive_gate import alive_gate
  result = alive_gate(positions, deaths, snapshots)

positions: match_key, round, second, side (2 or 3), player_id
deaths:    match_key, round, second (second of the death), victim_side (2 or 3)
snapshots: match_key, round, second (the grid rows that should be checked)
"""
import polars as pl

THRESHOLD = 0.999
SIDES = [2, 3]


def alive_gate(positions: pl.DataFrame, deaths: pl.DataFrame, snapshots: pl.DataFrame) -> dict:
    snaps = snapshots.select("match_key", "round", "second").unique()
    sides = pl.DataFrame({"side": SIDES}).with_columns(pl.lit(1).alias("_k"))
    grid = snaps.with_columns(pl.lit(1).alias("_k")).join(sides, on="_k").drop("_k")
    obs = (positions.group_by(["match_key", "round", "second", "side"]).agg(pl.len().alias("observed")))
    dead = (deaths.group_by(["match_key", "round", "second", "victim_side"]).agg(pl.len().alias("n_deaths"))
            .rename({"victim_side": "side"}))
    # cumulative deaths up to each second, per match, round and side
    cum_rows = []
    for side in SIDES:
        d = dead.filter(pl.col("side") == side).select("match_key", "round", "second", "n_deaths")
        cum = (grid.filter(pl.col("side") == side).select("match_key", "round", "second")
               .join(d, on=["match_key", "round", "second"], how="left")
               .with_columns(pl.col("n_deaths").fill_null(0)))
        cum = cum.sort(["match_key", "round", "second"]).with_columns(
            pl.col("n_deaths").cum_sum().over(["match_key", "round"]).alias("cum_deaths"), pl.lit(side).alias("side"))
        cum_rows.append(cum.select("match_key", "round", "second", "side", "cum_deaths"))
    cum_all = pl.concat(cum_rows)
    checked = (grid.join(cum_all, on=["match_key", "round", "second", "side"], how="left")
               .join(obs, on=["match_key", "round", "second", "side"], how="left")
               .with_columns(pl.col("observed").fill_null(0),
                             (5 - pl.col("cum_deaths").fill_null(0)).alias("expected")))
    checked = checked.with_columns((pl.col("observed") == pl.col("expected")).alias("agree"))
    n = checked.height
    agree = int(checked["agree"].sum()) if n else 0
    share = agree / n if n else 0.0
    mism = checked.filter(~pl.col("agree")).select("match_key", "round", "second", "side", "observed", "expected").head(20)
    return {"pairs": n, "agree": agree, "share": share, "threshold": THRESHOLD, "pass": bool(n and share >= THRESHOLD),
            "mismatch_sample": mism.to_dicts()}


if __name__ == "__main__":
    import sys
    print("alive_gate: call alive_gate(positions, deaths, snapshots) from a script; see test_2024_pipeline.py for the contract.")
    sys.exit(0)
