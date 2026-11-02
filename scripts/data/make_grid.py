"""Per-second (1 Hz) state grid for every used round (rounds.parquet use == True).

One row per second k = 0, 1, 2 ... of live play: tick = freeze_end + 64 k, strictly before the effective end of the round
(the decisive event: bomb outcome or the death that wipes a side; the recorded round_end tick otherwise). Alive counts and
the bomb state are piecewise constant between events; the two time features advance every second.

Features (model inputs): alive_ct, alive_t, ct_alive_minus_t, bomb_planted, seconds_since_round_live, seconds_since_plant.
Context columns kept for ablations only: round_number, is_pistol_round, is_overtime, score_lead_diag (freeze-end score
difference, a diagnostic: it is not a model feature).

  pixi run python -I -X utf8 scripts\\data\\make_grid.py [--store combined]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
T, CT = 2, 3
TICK = 64
BIG = 1 << 31


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--end", default="eff", choices=["eff", "recorded"], help="effective end (default) or the recorded round_end tick")
    ap.add_argument("--offset", type=int, default=0,
                    help="tick phase inside the second: 0 = the training grid; 1..63 shifts every point by that many ticks; "
                         "-1 draws a random phase per round (seed 0). Used to test that the model works at every tick, not only at whole seconds")
    args = ap.parse_args()
    d = os.path.join(ROOT, "data", args.store)
    t0 = time.time()
    R = pl.read_parquet(os.path.join(d, "rounds.parquet")).filter(pl.col("use")).sort(["match_key", "round"])
    M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "map_name", "upload_day", "max_rounds_final")
    R = R.join(M.select("match_key", "map_name", "upload_day"), on="match_key", how="left")
    DE = pl.read_parquet(os.path.join(d, "deaths.parquet"))
    nR = R.height
    R = R.with_columns(pl.int_range(pl.len()).alias("ridx"))
    DE = DE.join(R.select("match_key", "round", "ridx"), on=["match_key", "round"], how="inner").filter(pl.col("side_ok"))

    freeze = R["freeze"].to_numpy().astype(np.int64)
    end = (R["end_eff"] if args.end == "eff" else R["end"]).to_numpy().astype(np.int64)
    off = np.random.default_rng(0).integers(0, TICK, size=nR) if args.offset < 0 else np.full(nR, args.offset % TICK, dtype=np.int64)
    # Round clock: a round that is still unplanted cannot last past 115 s of live play (2026 competitive: round_time 1.92 min). 2024 rounds
    # that ended by time have a late-logged round_end (up to ~8 s), which would add rows from 115 s to 123 s that are AFTER the end of the
    # round; cut them. Rounds with a plant keep every row up to their decisive event.
    plant0 = R["plant_tick"].fill_null(-1).to_numpy().astype(np.int64)
    end = np.where(plant0 >= 0, end, np.minimum(end, freeze + 116 * TICK))
    n = np.maximum(0, -(-(end - freeze - off) // TICK)).astype(np.int64)  # ceil((end - freeze - phase) / 64) points
    total = int(n.sum())
    ridx = np.repeat(np.arange(nR, dtype=np.int64), n)
    start = np.cumsum(n) - n
    k = np.arange(total, dtype=np.int64) - np.repeat(start, n)
    ts = freeze[ridx] + TICK * k + off[ridx]

    def count_deaths(side):
        s = DE.filter(pl.col("side") == side)
        keys = np.sort(s["ridx"].to_numpy().astype(np.int64) * BIG + s["tick"].to_numpy().astype(np.int64))
        hi = np.searchsorted(keys, ridx * BIG + ts, side="right")
        lo = np.searchsorted(keys, ridx * BIG, side="left")
        return hi - lo

    dct, dt = count_deaths(CT), count_deaths(T)
    ac, at = 5 - dct, 5 - dt
    plant = R["plant_tick"].fill_null(-1).to_numpy().astype(np.int64)[ridx]
    planted = (plant >= 0) & (ts >= plant)
    mr = R["max_rounds_final"].to_numpy().astype(np.int64)[ridx]
    rnd = R["round"].to_numpy().astype(np.int64)[ridx]
    lead = (R["ct_sc"].fill_null(0) - R["t_sc"].fill_null(0)).to_numpy().astype(np.int64)[ridx]
    label = (R["final"].to_numpy().astype(np.int64) == CT).astype(np.int8)[ridx]
    G = pl.DataFrame({
        "match_key": R["match_key"].to_numpy()[ridx], "round": rnd, "tick": ts, "second": k.astype(np.int32),
        "alive_ct": ac.astype(np.int8), "alive_t": at.astype(np.int8), "ct_alive_minus_t": (ac - at).astype(np.int8),
        "bomb_planted": planted.astype(np.int8), "seconds_since_round_live": ((ts - freeze[ridx]) / TICK).astype(np.float32),
        "seconds_since_plant": np.where(planted, (ts - plant) / TICK, -1.0).astype(np.float32),
        "round_number": rnd.astype(np.int16), "is_pistol_round": ((rnd == 1) | (rnd == mr // 2 + 1)).astype(np.int8),
        "is_overtime": (rnd > mr).astype(np.int8), "score_lead_diag": lead.astype(np.int16), "label_ct_win": label,
        "w_round": (1.0 / np.maximum(n, 1))[ridx].astype(np.float32),
        "era": R["era"].to_numpy()[ridx], "build_num": R["build_num"].to_numpy().astype(np.int32)[ridx],
        "map_name": R["map_name"].to_numpy()[ridx], "upload_day": R["upload_day"].to_numpy()[ridx],
        "label_source": R["label_source"].to_numpy()[ridx],
    })
    name = "grid.parquet" if args.end == "eff" else "grid_recorded_end.parquet"
    if args.offset != 0:
        name = "grid_offset_random.parquet" if args.offset < 0 else f"grid_offset{args.offset % TICK}.parquet"
    out = os.path.join(d, name)
    G.write_parquet(out, compression="zstd")
    print(f"rounds {nR}  matches {R['match_key'].n_unique()}  grid rows {G.height}  mean points/round {G.height / nR:.1f}  "
          f"CT-win share {float(G['label_ct_win'].mean()):.4f}  ({time.time() - t0:.0f}s) -> {out}")
    print(G.group_by("era").agg(pl.len().alias("rows"), pl.col("match_key").n_unique().alias("matches"),
                                pl.col("label_ct_win").mean().alias("ct_share")).sort("era"))


if __name__ == "__main__":
    main()
