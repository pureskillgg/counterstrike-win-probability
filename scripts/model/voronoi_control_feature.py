"""Voronoi-proximity map control per second, de_mirage 2026 (side quest, GPU).

For every second with living players, each cell of a grid over the de_mirage scaling window is assigned to the nearest living player.
The CT control share is the fraction of cells whose nearest living player is a CT (the paper's Voronoi, its best single control
representation: Huang et al. 2026, section 4.2). Cells are 2D; no navigation mesh, so this is the proximity version only.

  ...torch env python.exe -I -X utf8 scripts\\model\\voronoi_control_feature.py
Writes data/diagnostics/voronoi_control_de_mirage.parquet (match_key, round, second, ct_voronoi, n_ct, n_t) and prints a sanity table.
"""
import glob
import json
import os
import sys

import numpy as np
import polars as pl
import torch

ROOT = r"G:\pureskill\csds-cs2"
GRID = 32
BATCH = 4096
CT, T = 3, 2
OUT = os.path.join(ROOT, "data", "diagnostics", "voronoi_control_de_mirage.parquet")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = json.load(open(os.path.join(ROOT, "data", "round_state", "de_mirage.json"), encoding="utf-8"))["window"]
    lo, hi = win["xy_min"], win["xy_max"]
    xs = torch.linspace(lo, hi, GRID, device=dev)
    cells = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).reshape(-1, 2)  # (GRID*GRID, 2)
    mk_map = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "map_name"])
    dm_keys = set(mk_map.filter(pl.col("map_name") == "de_mirage")["match_key"].to_list())
    rounds = pl.read_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet"), columns=["match_key", "round", "final"])
    rounds = rounds.with_columns((pl.col("final") == 3).cast(pl.Int8).alias("ct_win")).select("match_key", "round", "ct_win")
    parts = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "2026-*.parquet"))):
        pos = pl.read_parquet(f, columns=["match_key", "round", "second", "side", "x", "y"]).filter(pl.col("match_key").is_in(list(dm_keys)))
        if pos.height == 0:
            continue
        # one row per living player-second; group the seconds, keep at most 10 players per second
        pos = pos.with_columns(pl.struct(["match_key", "round", "second"]).rank("dense").alias("sid"))
        sec = pos.select("match_key", "round", "second", "sid").unique().sort("sid")
        sid_n = sec.height
        pos = pos.sort("sid").with_columns(pl.int_range(0, pl.len()).over("sid").alias("slot")).filter(pl.col("slot") < 10)
        xy = np.zeros((sid_n + 1, 10, 2), dtype=np.float32)
        side = np.zeros((sid_n + 1, 10), dtype=np.int8)
        valid = np.zeros((sid_n + 1, 10), dtype=bool)
        sids = pos["sid"].to_numpy() - 1
        slots = pos["slot"].to_numpy()
        xy[sids, slots, 0] = pos["x"].to_numpy()
        xy[sids, slots, 1] = pos["y"].to_numpy()
        side[sids, slots] = pos["side"].to_numpy()
        valid[sids, slots] = True
        out_ct = np.full(sid_n, np.nan, dtype=np.float32)
        n_ct = np.zeros(sid_n, dtype=np.int16)
        n_t = np.zeros(sid_n, dtype=np.int16)
        for s0 in range(0, sid_n, BATCH):
            s1 = min(s0 + BATCH, sid_n)
            P = torch.as_tensor(xy[s0:s1], device=dev)            # (B,10,2)
            V = torch.as_tensor(valid[s0:s1], device=dev)          # (B,10)
            S = torch.as_tensor(side[s0:s1], device=dev)           # (B,10)
            d2 = ((cells[None, :, None, :] - P[:, None, :, :]) ** 2).sum(-1)  # (B,cells,10)
            d2 = d2.masked_fill(~V[:, None, :], float("inf"))
            nearest = d2.argmin(-1)                                 # (B,cells)
            has = V.any(-1)
            nearest_side = torch.gather(S[:, None, :].expand(-1, cells.shape[0], -1), 2, nearest[..., None]).squeeze(-1)
            frac = (nearest_side == CT).float().mean(-1)
            frac = torch.where(has, frac, torch.full_like(frac, float("nan")))
            out_ct[s0:s1] = frac.cpu().numpy()
            n_ct[s0:s1] = ((S == CT) & V).sum(-1).cpu().numpy()
            n_t[s0:s1] = ((S == T) & V).sum(-1).cpu().numpy()
        part = sec.with_columns(pl.Series("ct_voronoi", out_ct), pl.Series("n_ct", n_ct), pl.Series("n_t", n_t)).drop("sid")
        parts.append(part)
        print(os.path.basename(f), part.height, "dm seconds", flush=True)
    if not parts:
        raise SystemExit("no de_mirage positions found")
    D = pl.concat(parts).join(rounds, on=["match_key", "round"], how="inner")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    D.write_parquet(OUT, compression="zstd")
    q = D.with_columns(pl.col("ct_voronoi").qcut(5, labels=["q1", "q2", "q3", "q4", "q5"]).alias("quintile"))
    tab = q.group_by("quintile").agg(pl.len().alias("seconds"), pl.col("ct_win").mean().alias("ct_win_rate"),
                                     pl.col("ct_voronoi").mean().alias("mean_ct_voronoi")).sort("quintile")
    print(tab, flush=True)
    print(json.dumps({"seconds": D.height, "rounds": D.select("match_key", "round").unique().height,
                      "device": str(dev), "grid": GRID}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
