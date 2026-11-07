import sys
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "G:/pureskill/csds-cs2"
K = "csds/2026/10/11/9c525f5e-d246-4d3d-9539-efc1de2d4028/csds"
R, TICK = 7, 49470
pos = pl.read_parquet(f"{ROOT}/data/positions_1hz/2026-10-11.parquet").filter((pl.col("match_key") == K) & (pl.col("round") == R) & (pl.col("tick") == TICK)).sort("side", "player_id")
print("alive players:", pos.height, " CT:", pos.filter(pl.col("side") == 3).height, " T:", pos.filter(pl.col("side") == 2).height)
cs = pos.filter(pl.col("side") == 3); ts = pos.filter(pl.col("side") == 2)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


for dt in (np.float64, np.float32):
    cx, cy, cyaw = (cs[c].to_numpy().astype(dt) for c in ("x", "y", "yaw"))
    tx, ty, tyaw = (ts[c].to_numpy().astype(dt) for c in ("x", "y", "yaw"))
    dx = tx[None, :] - cx[:, None]; dy = ty[None, :] - cy[:, None]
    D = np.sqrt(dx ** 2 + dy ** 2); brg = np.degrees(np.arctan2(dy, dx))
    mc = np.abs(wrap(cyaw[:, None] - brg)) - dt(45.0)
    mt = np.abs(wrap(tyaw[None, :] - (brg + dt(180.0)))) - dt(45.0)
    c_in_t = mc < 0; t_in_c = mt < 0
    exp = (D < 1500.0) & t_in_c & ~c_in_t
    print(f"\n[{dt.__name__}] c_exposed = {int(exp.any(1).sum())}  exposed pairs (CT ids, T ids):",
          [(int(cs['player_id'].to_list()[i]), int(ts['player_id'].to_list()[j])) for i, j in np.argwhere(exp)])
    near = np.argwhere((np.abs(mc) < 0.01) | (np.abs(mt) < 0.01))
    for i, j in near:
        print(f"  near pair CT {int(cs['player_id'].to_list()[i])} -> T {int(ts['player_id'].to_list()[j])}: D={float(D[i, j]):.4f}  margin CT-cone={float(mc[i, j]):.3e} deg  margin T-cone={float(mt[i, j]):.3e} deg")
    print("  distance-to-1500 near pairs:", [(int(cs['player_id'].to_list()[i]), int(ts['player_id'].to_list()[j]), float(D[i, j])) for i, j in np.argwhere(np.abs(D - 1500.0) < 1.0)])
