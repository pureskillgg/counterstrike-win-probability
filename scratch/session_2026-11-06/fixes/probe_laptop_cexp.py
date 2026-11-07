import sys
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "G:/pureskill/csds-cs2"
K = "csds/2026/10/11/m_ca87927edf9c081e6f6ff537/csds"
R, TICK = 3, 17056
EXPOSE_R, SEE_R = 1500.0, 2000.0


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


pos_all = pl.read_parquet(f"{ROOT}/data/positions_1hz/2026-10-11.parquet")
print("positions_1hz rows, all matches:", pos_all.height, "cols:", pos_all.columns)
pos = pos_all.filter((pl.col("match_key") == K) & (pl.col("round") == R) & (pl.col("tick").is_between(TICK - 128, TICK + 128)))
print("positions_1hz rows this match/round, ticks +-128:")
print(pos.sort("tick", "side", "player_id"))

sta = pl.read_parquet(f"{ROOT}/data/status_1hz/2026-10-11.parquet")
print("status_1hz cols:", sta.columns)
st = sta.filter((pl.col("match_key") == K) & (pl.col("round") == R) & (pl.col("tick") == TICK))
print("status_1hz at tick:")
print(st.select([c for c in st.columns if c in ("tick", "player_id", "health", "has_defuser", "is_defusing", "team_code", "side")]).sort("player_id"))

grid = pl.read_parquet(f"{ROOT}/data/combined/grid.parquet", columns=["match_key", "round", "tick", "second", "era", "bomb_planted", "seconds_since_plant", "map_name"])
g = grid.filter((pl.col("match_key") == K) & (pl.col("round") == R) & (pl.col("tick").is_between(TICK - 128, TICK + 128)))
print("grid rows around tick:")
print(g.sort("tick"))

day_new = pl.read_parquet(f"{ROOT}/data/pos_features_1hz/every2_sites_v1/2026-10-11.parquet")
day_old = pl.read_parquet(f"{ROOT}/data/pos_features_1hz/every2/2026-10-11.parquet")
for nm, d in (("every2_sites_v1", day_new), ("every2", day_old)):
    r = d.filter((pl.col("match_key") == K) & (pl.col("round") == R) & (pl.col("tick") == TICK))
    print(nm, "row at tick:", r.select("tick", "c_exposed", "t_exposed", "c_sees_enemy", "t_sees_enemy", "min_enemy_dist", "c_in_contact", "t_in_contact").to_dicts())

# recompute with the laptop definition from the raw positions at this tick
p = pos.filter(pl.col("tick") == TICK).sort("side", "player_id")
print("players at tick (alive per positions_1hz):", p.height)
cs = p.filter(pl.col("side") == 3)
ts = p.filter(pl.col("side") == 2)
print("CT n:", cs.height, "T n:", ts.height)
cx, cy, cyaw = cs["x"].to_numpy().astype("float64"), cs["y"].to_numpy().astype("float64"), cs["yaw"].to_numpy().astype("float64")
tx, ty, tyaw = ts["x"].to_numpy().astype("float64"), ts["y"].to_numpy().astype("float64"), ts["yaw"].to_numpy().astype("float64")
dx = tx[None, :] - cx[:, None]
dy = ty[None, :] - cy[:, None]
D = np.sqrt(dx ** 2 + dy ** 2)
brg = np.degrees(np.arctan2(dy, dx))
c_in_t = np.abs(wrap(cyaw[:, None] - brg)) < 45.0
t_in_c = np.abs(wrap(tyaw[None, :] - (brg + 180.0))) < 45.0
exp_pair = (D < EXPOSE_R) & t_in_c & ~c_in_t
print("c_exposed (laptop rule, positions_1hz):", int(exp_pair.any(1).sum()))
print("c_sees_enemy:", int(((D < SEE_R) & c_in_t).any(1).sum()), " t_sees_enemy:", int(((D < SEE_R) & t_in_c).any(0).sum()))
print("CT ids:", cs["player_id"].to_list(), "T ids:", ts["player_id"].to_list())
print("CT yaw:", np.round(cyaw, 4).tolist())
print("T yaw:", np.round(tyaw, 4).tolist())
np.set_printoptions(linewidth=200, precision=2, suppress=True)
print("D (CT rows x T cols):")
print(np.round(D, 1))
print("brg:")
print(np.round(brg, 2))
print("c_in_t (CT looks at T):")
print(c_in_t.astype(int))
print("t_in_c (T looks at CT):")
print(t_in_c.astype(int))
print("exposed pairs:")
print(exp_pair.astype(int))
