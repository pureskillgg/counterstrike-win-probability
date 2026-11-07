import sys
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "G:/pureskill/csds-cs2"
K = "csds/2026/10/11/ad02c41a-b6e5-4631-9db0-a8b9ec8d6d7c/csds"
R, TICK = 3, 17056

pos = pl.read_parquet(f"{ROOT}/data/positions_1hz/2026-10-11.parquet").filter((pl.col("match_key") == K) & (pl.col("round") == R) & (pl.col("tick") == TICK)).sort("player_id")
print("laptop positions_1hz dtypes:", dict(zip(pos.columns, [str(t) for t in pos.dtypes])))
for r in pos.iter_rows(named=True):
    x, y, yaw = np.float32(r["x"]), np.float32(r["y"]), np.float32(r["yaw"])
    print(f"PID {r['player_id']} side {r['side']}: x={float(x)!r} y={float(y)!r} yaw={float(yaw)!r} (f32 hex yaw {yaw.tobytes().hex()})")

# MPF-style float32 margin for CT 8 vs T 7 on the laptop values
d = {r["player_id"]: r for r in pos.iter_rows(named=True)}
cx, cy, cyaw = np.float32(d[8]["x"]), np.float32(d[8]["y"]), np.float32(d[8]["yaw"])
tx, ty, tyaw = np.float32(d[7]["x"]), np.float32(d[7]["y"]), np.float32(d[7]["yaw"])
dx32, dy32 = tx - cx, ty - cy
brg32 = np.degrees(np.arctan2(dy32, dx32))
f32 = np.float32
m = np.abs(((cyaw - brg32 + f32(180)) % f32(360)) - f32(180)) - f32(45)
print("laptop float32 CT8->T7: brg", float(brg32), " margin", float(m), " (<0 means inside cone)")
dx64, dy64 = float(tx) - float(cx), float(ty) - float(cy)
brg64 = np.degrees(np.arctan2(dy64, dx64))
print("laptop float64 CT8->T7: margin", abs(((float(cyaw) - brg64 + 180) % 360) - 180) - 45)
