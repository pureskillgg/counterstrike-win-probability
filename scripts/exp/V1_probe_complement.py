import os
import numpy as np
import polars as pl

PLAYERS = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
ROUNDS = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
THR = 20000
N_BOOT = 500
SEED = 0

rounds = pl.read_parquet(ROUNDS).with_columns(pl.col("round").cast(pl.Int64))
p0 = (pl.scan_parquet(PLAYERS)
      .filter(pl.col("seconds_since_live") == 0)
      .select(["match_key", "round", "team_code", "equipment_value_calc"])
      .collect()
      .with_columns(pl.col("round").cast(pl.Int64)))
ag = p0.group_by(["match_key", "round"]).agg([
    pl.len().alias("n_rows"),
    (pl.col("team_code") == 3).sum().alias("n_ct"),
    (pl.col("team_code") == 2).sum().alias("n_t"),
    pl.col("equipment_value_calc").filter(pl.col("team_code") == 3).sum().alias("ct_eq"),
    pl.col("equipment_value_calc").filter(pl.col("team_code") == 2).sum().alias("t_eq"),
])
j = rounds.join(ag, on=["match_key", "round"], how="left").filter(
    (pl.col("n_rows") == 10) & (pl.col("n_ct") == 5) & (pl.col("n_t") == 5))
mkeys = sorted(j["match_key"].unique().to_list())
mindex = {k: i for i, k in enumerate(mkeys)}
M = len(mkeys)
mi = np.array([mindex[k] for k in j["match_key"].to_list()])
y = j["ct_win"].to_numpy().astype(float)
fvf = ((j["ct_eq"] >= THR) & (j["t_eq"] >= THR)).to_numpy()
non = ~fvf
print("clean rounds:", y.size, "matches:", M)
print("fvf rounds:", int(fvf.sum()), "ct wins:", int(y[fvf].sum()), "rate:", round(float(y[fvf].mean()), 5))
print("non-fvf rounds:", int(non.sum()), "ct wins:", int(y[non].sum()), "rate:", round(float(y[non].mean()), 5))
print("non-fvf minus fvf:", round(float(y[non].mean() - y[fvf].mean()), 5))

rng = np.random.default_rng(SEED)
draws = rng.integers(0, M, size=(N_BOOT, M))
C = np.empty((N_BOOT, M))
for b in range(N_BOOT):
    C[b] = np.bincount(draws[b], minlength=M)
W = C[:, mi]
nf = non.astype(float)
bt_non = (W @ (nf * y)) / (W @ nf)
ff = fvf.astype(float)
bt_fvf = (W @ (ff * y)) / (W @ ff)
diff_bt = bt_non - bt_fvf
print("non-fvf rate CI:", np.percentile(bt_non, [2.5, 97.5]).round(4).tolist())
print("non-fvf minus fvf CI:", np.percentile(diff_bt, [2.5, 97.5]).round(4).tolist())
