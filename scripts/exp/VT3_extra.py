import os, glob, json
import numpy as np
import polars as pl
BASE = r"G:\pureskill\csds-cs2"
VIZ = os.path.join(BASE, "data", "viz")
RE_DIR = os.path.join(BASE, "tomes", "combined_test", "tome", "csds", "basic_round_end.COMBINED")
OUT = os.path.join(VIZ, "VT3_extra_results.json")
X = {}
g = pl.read_parquet(os.path.join(VIZ, "per_tick_states.parquet"), columns=["match_key","round","tick","seconds_since_round_live","bomb_planted","label_ct_win","ct_alive_minus_t"])
g = g.with_columns(pl.col("seconds_since_round_live").round(0).cast(pl.Int64).alias("sec")).sort(["match_key","round","tick"])
# a) monotonicity of bomb_planted within round
g = g.with_columns(pl.col("bomb_planted").shift(1).over(["match_key","round"]).alias("prev"))
X["planted_1_to_0_transitions"] = g.filter((pl.col("prev") == 1) & (pl.col("bomb_planted") == 0)).height
# round-level flags
rt = g.group_by(["match_key","round"]).agg([
    pl.col("sec").max().alias("maxsec"),
    pl.col("bomb_planted").max().alias("ever_planted"),
    pl.col("label_ct_win").max().alias("y"),
])
# b) never planted and reaches 110
np_ = rt.filter((pl.col("ever_planted") == 0) & (pl.col("maxsec") >= 110))
X["never_planted_reach110_rounds"] = np_.height
X["never_planted_reach110_ct_rate"] = float(np_["y"].mean())
X["never_planted_reach110_matches"] = int(np_["match_key"].n_unique())
X["never_planted_total_rounds"] = int(rt.filter(pl.col("ever_planted") == 0).height)
# c) exclusion of ten-player round_end rows from grid
files = sorted(glob.glob(os.path.join(RE_DIR, "dataframe_*")))
re_all = pl.concat([pl.read_parquet(f, columns=["match_key","round","tick","win_reason_code","winner_team_code","player_count"]) for f in files], how="vertical_relaxed")
re10 = re_all.filter(pl.col("match_key").str.starts_with("csds/2026/") & (pl.col("player_count") == 10))
gr = g.select(["match_key","round"]).unique()
ex = re10.join(gr, on=["match_key","round"], how="anti")
X["ten_player_rounds_not_in_grid"] = ex.height
X["ten_player_rounds_not_in_grid_reason12"] = ex.filter(pl.col("win_reason_code") == 12).height
X["ten_player_rounds_in_grid_reason12"] = re10.join(gr, on=["match_key","round"], how="inner").filter(pl.col("win_reason_code") == 12).height
X["grid_share_of_ten_player_rounds"] = gr.height / re10.height
# d) test-split bootstrap for 112-114 window and 110-114 window
preds = pl.read_parquet(os.path.join(VIZ, "per_tick_test_preds.parquet"), columns=["match_key","tick","label_ct_win","p_xgb_grid"])
tkeys = sorted(preds["match_key"].unique().to_list())
pj = preds.join(g.select(["match_key","tick","sec","bomb_planted"]), on=["match_key","tick"], how="inner").filter((pl.col("bomb_planted")==0) & (pl.col("sec")>=110) & (pl.col("sec")<=114))
mi = {k:i for i,k in enumerate(tkeys)}
M = len(tkeys)
def boot_diff(df, seed=0, B=500):
    idx = df["match_key"].replace_strict(mi, default=None).to_numpy().astype(np.int64)
    y = df["label_ct_win"].to_numpy().astype(float)
    p = df["p_xgb_grid"].to_numpy().astype(float)
    n_m = np.bincount(idx, minlength=M).astype(float)
    y_m = np.bincount(idx, weights=y, minlength=M)
    p_m = np.bincount(idx, weights=p, minlength=M)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, M, size=(B, M))
    mult = np.zeros((B, M))
    for b in range(B):
        mult[b] = np.bincount(draws[b], minlength=M)
    bn = mult @ n_m; by = mult @ y_m; bp = mult @ p_m
    d = by/bn - bp/bn
    return {"n": int(len(y)), "matches": int((n_m>0).sum()), "obs": float(y.mean()), "pred": float(p.mean()), "obs_minus_pred": float(y.mean()-p.mean()), "ci": [float(np.percentile(d,2.5)), float(np.percentile(d,97.5))]}
X["test_112_114_unplanted"] = boot_diff(pj.filter(pl.col("sec")>=112))
X["test_110_114_unplanted"] = boot_diff(pj)
X["test_114_only_unplanted"] = boot_diff(pj.filter(pl.col("sec")==114))
json.dump(X, open(OUT, "w", encoding="utf-8"), indent=1)
for k, v in X.items():
    print(k, v)
