import os, glob, json
import polars as pl
BASE = r"G:\pureskill\csds-cs2"
VIZ = os.path.join(BASE, "data", "viz")
RE_DIR = os.path.join(BASE, "tomes", "combined_test", "tome", "csds", "basic_round_end.COMBINED")
OUT = os.path.join(VIZ, "VT3_testcover_results.json")
files = sorted(glob.glob(os.path.join(RE_DIR, "dataframe_*")))
re_all = pl.concat([pl.read_parquet(f, columns=["match_key","round","win_reason_code","player_count"]) for f in files], how="vertical_relaxed")
re10 = re_all.filter(pl.col("match_key").str.starts_with("csds/2026/") & (pl.col("player_count") == 10)).select(["match_key","round","win_reason_code"])
preds = pl.read_parquet(os.path.join(VIZ, "per_tick_test_preds.parquet"), columns=["match_key","round","tick","seconds_since_round_live","p_xgb_grid","label_ct_win"])
test_rounds = preds.select(["match_key","round"]).unique()
tr = test_rounds.join(re10, on=["match_key","round"], how="left")
X = {}
X["test_split_rounds"] = tr.height
X["test_split_reason12_rounds"] = tr.filter(pl.col("win_reason_code") == 12).height
X["test_split_matches"] = int(preds["match_key"].n_unique())
r12 = tr.filter(pl.col("win_reason_code") == 12).select(["match_key","round"])
pr = preds.join(r12, on=["match_key","round"], how="inner")
X["test_states_in_reason12_rounds"] = pr.height
X["test_states_in_reason12_rounds_ge90"] = pr.filter(pl.col("seconds_since_round_live") >= 90).height
X["test_states_in_reason12_rounds_ge110"] = pr.filter(pl.col("seconds_since_round_live") >= 110).height
X["test_reason12_mean_pred_ge110"] = float(pr.filter(pl.col("seconds_since_round_live") >= 110)["p_xgb_grid"].mean())
json.dump(X, open(OUT, "w", encoding="utf-8"), indent=1)
for k, v in X.items():
    print(k, v)
