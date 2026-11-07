import json, glob
import numpy as np, polars as pl
B = "G:/pureskill/csds-cs2/data"
TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
R = json.load(open(f"{B}/viz/V1_verify_results.json"))
print("== diag per qualifying state: model LL vs non-test base-rate LL vs in-sample entropy")
for r in R["diag_state_baseline"]:
    print(f"ct{r['alive_ct']} t{r['alive_t']} rows={r['rows']} model={r['model_row_logloss']:.4f} base={r['nontest_base_rate_logloss']:.4f} entropy={r['in_sample_state_entropy_nats']:.4f} nontest_rate={r['nontest_base_rate']:.3f}")
print("row-level model LL", round(R["diag_row_level_model_mean_logloss"],4), "baseline", round(R["diag_row_level_baseline_mean_logloss"],4))
print("== model mean predicted P(CT) vs empirical rate, qualifying states")
for r in R["state_table_test"]:
    if r["qualifies"] and r["alive_ct"] in (4,5) or (r["qualifies"] and r["alive_t"]==5):
        ci = r["ct_win_rate_ci95_cluster"]
        flag = "" if ci[0] <= r["mean_pred_pct_ct"] <= ci[1] else " <-- pred outside CI"
        print(f"ct{r['alive_ct']} t{r['alive_t']} emp={r['ct_win_rate']:.4f} ci=[{ci[0]:.4f},{ci[1]:.4f}] pred={r['mean_pred_pct_ct']:.4f}{flag}")

# 1) round split vs test-prediction membership
rd = pl.read_parquet(f"{B}/state_sample/rounds.parquet", columns=["match_key","split"]).unique()
test_keys = pl.read_parquet(f"{B}/viz/per_tick_test_preds.parquet", columns=["match_key"]).unique()
x = rd.join(test_keys.with_columns(pl.lit(True).alias("in_grid_test")), on="match_key", how="left").with_columns(pl.col("in_grid_test").fill_null(False))
print("== state-sample split vs grid test membership (matches)")
print(x.group_by(["split","in_grid_test"]).len().sort(["split","in_grid_test"]))

# 2) 1 Hz spacing check in grid
pt = pl.read_parquet(f"{B}/viz/per_tick_states.parquet", columns=["match_key","round","tick","seconds_since_round_live"])
pt = pt.sort(["match_key","round","tick"]).with_columns(
    (pl.col("tick") - pl.col("tick").shift(1).over(["match_key","round"])).alias("dt"))
dts = pt.filter(pl.col("dt").is_not_null())["dt"].value_counts().sort("count", descending=True).head(5)
print("== grid tick deltas within round (top):", dts.to_dicts())
print("seconds diff vs tick/64 check: max |sec - (tick-live0)/64|:")
live0 = pt.filter(pl.col("seconds_since_round_live")==0).group_by(["match_key","round"]).agg(pl.col("tick").min().alias("lt"))
chk = pt.join(live0, on=["match_key","round"]).with_columns(((pl.col("tick")-pl.col("lt"))/64.0 - pl.col("seconds_since_round_live")).abs().alias("e"))
print(float(chk["e"].max()))

# 3) post-round deaths sensitivity for death means
def load_pages(ch, cols):
    files = sorted(glob.glob(f"{TOME}/basic_{ch}.COMBINED/dataframe_*"))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
re26 = load_pages("round_end", ["match_key","round","tick","winner_team_code"]).filter(pl.col("match_key").str.starts_with("csds/2026/"))
de = load_pages("player_death", ["match_key","round","tick","player_id","player_team_code"]).filter(pl.col("match_key").str.starts_with("csds/2026/"))
live = pt.group_by(["match_key","round"]).agg(pl.col("tick").filter(pl.col("seconds_since_round_live")==0).min().alias("live_tick"))
rt = live.join(re26.rename({"tick":"end_tick"}), on=["match_key","round"], how="inner")
dd = de.join(rt, on=["match_key","round"], how="inner").filter(pl.col("tick")>=pl.col("live_tick"))
dd = dd.unique(["match_key","round","player_id"], keep="first")
inc_post = dd.group_by(["match_key","round","player_team_code"]).len()
pv = rt.select(["match_key","round","winner_team_code"]).join(
    dd.filter(pl.col("tick")<=pl.col("end_tick")).group_by(["match_key","round","player_team_code"]).len().pivot(on="player_team_code", index=["match_key","round"], values="len").fill_null(0),
    on=["match_key","round"], how="left").fill_null(0)
pa = rt.select(["match_key","round","winner_team_code"]).join(
    dd.group_by(["match_key","round","player_team_code"]).len().pivot(on="player_team_code", index=["match_key","round"], values="len").fill_null(0),
    on=["match_key","round"], how="left").fill_null(0)
def means(t):
    w = t["winner_team_code"].to_numpy(); ct = t["3"].to_numpy(); tt = t["2"].to_numpy()
    wd = np.where(w==3, ct, tt); ld = np.where(w==3, tt, ct)
    return round(float(wd.mean()),4), round(float(ld.mean()),4)
print("== death means in-window (end_tick inclusive):", means(pv))
print("== death means if post-round deaths also counted:", means(pa))
