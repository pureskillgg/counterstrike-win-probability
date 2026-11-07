import json, os
import polars as pl
import numpy as np
VIZ = r"G:\pureskill\csds-cs2\data\viz"
R = json.load(open(os.path.join(VIZ, "VT3_timeexpiry_indep_results.json"), encoding="utf-8"))
print("FULL per-second unplanted (sec, n, obs):")
for r in R["full_grid_unplanted_per_second"]:
    if r["sec"] >= 104:
        print(r)
print("TEST per-second unplanted (sec, n, obs, pred):")
for r in R["test_unplanted_per_second"]:
    if r["sec"] >= 104:
        print(r)
print("round-weighted overall unplanted90:", R["part2_full_grid_unplanted_ge90"]["all_90plus"]["round_weighted_obs_ct_rate"])
print("n_rounds unplanted90 all:", R["part2_full_grid_unplanted_ge90"]["all_90plus"]["n_rounds"])
print("grid maxsec quantiles:", R["grid_maxsec_quantiles_all_rounds"])
print("rounds reason by code:", R["rounds_reason_by_code_all_grid"])
print("re table:", R["re_2026_ten_reason_winner_table"])
# alive consistency check on grid (my own), not in the main script
g = pl.read_parquet(r"G:\pureskill\csds-cs2\data\viz\per_tick_states.parquet", columns=["alive_ct","alive_t","ct_alive_minus_t","bomb_planted","seconds_since_round_live","match_key","round","tick","label_ct_win"])
print("ct_alive_minus_t == alive_ct - alive_t mismatches:", g.filter(pl.col("ct_alive_minus_t") != pl.col("alive_ct") - pl.col("alive_t")).height, "of", g.height)
print("alive_ct range", g["alive_ct"].min(), g["alive_ct"].max(), "alive_t range", g["alive_t"].min(), g["alive_t"].max())
