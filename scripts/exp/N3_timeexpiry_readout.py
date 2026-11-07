import json
d = json.load(open(r"G:\pureskill\csds-cs2\data\viz\N3_timeexpiry.json", encoding="utf-8"))
print("PER SECOND (last 20 s, unplanted):")
for r in d["sections"]["final20_per_second"]:
    print(r["second"], "grid n=%d obs=%.3f [%.3f,%.3f]" % (r["grid_n_rows"], r["grid_obs_ct_rate"], r["grid_obs_ci_low"], r["grid_obs_ci_high"]),
          "| test n=%d obs=%.3f pred=%.3f diff=%+.3f [%+.3f,%+.3f]" % (r["test_n_rows"], r["test_obs_ct_rate"], r["test_pred_mean"], r["test_diff_obs_minus_pred"], r["test_diff_ci_low"], r["test_diff_ci_high"]))
print("LAST SECOND DIST (sec: all, expiry, planted) for 100..120 and tails:")
ls = d["sections"]["last_second_distribution"]
print([ (int(r["last_sec"]), r["n_all"], r["n_expiry"], r["n_planted"]) for r in ls if 105 <= r["last_sec"] <= 120])
print("tail >=121 planted-only rounds:", sum(r["n_all"] for r in ls if r["last_sec"] >= 121), "sum all:", sum(r["n_all"] for r in ls))
print("rounds with last_sec<=30:", sum(r["n_all"] for r in ls if r["last_sec"] <= 30))
print("REACH 110 by reason:", d["sections"]["unplanted_reach_110_by_reason"])
print("R110 reason-12 expiry:", [n for n in d["numbers"] if "reach 110" in n["name"] or "reaching 110" in n["name"]])
print("CHECKS subset:", {k: d["checks"][k] for k in ["unplanted_rows_sec_ge_115","planted_flag_reverts_count","final20_constant_baseline_rate_nontest","r12_last_sec_unique_values","r12_round_end_minus_tick_min_seconds_unique"]})
print("POOLED 110-115 all-class grid:", [ (r["ct_diff_class"], r["n_rows"], round(r["obs_ct_rate"],4), round(r["obs_ci_low"],4), round(r["obs_ci_high"],4)) for r in d["table"] if r["section"]=="2_grid_all_matches" and r["bin_s"]=="110-115"])
print("TEST 110-115:", [ (r["ct_diff_class"], r["n_rows"], round(r["obs_ct_rate"],3), round(r["pred_mean_p_xgb_grid"],3), round(r["diff_obs_minus_pred"],3), round(r["diff_ci_low"],3), round(r["diff_ci_high"],3)) for r in d["table"] if r["section"]=="3_test_model" and r["bin_s"]=="110-115"])
for n in d["numbers"]:
    print(n["name"][:90], n["value"], n["ci95_low"], n["ci95_high"])
