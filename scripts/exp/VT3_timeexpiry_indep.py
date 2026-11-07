# VT3 independent verification of the time-expiry claims.
# Written from raw parquet: round_end tome pages, per_tick_states grid, per_tick_test_preds.
# Does NOT read the analyst's N3 script or output.
import os
import glob
import json
import numpy as np
import polars as pl

BASE = r"G:\pureskill\csds-cs2"
VIZ = os.path.join(BASE, "data", "viz")
RE_DIR = os.path.join(BASE, "tomes", "combined_test", "tome", "csds", "basic_round_end.COMBINED")
GRID = os.path.join(VIZ, "per_tick_states.parquet")
PREDS = os.path.join(VIZ, "per_tick_test_preds.parquet")
OUT = os.path.join(VIZ, "VT3_timeexpiry_indep_results.json")

B = 500          # bootstrap resamples
SEED = 0         # rng seed (np.random.default_rng)
CLIP = 1e-6      # ASSUMPTION: clip predicted p to [CLIP, 1-CLIP] before log loss
LAST20_LO = 95   # ASSUMPTION: "final 20 s before expiry" = seconds_since_round_live in [95, 114]
LAST20_HI = 114

R = {}


def J(x):
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    ok = b > 0
    out[ok] = (a / b)[ok] if out.shape else (a / b)
    return out


def ci(boot):
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return [None, None]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def make_mult(M):
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, M, size=(B, M))
    mult = np.zeros((B, M))
    for b in range(B):
        mult[b] = np.bincount(draws[b], minlength=M)
    return mult


def mats_from(df, M, K, cols):
    """Per-match, per-cell sums. df needs mi (match index), cell, and the cols to sum."""
    agg = df.group_by(["mi", "cell"]).agg([pl.len().alias("n")] + [pl.col(c).sum().alias(c) for c in cols])
    mi = agg["mi"].to_numpy()
    ce = agg["cell"].to_numpy()
    out = {"n": np.zeros((M, K))}
    out["n"][mi, ce] = agg["n"].to_numpy().astype(float)
    for c in cols:
        out[c] = np.zeros((M, K))
        out[c][mi, ce] = agg[c].to_numpy().astype(float)
    return out


def boots_from(mats, mult):
    return {k: mult @ v for k, v in mats.items()}


def summarize(mats, bts, mask, with_pred=False):
    n_pt = mats["n"][:, mask].sum()
    y_pt = mats["y"][:, mask].sum()
    bn = bts["n"][:, mask].sum(1)
    by = bts["y"][:, mask].sum(1)
    res = {"n_states": J(n_pt), "obs_ct_rate": J(y_pt / n_pt) if n_pt > 0 else None}
    res["obs_ct_rate_ci"] = ci(safe_div(by, bn))
    if with_pred:
        p_pt = mats["p"][:, mask].sum()
        bp = bts["p"][:, mask].sum(1)
        res["mean_pred_ct"] = J(p_pt / n_pt) if n_pt > 0 else None
        res["mean_pred_ct_ci"] = ci(safe_div(bp, bn))
        res["obs_minus_pred"] = J((y_pt - p_pt) / n_pt) if n_pt > 0 else None
        res["obs_minus_pred_ci"] = ci(safe_div(by - bp, bn))
        ll_pt = mats["ll"][:, mask].sum()
        bl = bts["ll"][:, mask].sum(1)
        res["logloss"] = J(ll_pt / n_pt) if n_pt > 0 else None
        res["logloss_ci"] = ci(safe_div(bl, bn))
        llc_pt = mats["llc"][:, mask].sum()
        blc = bts["llc"][:, mask].sum(1)
        res["logloss_const_baseline"] = J(llc_pt / n_pt) if n_pt > 0 else None
        res["logloss_const_baseline_ci"] = ci(safe_div(blc, bn))
    return res


def cells_states(df):
    return (
        df.with_columns(
            [
                pl.when(pl.col("sec") >= 115).then(pl.lit(5)).otherwise((pl.col("sec") - 90) // 5).cast(pl.Int64).alias("bin"),
                pl.when(pl.col("ct_alive_minus_t") <= -2)
                .then(pl.lit(0))
                .when(pl.col("ct_alive_minus_t") >= 2)
                .then(pl.lit(2))
                .otherwise(pl.lit(1))
                .cast(pl.Int64)
                .alias("cls"),
            ]
        )
        .with_columns((pl.col("bin") * 3 + pl.col("cls")).alias("cell"))
    )


BIN_OF_CELL = np.arange(18) // 3
CLS_OF_CELL = np.arange(18) % 3


def mask_for(kind, val=None):
    if kind == "all":
        return np.ones(18, dtype=bool)
    if kind == "bin":
        return BIN_OF_CELL == val
    if kind == "cls":
        return CLS_OF_CELL == val
    if kind == "bin_cls":
        return np.arange(18) == val[0] * 3 + val[1]
    if kind == "bin_range":
        lo, hi = val
        return (BIN_OF_CELL >= lo) & (BIN_OF_CELL <= hi)
    raise ValueError(kind)


# ---------------------------------------------------------------- Part 0: round_end
files = sorted(glob.glob(os.path.join(RE_DIR, "dataframe_*")))
R["round_end_pages"] = len(files)
re_all = pl.concat(
    [
        pl.read_parquet(f, columns=["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"])
        for f in files
    ],
    how="vertical_relaxed",
)
re26 = re_all.filter(pl.col("match_key").str.starts_with("csds/2026/"))
re10 = re26.filter(pl.col("player_count") == 10)
assert re10.select(["match_key", "round"]).unique().height == re10.height
R["re_2026_all_rounds"] = re26.height
R["re_2026_ten_rounds"] = re10.height
R["re_2026_ten_matches"] = re10["match_key"].n_unique()
re_codes = re10.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"])
R["re_2026_ten_reason_winner_table"] = [
    {"reason": J(r["win_reason_code"]), "msg": r["win_reason_message"], "winner": J(r["winner_team_code"]), "n": J(r["len"])}
    for r in re_codes.to_dicts()
]
r12_re = re10.filter(pl.col("win_reason_code") == 12)
R["re_2026_ten_reason12_count"] = r12_re.height
R["re_2026_ten_reason12_share"] = r12_re.height / re10.height
R["re_2026_ten_reason12_ct_winner_count"] = r12_re.filter(pl.col("winner_team_code") == 3).height

# ---------------------------------------------------------------- Part 0b: grid
gcols = ["match_key", "round", "tick", "seconds_since_round_live", "bomb_planted", "ct_alive_minus_t", "label_ct_win", "w_round"]
g = pl.read_parquet(GRID, columns=gcols)
R["grid_rows"] = g.height
R["grid_matches"] = g["match_key"].n_unique()
frac = (g["seconds_since_round_live"] - g["seconds_since_round_live"].round(0)).abs().max()
assert frac < 1e-9, "seconds_since_round_live not integer"
g = g.with_columns(pl.col("seconds_since_round_live").round(0).cast(pl.Int64).alias("sec"))
R["grid_dup_match_tick"] = g.height - g.select(["match_key", "tick"]).unique().height

g = g.join(
    re10.select(["match_key", "round", pl.col("tick").alias("re_tick"), "winner_team_code", "win_reason_code"]),
    on=["match_key", "round"],
    how="left",
)
R["grid_rows_without_round_end"] = g.filter(pl.col("re_tick").is_null()).height
R["grid_rows_post_round_end"] = g.filter(pl.col("tick") > pl.col("re_tick")).height
R["label_vs_winner_mismatch_rows"] = g.filter(pl.col("label_ct_win") != (pl.col("winner_team_code") == 3).cast(pl.Int64)).height

mkeys = sorted(g["match_key"].unique().to_list())
M = len(mkeys)
mmap = pl.DataFrame({"match_key": mkeys, "mi": list(range(M))}).with_columns(pl.col("mi").cast(pl.Int64))
g = g.join(mmap, on="match_key", how="left")

# ---------------------------------------------------------------- Part 1: round table
rt = g.group_by(["match_key", "round"]).agg(
    [
        pl.len().alias("n"),
        pl.col("sec").min().alias("minsec"),
        pl.col("sec").max().alias("maxsec"),
        pl.col("tick").min().alias("mintick"),
        pl.col("tick").max().alias("maxtick"),
        pl.col("re_tick").first().alias("re_tick"),
        pl.col("win_reason_code").first().alias("reason"),
        pl.col("winner_team_code").first().alias("winner"),
        pl.col("label_ct_win").max().alias("label_max"),
        pl.col("label_ct_win").min().alias("label_min"),
        pl.col("mi").first().alias("mi"),
        (pl.col("sec") >= 110).any().alias("reach110"),
        ((pl.col("sec") == 110) & (pl.col("bomb_planted") == 0)).any().alias("unpl110"),
        ((pl.col("sec") >= 110) & (pl.col("bomb_planted") == 0)).any().alias("unplany110"),
    ]
)
R["grid_rounds"] = rt.height
R["grid_rounds_label_inconsistent"] = rt.filter(pl.col("label_max") != pl.col("label_min")).height
R["grid_rounds_noncontiguous_secs"] = rt.filter(pl.col("n") != (pl.col("maxsec") - pl.col("minsec") + 1)).height
R["grid_rounds_tick_spacing_bad"] = rt.filter((pl.col("maxtick") - pl.col("mintick")) != 64 * (pl.col("maxsec") - pl.col("minsec"))).height

r12 = rt.filter(pl.col("reason") == 12)
R["grid_reason12_count"] = r12.height
R["grid_reason12_share"] = r12.height / rt.height
R["grid_reason12_ct_winner_count"] = r12.filter(pl.col("winner") == 3).height
R["grid_reason12_label_mean"] = float(r12["label_max"].mean())
R["grid_reason12_maxsec_values"] = sorted(set(int(v) for v in r12["maxsec"].to_list()))
R["grid_reason12_time_from_live_start_to_round_end_s"] = {
    "min": float(((r12["re_tick"] - r12["mintick"]) / 64.0).min()),
    "median": float(((r12["re_tick"] - r12["mintick"]) / 64.0).median()),
    "max": float(((r12["re_tick"] - r12["mintick"]) / 64.0).max()),
}
R["grid_reason12_gap_last_grid_tick_to_round_end_s"] = {
    "min": float(((r12["re_tick"] - r12["maxtick"]) / 64.0).min()),
    "median": float(((r12["re_tick"] - r12["maxtick"]) / 64.0).median()),
    "max": float(((r12["re_tick"] - r12["maxtick"]) / 64.0).max()),
}

maxsec_all = rt["maxsec"].to_numpy().astype(float)
R["last_grid_second_median"] = float(np.median(maxsec_all))
R["last_grid_second_p5_linear"] = float(np.percentile(maxsec_all, 5))
R["last_grid_second_p95_linear"] = float(np.percentile(maxsec_all, 95))
R["last_grid_second_mean"] = float(maxsec_all.mean())
R["rounds_with_last_grid_second_exactly_114"] = rt.filter(pl.col("maxsec") == 114).height
R["rounds_with_last_grid_second_exactly_114_by_reason"] = {
    str(int(k)): int(v)
    for k, v in rt.filter(pl.col("maxsec") == 114).group_by("reason").len().iter_rows()
}
R["rounds_reason_by_code_all_grid"] = {
    str(int(k)): int(v) for k, v in rt.group_by("reason").len().iter_rows()
}
R["grid_maxsec_quantiles_all_rounds"] = {
    str(q): float(np.percentile(maxsec_all, q)) for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]
}

# ---------------------------------------------------------------- Part 2: unplanted >= 90 on full grid
mult = make_mult(M)
sub = cells_states(g.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") >= 90)).with_columns(pl.col("w_round").alias("w")))
R["unplanted_ge90_states"] = sub.height
R["unplanted_ge115_states"] = sub.filter(pl.col("sec") >= 115).height
sub = sub.with_columns(
    [
        pl.col("label_ct_win").cast(pl.Float64).alias("y"),
        (pl.col("w") * pl.col("label_ct_win").cast(pl.Float64)).alias("wy"),
        pl.lit(0.0).alias("p"),
        pl.lit(0.0).alias("ll"),
        pl.lit(0.0).alias("llc"),
    ]
)
mats2 = mats_from(sub.select(["mi", "cell", "y", "wy", "w", "p", "ll", "llc"]), M, 18, ["y", "wy", "w"])
bts2 = boots_from(mats2, mult)
part2 = {}
part2["all_90plus"] = summarize(mats2, bts2, mask_for("all"))
part2["all_90plus"]["n_rounds"] = sub.select(pl.concat_str([pl.col("match_key"), pl.col("round").cast(pl.String)], separator="|")).n_unique()
part2["all_90plus"]["round_weighted_obs_ct_rate"] = float(mats2["wy"].sum() / mats2["w"].sum())
bin_names = ["90-94", "95-99", "100-104", "105-109", "110-114", "115+"]
cls_names = ["ct_alive_minus_t<=-2", "-1..1", ">=2"]
for b in range(6):
    part2[f"bin_{bin_names[b]}_all"] = summarize(mats2, bts2, mask_for("bin", b))
    for c in range(3):
        part2[f"bin_{bin_names[b]}_{cls_names[c]}"] = summarize(mats2, bts2, mask_for("bin_cls", (b, c)))
for c in range(3):
    part2[f"cls_{cls_names[c]}_all_bins_90plus"] = summarize(mats2, bts2, mask_for("cls", c))
R["part2_full_grid_unplanted_ge90"] = part2

# per-second observed on full grid, unplanted, 90..114
ps_full = (
    g.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") >= 90) & (pl.col("sec") <= 115))
    .group_by("sec")
    .agg([pl.len().alias("n"), pl.col("label_ct_win").cast(pl.Float64).mean().alias("obs")])
    .sort("sec")
)
R["full_grid_unplanted_per_second"] = [
    {"sec": J(r["sec"]), "n": J(r["n"]), "obs_ct_rate": J(r["obs"])} for r in ps_full.to_dicts()
]

# ---------------------------------------------------------------- Part 3: test-split model
preds = pl.read_parquet(PREDS, columns=["match_key", "tick", "round", "seconds_since_round_live", "label_ct_win", "p_xgb_grid", "p_constant"])
R["preds_rows"] = preds.height
R["preds_dup_match_tick"] = preds.height - preds.select(["match_key", "tick"]).unique().height
test_keys = sorted(preds["match_key"].unique().to_list())
R["preds_test_matches"] = len(test_keys)
R["grid_rows_in_test_matches"] = g.filter(pl.col("match_key").is_in(test_keys)).height
R["p_xgb_grid_min"] = float(preds["p_xgb_grid"].min())
R["p_xgb_grid_max"] = float(preds["p_xgb_grid"].max())

pj = preds.join(
    g.select(["match_key", "tick", "sec", "bomb_planted", "ct_alive_minus_t", "w_round", "label_ct_win"]).rename({"label_ct_win": "glabel"}),
    on=["match_key", "tick"],
    how="inner",
)
R["preds_joined_rows"] = pj.height
R["preds_join_sec_mismatch"] = pj.filter(pl.col("sec") != pl.col("seconds_since_round_live").round(0).cast(pl.Int64)).height
R["preds_join_label_mismatch"] = pj.filter(pl.col("glabel") != pl.col("label_ct_win")).height

tmap = pl.DataFrame({"match_key": test_keys, "mi": list(range(len(test_keys)))}).with_columns(pl.col("mi").cast(pl.Int64))
Mt = len(test_keys)
multt = make_mult(Mt)
tsub = pj.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") >= 90)).join(tmap, on="match_key", how="left")
tsub = cells_states(tsub)
pc = pl.col("p_xgb_grid").cast(pl.Float64).clip(CLIP, 1.0 - CLIP)
pcc = pl.col("p_constant").cast(pl.Float64).clip(CLIP, 1.0 - CLIP)
y = pl.col("label_ct_win").cast(pl.Float64)
tsub = tsub.with_columns(
    [
        y.alias("y"),
        pl.col("p_xgb_grid").cast(pl.Float64).alias("p"),
        (-(y * pc.log() + (1.0 - y) * (1.0 - pc).log())).alias("ll"),
        (-(y * pcc.log() + (1.0 - y) * (1.0 - pcc).log())).alias("llc"),
        pl.lit(0.0).alias("wy"),
        pl.lit(1.0).alias("w"),
    ]
)
R["test_unplanted_ge90_states"] = tsub.height
R["test_unplanted_final20_states"] = tsub.filter((pl.col("sec") >= LAST20_LO) & (pl.col("sec") <= LAST20_HI)).height
R["test_unplanted_final20_p_clipped_count"] = tsub.filter(
    (pl.col("sec") >= LAST20_LO) & (pl.col("sec") <= LAST20_HI) & ((pl.col("p_xgb_grid") < CLIP) | (pl.col("p_xgb_grid") > 1 - CLIP))
).height
R["test_unplanted_final20_p_min"] = float(tsub.filter((pl.col("sec") >= LAST20_LO) & (pl.col("sec") <= LAST20_HI))["p_xgb_grid"].min())
R["test_unplanted_final20_p_max"] = float(tsub.filter((pl.col("sec") >= LAST20_LO) & (pl.col("sec") <= LAST20_HI))["p_xgb_grid"].max())

mats3 = mats_from(tsub.select(["mi", "cell", "y", "p", "ll", "llc", "wy", "w"]), Mt, 18, ["y", "p", "ll", "llc", "wy", "w"])
bts3 = boots_from(mats3, multt)
part3 = {}
part3["all_90plus"] = summarize(mats3, bts3, mask_for("all"), with_pred=True)
for b in range(6):
    part3[f"bin_{bin_names[b]}_all"] = summarize(mats3, bts3, mask_for("bin", b), with_pred=True)
    for c in range(3):
        part3[f"bin_{bin_names[b]}_{cls_names[c]}"] = summarize(mats3, bts3, mask_for("bin_cls", (b, c)), with_pred=True)
part3["final20_95_114_all"] = summarize(mats3, bts3, mask_for("bin_range", (1, 4)), with_pred=True)
for c in range(3):
    part3[f"final20_95_114_{cls_names[c]}"] = summarize(mats3, bts3, mask_for("cls", c) & mask_for("bin_range", (1, 4)), with_pred=True)
R["part3_test_model_unplanted_ge90"] = part3

ps_test = (
    tsub.group_by("sec")
    .agg([pl.len().alias("n"), pl.col("label_ct_win").cast(pl.Float64).mean().alias("obs"), pl.col("p_xgb_grid").cast(pl.Float64).mean().alias("pred")])
    .sort("sec")
)
R["test_unplanted_per_second"] = [
    {"sec": J(r["sec"]), "n": J(r["n"]), "obs_ct_rate": J(r["obs"]), "mean_pred_ct": J(r["pred"])} for r in ps_test.to_dicts()
]

# ---------------------------------------------------------------- Part 4: unplanted rounds reaching 110+
part4 = {}
rt_m = rt.filter(pl.col("mi").is_not_null())
for flag, label in [("reach110", "rounds_reaching_sec_ge110"), ("unpl110", "rounds_unplanted_at_sec110"), ("unplany110", "rounds_with_any_unplanted_state_ge110")]:
    fl = rt_m.filter(pl.col(flag))
    n_arr = np.zeros((M, 1))
    y_arr = np.zeros((M, 1))
    agg = fl.group_by("mi").agg([pl.len().alias("n"), pl.col("label_max").cast(pl.Float64).sum().alias("y")])
    n_arr[agg["mi"].to_numpy(), 0] = agg["n"].to_numpy().astype(float)
    y_arr[agg["mi"].to_numpy(), 0] = agg["y"].to_numpy().astype(float)
    bn = mult @ n_arr
    by = mult @ y_arr
    n_pt = n_arr.sum()
    y_pt = y_arr.sum()
    reasons = fl.group_by("reason").len()
    rs = {str(int(k)): int(v) for k, v in reasons.iter_rows()}
    part4[label] = {
        "n_rounds": int(n_pt),
        "ct_win_rate": float(y_pt / n_pt) if n_pt > 0 else None,
        "ct_win_rate_ci": ci(safe_div(by[:, 0], bn[:, 0])),
        "n_matches": int(fl["mi"].n_unique()),
        "reason_breakdown": rs,
        "ct_wins_by_reason_12": int(fl.filter((pl.col("reason") == 12) & (pl.col("winner") == 3)).height),
        "reason_12_rounds": int(fl.filter(pl.col("reason") == 12).height),
    }
R["part4_rounds_reaching_110"] = part4
R["part4_all_grid_round_ct_rate"] = float(rt["label_max"].mean())

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=1, default=J)

# ---------------------------------------------------------------- console summary
print("OUT", OUT)
print("RE2026 ten rounds", R["re_2026_ten_rounds"], "reason12", R["re_2026_ten_reason12_count"], "share", round(R["re_2026_ten_reason12_share"], 6), "ct_winner_of_r12", R["re_2026_ten_reason12_ct_winner_count"])
print("GRID rows", R["grid_rows"], "matches", R["grid_matches"], "rounds", R["grid_rounds"], "reason12", R["grid_reason12_count"], "share", round(R["grid_reason12_share"], 6))
print("GRID r12 ct winners", R["grid_reason12_ct_winner_count"], "r12 maxsec values", R["grid_reason12_maxsec_values"])
print("GRID r12 time live->end s", R["grid_reason12_time_from_live_start_to_round_end_s"])
print("GRID r12 gap last grid tick->end s", R["grid_reason12_gap_last_grid_tick_to_round_end_s"])
print("last grid second median/p5/p95", R["last_grid_second_median"], R["last_grid_second_p5_linear"], R["last_grid_second_p95_linear"])
print("rounds maxsec==114", R["rounds_with_last_grid_second_exactly_114"], R["rounds_with_last_grid_second_exactly_114_by_reason"])
print("checks: label mismatch", R["label_vs_winner_mismatch_rows"], "no-roundend rows", R["grid_rows_without_round_end"], "post-round rows", R["grid_rows_post_round_end"], "noncontig", R["grid_rounds_noncontiguous_secs"], "tickspacing bad", R["grid_rounds_tick_spacing_bad"], "label inconsistent rounds", R["grid_rounds_label_inconsistent"])
print("unplanted>=90 states", R["unplanted_ge90_states"], "unplanted>=115 states", R["unplanted_ge115_states"])
for k, v in part2.items():
    print("P2", k, "n", v["n_states"], "obs", v["obs_ct_rate"], "ci", v["obs_ct_rate_ci"])
print("test preds rows", R["preds_rows"], "joined", R["preds_joined_rows"], "grid rows in test matches", R["grid_rows_in_test_matches"], "dup", R["preds_dup_match_tick"], "sec mismatch", R["preds_join_sec_mismatch"], "label mismatch", R["preds_join_label_mismatch"])
for k, v in part3.items():
    print("P3", k, "n", v["n_states"], "obs", v["obs_ct_rate"], "pred", v.get("mean_pred_ct"), "obs-pred", v.get("obs_minus_pred"), v.get("obs_minus_pred_ci"), "LL", v.get("logloss"), v.get("logloss_ci"), "LLconst", v.get("logloss_const_baseline"))
print("test final20 p clipped", R["test_unplanted_final20_p_clipped_count"], "p min/max", R["test_unplanted_final20_p_min"], R["test_unplanted_final20_p_max"])
for k, v in part4.items():
    print("P4", k, v)
