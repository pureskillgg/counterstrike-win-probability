import os
import glob
import json
import math

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
RE_DIR = os.path.join(ROOT, "tomes", "combined_test", "tome", "csds", "basic_round_end.COMBINED")
GRID_PATH = os.path.join(ROOT, "data", "viz", "per_tick_states.parquet")
PRED_PATH = os.path.join(ROOT, "data", "viz", "per_tick_test_preds.parquet")
OUT_PATH = os.path.join(ROOT, "data", "viz", "N3_timeexpiry.json")
SCRIPT_PATH = os.path.join(ROOT, "scripts", "exp", "N3_timeexpiry_analysis.py")

B = 500
SEED = 0
CLIP = 1e-6
EXPIRY_CODE = 12
WIN_LO, WIN_HI = 95, 115
BIN_LOS = [90, 95, 100, 105, 110, 115, 120]
CLASSES = ["all", "<=-2", "-1..1", ">=2"]

NUMBERS = []
TABLE = []
SECTIONS = {}
CHECKS = {}

ASSUMPTIONS = [
    "ASSUMPTION: round_end rows with player_count == 10 and match_key prefix csds/2026/ are the complete 5v5 rounds (task definition; the model card also filters on player_count == 10).",
    "ASSUMPTION: win_reason_code 12 = #SFUI_Notice_Target_Saved = CT win (notes/spec/feature_spec.md and notes/research/csds_docs_facts.md). Checked in data: every reason-12 row has winner_team_code == 3.",
    "ASSUMPTION: time expiry happens 115.0 s after live start (seconds_since_round_live). Data check: for every grid round with reason 12 the last state is at 114 s and round_end.tick is 64 ticks after it. The 115 s timer is inferred from data in the notes, not checked against game documentation.",
    "ASSUMPTION: seconds_since_round_live is measured from the freeze end (feature spec definition). The notes disagree on freeze time (15 s vs 20 s); this does not change grid-relative timing.",
    "ASSUMPTION: a state is one grid row (1 Hz, integer seconds). Bin CT win rates are row-level means within each bin. Rounds that stay unplanted through a bin contribute more rows to it. w_round is not used: it normalises over the whole round (n_pts), so it would up-weight short rounds within a bin.",
    "ASSUMPTION: uncertainty is a match-cluster bootstrap with B=500 resamples of matches with replacement (multinomial weights from numpy default_rng(0)), ratio estimator sum(numerator)/sum(denominator) per resample, percentile 95% interval. The same resample weights are reused across cells in one population.",
    "ASSUMPTION: the grid population is all 1,521 matches in per_tick_states.parquet (28,683 rounds). The task text says '150+ matches'; the file actually holds 1,521 matches, matching the model card's deduplicated count.",
    "ASSUMPTION: p_xgb_grid is an out-of-sample prediction for the 304 test matches in per_tick_test_preds.parquet. The notes do not document how p_xgb_grid was trained (the model card documents a death-event XGBoost), so its provenance comes from the task text.",
    "ASSUMPTION: 'final 20 seconds before expiry' means states with seconds_since_round_live in [95, 115), i.e. 95..114 s, since the last state of an expiry round is at 114 s.",
    "ASSUMPTION: log loss clips p to [1e-6, 1-1e-6]. The constant baseline is the CT rate of unplanted window states in non-test matches (train plus validation), so it uses no test labels.",
    "ASSUMPTION: an unplanted round 'reaches second 110' when it has a grid row with seconds_since_round_live >= 110 and bomb_planted == 0. The planted flag never reverts within a round (checked), so this equals the first state at or after 110 s being unplanted.",
]

CAVEATS = [
    "round_end has 32,041 2026 10-player rounds; 463 of them are time expiry. 421 of those 463 are in the grid (the model-card dedup and round filters remove the rest), so grid shares and round_end shares differ slightly.",
    "Expiry rounds have no grid state at 115 s (last state is 114 s), so the 115+ bins only contain planted-round states.",
    "The test split has 304 matches and few unplanted states at 110+ s, so per-second cells carry wide intervals.",
    "CT wins in late unplanted states mix time-expiry wins with CT eliminations and defusals and with T wins that occur after a late plant or elimination; the bins do not separate these.",
]


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        return None if (math.isnan(xf) or math.isinf(xf)) else xf
    return x


def add_num(name, value, lo=None, hi=None, n_matches=None, n_rows=None):
    NUMBERS.append({
        "name": name,
        "value": clean(value),
        "ci95_low": clean(lo),
        "ci95_high": clean(hi),
        "n_matches": clean(n_matches),
        "n_rows": clean(n_rows),
    })


def make_weights(m):
    rng = np.random.default_rng(SEED)
    return rng.multinomial(m, np.full(m, 1.0 / m), size=B).astype(np.float64)


def per_match_sums(df, pop_keys, sum_cols):
    pop = pl.DataFrame({"match_key": pop_keys, "_idx": np.arange(len(pop_keys), dtype=np.int64)})
    aggs = [pl.len().alias("_N")] + [pl.col(c).cast(pl.Float64).sum().alias(c) for c in sum_cols]
    agg = df.group_by("match_key").agg(aggs)
    m = pop.join(agg, on="match_key", how="left").sort("_idx")
    N = m["_N"].fill_null(0).cast(pl.Float64).to_numpy()
    S = {c: m[c].fill_null(0.0).cast(pl.Float64).to_numpy() for c in sum_cols}
    return N, S


def boot_ratio(num, den, W):
    bn = W @ num
    bd = W @ den
    ok = bd > 0
    r = bn[ok] / bd[ok]
    n_bad = int((~ok).sum())
    if r.size == 0:
        return float("nan"), float("nan"), n_bad
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi), n_bad


def cell(df, pop_keys, W, sum_cols):
    """Ratio sum(col)/count over rows in df, with match-cluster bootstrap CI.
    Each row is one unit (state, or round when df is round-level)."""
    n_rows = df.height
    out = {"n_rows": n_rows, "n_matches": 0, "n_rounds": 0}
    if n_rows == 0:
        for c in sum_cols:
            out[c] = None
            out[c + "_lo"] = None
            out[c + "_hi"] = None
        out["n_boot_zero_den"] = None
        return out
    out["n_matches"] = df["match_key"].n_unique()
    out["n_rounds"] = df.select(["match_key", "round"]).unique().height
    N, S = per_match_sums(df, pop_keys, sum_cols)
    den_tot = float(N.sum())
    nb_max = 0
    for c in sum_cols:
        num_tot = float(S[c].sum())
        lo, hi, nb = boot_ratio(S[c], N, W)
        out[c] = num_tot / den_tot
        out[c + "_lo"] = lo
        out[c + "_hi"] = hi
        nb_max = max(nb_max, nb)
    out["n_boot_zero_den"] = nb_max
    return out


def bin_filter(df, lo):
    if lo < 120:
        return df.filter((pl.col("sec") >= lo) & (pl.col("sec") < lo + 5))
    return df.filter(pl.col("sec") >= lo)


def bin_label(lo):
    return f"{lo}-{lo + 5}" if lo < 120 else "120+"


def cls_filter(df, cls):
    if cls == "all":
        return df
    return df.filter(pl.col("cls") == cls)


def clip_np(p):
    return np.clip(p, CLIP, 1.0 - CLIP)


def ll_np(y, p):
    pc = clip_np(p)
    return -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))


def main():
    # ---------------- round_end (2026, player_count == 10) ----------------
    pages = sorted(glob.glob(os.path.join(RE_DIR, "dataframe_*")))
    assert len(pages) > 0, "no round_end pages"
    re_all = pl.concat([pl.read_parquet(p) for p in pages], how="diagonal_relaxed")
    re26 = re_all.filter(pl.col("match_key").str.starts_with("csds/2026/"))
    re10 = (re26.filter(pl.col("player_count") == 10)
            .select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message"]))
    CHECKS["round_end_2026_rows_all_counts"] = re26.height
    CHECKS["round_end_2026_matches"] = re26["match_key"].n_unique()
    CHECKS["round_end_10_rows"] = re10.height
    CHECKS["round_end_10_matches"] = re10["match_key"].n_unique()
    CHECKS["round_end_10_unique_match_round"] = re10.select(["match_key", "round"]).unique().height
    re10_pop = sorted(re10["match_key"].unique().to_list())

    rt = re10.group_by(["win_reason_code", "win_reason_message"]).agg(pl.len().alias("n")).sort("win_reason_code")
    reason_table = [
        {"win_reason_code": int(r["win_reason_code"]), "win_reason_message": r["win_reason_message"],
         "n": int(r["n"]), "share": float(r["n"]) / re10.height}
        for r in rt.iter_rows(named=True)
    ]
    SECTIONS["round_end_reason_table_2026_p10"] = reason_table
    r12_winner = re10.filter(pl.col("win_reason_code") == EXPIRY_CODE)["winner_team_code"]
    CHECKS["reason12_rows_winner_ct_count"] = int((r12_winner == 3).sum())
    CHECKS["reason12_rows_total"] = int(r12_winner.len())

    re10 = re10.with_columns((pl.col("win_reason_code") == EXPIRY_CODE).cast(pl.Float64).alias("r12"))
    W_re = make_weights(len(re10_pop))
    c_re = cell(re10, re10_pop, W_re, ["r12"])
    SECTIONS["expiry_share_round_end_2026_p10"] = c_re
    add_num("round_end 2026 10-player rounds: share ending by time expiry (reason 12)",
            c_re["r12"], c_re["r12_lo"], c_re["r12_hi"], c_re["n_matches"], c_re["n_rows"])
    add_num("round_end 2026 10-player rounds: count ending by time expiry (reason 12)",
            int(re10["r12"].sum()), None, None, c_re["n_matches"], c_re["n_rows"])

    # ---------------- grid (1 Hz) ----------------
    grid = pl.read_parquet(GRID_PATH)
    CHECKS["grid_rows"] = grid.height
    CHECKS["grid_matches"] = grid["match_key"].n_unique()
    CHECKS["grid_rounds"] = grid.select(["match_key", "round"]).unique().height
    CHECKS["grid_duplicate_match_round_tick_rows"] = grid.height - grid.select(["match_key", "round", "tick"]).unique().height
    CHECKS["grid_non_integer_seconds"] = grid.filter(
        (pl.col("seconds_since_round_live") - pl.col("seconds_since_round_live").round(0)).abs() > 1e-9).height

    gr = (grid.select(["match_key", "round", "tick", "seconds_since_round_live", "bomb_planted",
                       "ct_alive_minus_t", "label_ct_win"])
          .rename({"seconds_since_round_live": "sec", "label_ct_win": "y"}))
    gr = gr.with_columns(
        pl.when(pl.col("ct_alive_minus_t") <= -2).then(pl.lit("<=-2"))
        .when(pl.col("ct_alive_minus_t") >= 2).then(pl.lit(">=2"))
        .otherwise(pl.lit("-1..1")).alias("cls"))
    grid_pop = sorted(gr["match_key"].unique().to_list())
    W_g = make_weights(len(grid_pop))

    srt = (gr.sort(["match_key", "round", "tick"])
           .with_columns(pl.col("bomb_planted").shift(1).over(["match_key", "round"]).alias("prev_planted")))
    CHECKS["planted_flag_reverts_count"] = srt.filter(
        (pl.col("prev_planted") == 1) & (pl.col("bomb_planted") == 0)).height
    del srt
    CHECKS["unplanted_rows_sec_ge_115"] = gr.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") >= 115)).height

    rnd = (gr.group_by(["match_key", "round"]).agg([
        pl.col("sec").max().alias("last_sec"),
        pl.col("sec").min().alias("first_sec"),
        pl.len().alias("n_rows"),
        pl.col("tick").min().alias("tick_min"),
        pl.col("tick").max().alias("tick_max"),
        pl.col("y").max().alias("ct_win"),
        pl.col("y").min().alias("ct_win_min"),
        pl.col("bomb_planted").max().alias("planted_any"),
    ]))
    rnd = rnd.join(
        re10.select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
        .rename({"tick": "re_tick"}),
        on=["match_key", "round"], how="left")
    CHECKS["grid_rounds_matched_round_end_p10"] = rnd.filter(pl.col("win_reason_code").is_not_null()).height
    CHECKS["grid_rounds_label_mismatch_vs_round_end_winner"] = rnd.filter(
        pl.col("winner_team_code").is_not_null()
        & ((pl.col("winner_team_code") == 3).cast(pl.Int64) != pl.col("ct_win"))).height
    CHECKS["grid_rounds_label_inconsistent_within_round"] = rnd.filter(pl.col("ct_win") != pl.col("ct_win_min")).height
    rnd = rnd.with_columns(
        (pl.col("win_reason_code") == EXPIRY_CODE).cast(pl.Float64).alias("r12"),
        pl.col("ct_win").cast(pl.Float64).alias("ctw"),
        pl.col("planted_any").cast(pl.Float64).alias("planted_f"),
    )

    r12 = rnd.filter(pl.col("win_reason_code") == EXPIRY_CODE)
    CHECKS["r12_grid_rounds"] = r12.height
    CHECKS["r12_last_sec_unique_values"] = sorted(set(r12["last_sec"].to_list()))
    CHECKS["r12_rows_per_round_unique"] = sorted(set(r12["n_rows"].to_list()))
    CHECKS["r12_tick_max_minus_tick_min_seconds_unique"] = sorted(
        set(((r12["tick_max"] - r12["tick_min"]) / 64.0).to_list()))
    CHECKS["r12_round_end_minus_tick_min_seconds_unique"] = sorted(
        set(((r12["re_tick"] - r12["tick_min"]) / 64.0).to_list()))
    CHECKS["r12_round_end_minus_tick_max_seconds_unique"] = sorted(
        set(((r12["re_tick"] - r12["tick_max"]) / 64.0).to_list()))
    CHECKS["grid_expiry_round_count_not_in_grid_from_round_end"] = int(
        re10.filter(pl.col("win_reason_code") == EXPIRY_CODE).height - r12.height)

    # 1. Share of rounds by time expiry, and last grid second distribution
    c_grid_r12 = cell(rnd, grid_pop, W_g, ["r12"])
    SECTIONS["expiry_share_grid_rounds"] = c_grid_r12
    add_num("grid rounds (1 Hz grid, 1,521 matches): share ending by time expiry (reason 12)",
            c_grid_r12["r12"], c_grid_r12["r12_lo"], c_grid_r12["r12_hi"], c_grid_r12["n_matches"], c_grid_r12["n_rows"])
    add_num("grid rounds: count ending by time expiry (reason 12)",
            int(rnd["r12"].sum()), None, None, c_grid_r12["n_matches"], c_grid_r12["n_rows"])

    last = rnd["last_sec"].to_numpy()
    qs = np.percentile(last, [5, 25, 50, 75, 95])
    add_num("grid rounds: last grid second (seconds_since_round_live max), median", float(qs[2]), None, None,
            None, int(rnd.height))
    add_num("grid rounds: last grid second, 5th percentile", float(qs[0]), None, None, None, int(rnd.height))
    add_num("grid rounds: last grid second, 95th percentile", float(qs[4]), None, None, None, int(rnd.height))
    n_last114 = int((last == 114.0).sum())
    add_num("grid rounds: count with last grid second exactly 114", n_last114, None, None, None, int(rnd.height))
    add_num("grid rounds: count with last grid second exactly 114 that are time expiry",
            int(rnd.filter((pl.col("last_sec") == 114.0) & (pl.col("r12") == 1.0)).height),
            None, None, None, int(rnd.height))

    last_dist = (rnd.group_by("last_sec")
                 .agg([pl.len().alias("n_all"),
                       pl.col("r12").sum().alias("n_expiry"),
                       pl.col("planted_f").sum().alias("n_planted")])
                 .sort("last_sec"))
    SECTIONS["last_second_distribution"] = [
        {"last_sec": float(r["last_sec"]), "n_all": int(r["n_all"]),
         "n_expiry": int(r["n_expiry"]), "n_planted": int(r["n_planted"])}
        for r in last_dist.iter_rows(named=True)
    ]

    # 2. Unplanted states, seconds_since_round_live >= 90, grid (all matches)
    un = gr.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") >= 90))
    CHECKS["unplanted_states_sec_ge_90"] = un.height
    sec2 = {}
    for lo in BIN_LOS:
        bin_all_n = 0
        cls_n_sum = 0
        for cls in CLASSES:
            sub = cls_filter(bin_filter(un, lo), cls)
            c = cell(sub, grid_pop, W_g, ["y"])
            TABLE.append({
                "section": "2_grid_all_matches", "bin_s": bin_label(lo), "ct_diff_class": cls,
                "n_rows": c["n_rows"], "n_matches": c["n_matches"], "n_rounds": c["n_rounds"],
                "obs_ct_rate": c["y"], "obs_ci_low": c["y_lo"], "obs_ci_high": c["y_hi"],
                "pred_mean_p_xgb_grid": None, "pred_ci_low": None, "pred_ci_high": None,
                "diff_obs_minus_pred": None, "diff_ci_low": None, "diff_ci_high": None,
            })
            if cls == "all":
                bin_all_n = c["n_rows"]
                sec2[bin_label(lo)] = c
            else:
                cls_n_sum += c["n_rows"]
        CHECKS[f"sec2_class_rows_sum_equals_all_{bin_label(lo)}"] = (cls_n_sum == bin_all_n)
    SECTIONS["unplanted_sec_ge_90_grid_bins"] = sec2
    c_un_all = cell(un, grid_pop, W_g, ["y"])
    add_num("grid unplanted states at 90+ s (all matches): CT win rate",
            c_un_all["y"], c_un_all["y_lo"], c_un_all["y_hi"], c_un_all["n_matches"], c_un_all["n_rows"])

    # 3. Test split: model comparison
    pred = pl.read_parquet(PRED_PATH)
    CHECKS["pred_rows"] = pred.height
    CHECKS["pred_matches"] = pred["match_key"].n_unique()
    CHECKS["pred_unique_match_round_tick"] = pred.select(["match_key", "round", "tick"]).unique().height
    CHECKS["pred_p_xgb_grid_null_count"] = pred["p_xgb_grid"].null_count()
    CHECKS["pred_p_xgb_grid_min"] = float(pred["p_xgb_grid"].min())
    CHECKS["pred_p_xgb_grid_max"] = float(pred["p_xgb_grid"].max())
    pred = (pred.select(["match_key", "round", "tick", "seconds_since_round_live", "label_ct_win",
                         "p_xgb_grid", "p_lr_all"])
            .rename({"seconds_since_round_live": "pred_sec", "label_ct_win": "pred_y"}))
    pj = pred.join(gr, on=["match_key", "round", "tick"], how="left")
    CHECKS["pred_join_unmatched_rows"] = pj.filter(pl.col("bomb_planted").is_null()).height
    CHECKS["pred_sec_mismatch_rows"] = pj.filter((pl.col("pred_sec") - pl.col("sec")).abs() > 1e-9).height
    CHECKS["pred_label_mismatch_rows"] = pj.filter(pl.col("pred_y") != pl.col("y")).height
    test_pop = sorted(pred["match_key"].unique().to_list())
    W_t = make_weights(len(test_pop))
    CHECKS["test_matches"] = len(test_pop)
    CHECKS["test_rounds"] = pred.select(["match_key", "round"]).unique().height
    CHECKS["grid_rows_of_test_matches"] = gr.filter(pl.col("match_key").is_in(test_pop)).height

    tu = (pj.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") >= 90))
          .select(["match_key", "round", "tick", "sec", "cls", "y",
                   pl.col("p_xgb_grid").cast(pl.Float64).alias("p"),
                   pl.col("p_lr_all").cast(pl.Float64).alias("p_lr")]))
    tu = tu.with_columns((pl.col("y").cast(pl.Float64) - pl.col("p")).alias("d"))
    y_np = tu["y"].cast(pl.Float64).to_numpy()
    p_np = tu["p"].to_numpy()
    plr_np = tu["p_lr"].to_numpy()
    tu = tu.with_columns([
        pl.Series("ll_m", ll_np(y_np, p_np)),
        pl.Series("ll_lr", ll_np(y_np, plr_np)),
    ])
    CHECKS["test_unplanted_states_sec_ge_90"] = tu.height

    for lo in BIN_LOS:
        for cls in CLASSES:
            sub = cls_filter(bin_filter(tu, lo), cls)
            c = cell(sub, test_pop, W_t, ["y", "p", "d"])
            TABLE.append({
                "section": "3_test_model", "bin_s": bin_label(lo), "ct_diff_class": cls,
                "n_rows": c["n_rows"], "n_matches": c["n_matches"], "n_rounds": c["n_rounds"],
                "obs_ct_rate": c["y"], "obs_ci_low": c["y_lo"], "obs_ci_high": c["y_hi"],
                "pred_mean_p_xgb_grid": c["p"], "pred_ci_low": c["p_lo"], "pred_ci_high": c["p_hi"],
                "diff_obs_minus_pred": c["d"], "diff_ci_low": c["d_lo"], "diff_ci_high": c["d_hi"],
            })
            if cls == "all":
                SECTIONS.setdefault("test_model_bins", {})[bin_label(lo)] = c

    # 3b. Final 20 s before expiry: per-second trajectory
    per_sec = []
    for s in range(WIN_LO, WIN_HI):
        g_s = gr.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") == float(s)))
        cg = cell(g_s, grid_pop, W_g, ["y"])
        t_s = tu.filter(pl.col("sec") == float(s))
        ct = cell(t_s, test_pop, W_t, ["y", "p", "d"])
        per_sec.append({
            "second": s,
            "grid_n_rows": cg["n_rows"], "grid_obs_ct_rate": cg["y"], "grid_obs_ci_low": cg["y_lo"],
            "grid_obs_ci_high": cg["y_hi"],
            "test_n_rows": ct["n_rows"], "test_obs_ct_rate": ct["y"], "test_obs_ci_low": ct["y_lo"],
            "test_obs_ci_high": ct["y_hi"], "test_pred_mean": ct["p"], "test_pred_ci_low": ct["p_lo"],
            "test_pred_ci_high": ct["p_hi"], "test_diff_obs_minus_pred": ct["d"],
            "test_diff_ci_low": ct["d_lo"], "test_diff_ci_high": ct["d_hi"],
        })
    SECTIONS["final20_per_second"] = per_sec

    # 3c. Final 20 s window (test): log loss and calibration
    win = tu.filter((pl.col("sec") >= WIN_LO) & (pl.col("sec") < WIN_HI))
    nt_win = gr.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") >= WIN_LO) & (pl.col("sec") < WIN_HI)
                       & (~pl.col("match_key").is_in(test_pop)))
    c_const = float(nt_win["y"].cast(pl.Float64).mean())
    CHECKS["final20_constant_baseline_rate_nontest"] = c_const
    CHECKS["final20_constant_baseline_rows_nontest"] = nt_win.height
    CHECKS["final20_constant_baseline_matches_nontest"] = nt_win["match_key"].n_unique()
    y_w = win["y"].cast(pl.Float64).to_numpy()
    ll_c_np = ll_np(y_w, np.full(y_w.shape[0], c_const))
    win = win.with_columns([
        pl.Series("ll_c", ll_c_np),
    ]).with_columns((pl.col("ll_m") - pl.col("ll_c")).alias("d_mc"))
    c_win = cell(win, test_pop, W_t, ["y", "p", "d", "ll_m", "ll_lr", "ll_c", "d_mc"])
    SECTIONS["final20_test_window"] = c_win
    add_num("final 20 s before expiry, unplanted, test split: CT win rate (observed)",
            c_win["y"], c_win["y_lo"], c_win["y_hi"], c_win["n_matches"], c_win["n_rows"])
    add_num("final 20 s, unplanted, test split: mean p_xgb_grid (predicted P(CT))",
            c_win["p"], c_win["p_lo"], c_win["p_hi"], c_win["n_matches"], c_win["n_rows"])
    add_num("final 20 s, unplanted, test split: observed minus predicted (calibration gap)",
            c_win["d"], c_win["d_lo"], c_win["d_hi"], c_win["n_matches"], c_win["n_rows"])
    add_num("final 20 s, unplanted, test split: log loss p_xgb_grid",
            c_win["ll_m"], c_win["ll_m_lo"], c_win["ll_m_hi"], c_win["n_matches"], c_win["n_rows"])
    add_num("final 20 s, unplanted, test split: log loss constant baseline (non-test CT rate)",
            c_win["ll_c"], c_win["ll_c_lo"], c_win["ll_c_hi"], c_win["n_matches"], c_win["n_rows"])
    add_num("final 20 s, unplanted, test split: log loss p_lr_all (reference)",
            c_win["ll_lr"], c_win["ll_lr_lo"], c_win["ll_lr_hi"], c_win["n_matches"], c_win["n_rows"])
    add_num("final 20 s, unplanted, test split: log loss difference model minus constant (negative = model better)",
            c_win["d_mc"], c_win["d_mc_lo"], c_win["d_mc_hi"], c_win["n_matches"], c_win["n_rows"])

    pbins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0000001]
    pbin_rows = []
    for lo, hi in zip(pbins[:-1], pbins[1:]):
        sub = win.filter((pl.col("p") >= lo) & (pl.col("p") < hi))
        c = cell(sub, test_pop, W_t, ["y", "p"])
        pbin_rows.append({
            "p_lo": lo, "p_hi": min(hi, 1.0), "n_rows": c["n_rows"], "n_matches": c["n_matches"],
            "obs_ct_rate": c["y"], "obs_ci_low": c["y_lo"], "obs_ci_high": c["y_hi"],
            "mean_p": c["p"], "mean_p_ci_low": c["p_lo"], "mean_p_ci_high": c["p_hi"],
        })
    SECTIONS["final20_test_p_bins"] = pbin_rows
    for r in pbin_rows:
        TABLE.append({
            "section": "3_final20_p_bins", "bin_s": f"p[{r['p_lo']},{r['p_hi']})", "ct_diff_class": None,
            "n_rows": r["n_rows"], "n_matches": r["n_matches"], "n_rounds": None,
            "obs_ct_rate": r["obs_ct_rate"], "obs_ci_low": r["obs_ci_low"], "obs_ci_high": r["obs_ci_high"],
            "pred_mean_p_xgb_grid": r["mean_p"], "pred_ci_low": r["mean_p_ci_low"], "pred_ci_high": r["mean_p_ci_high"],
            "diff_obs_minus_pred": None, "diff_ci_low": None, "diff_ci_high": None,
        })

    # ---------------- 4. Unplanted rounds reaching 110 s ----------------
    ur = (gr.filter((pl.col("bomb_planted") == 0) & (pl.col("sec") >= 110))
          .select(["match_key", "round"]).unique()
          .join(rnd.select(["match_key", "round", "ctw", "win_reason_code", "last_sec"]),
                on=["match_key", "round"], how="left"))
    c4 = cell(ur, grid_pop, W_g, ["ctw"])
    SECTIONS["unplanted_reach_110"] = c4
    add_num("grid: unplanted rounds that reach second 110 or later (count of rounds)",
            c4["n_rows"], None, None, c4["n_matches"], c4["n_rows"])
    add_num("grid: CT win rate among unplanted rounds reaching second 110 or later",
            c4["ctw"], c4["ctw_lo"], c4["ctw_hi"], c4["n_matches"], c4["n_rows"])
    reason_break = (ur.group_by("win_reason_code").agg(pl.len().alias("n")).sort("win_reason_code"))
    SECTIONS["unplanted_reach_110_by_reason"] = [
        {"win_reason_code": (None if r["win_reason_code"] is None else int(r["win_reason_code"])),
         "n_rounds": int(r["n"])}
        for r in reason_break.iter_rows(named=True)
    ]
    add_num("grid: unplanted rounds reaching 110 s that end by time expiry (reason 12)",
            int(ur.filter(pl.col("win_reason_code") == EXPIRY_CODE).height), None, None,
            c4["n_matches"], c4["n_rows"])

    # ---------------- outputs ----------------
    out = {
        "id": "N3_timeexpiry",
        "script": SCRIPT_PATH,
        "output_json": OUT_PATH,
        "numbers": NUMBERS,
        "table": TABLE,
        "sections": SECTIONS,
        "checks": CHECKS,
        "assumptions": ASSUMPTIONS,
        "caveats": CAVEATS,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(clean(out), f, indent=2)

    # ---------------- console summary ----------------
    print("CHECKS:")
    for k, v in CHECKS.items():
        print("  ", k, "=", v)
    print("REASON TABLE (round_end 2026 p10):")
    for r in reason_table:
        print("  ", r)
    print("NUMBERS:")
    for n in NUMBERS:
        print("  ", n)
    print("TABLE rows:")
    for r in TABLE:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("FINAL20 PER SECOND:")
    for r in per_sec:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("FINAL20 P BINS:")
    for r in pbin_rows:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("UNPLANTED REACH 110 by reason:", SECTIONS["unplanted_reach_110_by_reason"])
    print("LAST-SECOND DISTRIBUTION (sec, all, expiry, planted):")
    print("  ", [(r["last_sec"], r["n_all"], r["n_expiry"], r["n_planted"]) for r in SECTIONS["last_second_distribution"]])
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
