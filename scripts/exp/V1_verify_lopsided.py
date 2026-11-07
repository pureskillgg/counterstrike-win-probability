"""V1 independent verification of the lopsided-state log loss / death count / full-buy claims.

Written from raw parquet files only. Does not read the analyst's N1 script or output.
Python: run as  <python> -I -X utf8 <this file>
Outputs: G:/pureskill/csds-cs2/data/viz/V1_verify_results.json
"""
import glob
import json

import numpy as np
import polars as pl

B = "G:/pureskill/csds-cs2/data"
TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
OUT = f"{B}/viz/V1_verify_results.json"

NB = 500          # match-cluster bootstrap resamples
SEED = 0          # ASSUMPTION: numpy default_rng(0) for all bootstraps
EPS = 1e-15       # ASSUMPTION: probability clip before log loss (natural log)
FULLBUY = 20000   # ASSUMPTION from task: team equipment sum at seconds_since_live==0
K = 36            # alive_ct*6 + alive_t, alive counts 0..5

RES = {"assumptions": [
    "ASSUMPTION: test matches = match_keys in per_tick_test_preds.parquet (304).",
    "ASSUMPTION: log loss uses natural log, p clipped to [1e-15, 1-1e-15]; rows are 1 Hz grid rows.",
    "ASSUMPTION: bootstrap = percentile 95% over 500 match-cluster resamples, numpy default_rng(0).",
    "ASSUMPTION: death window = [first tick with seconds_since_round_live==0, round_end tick] inclusive.",
    "ASSUMPTION: full buy = both teams have 5 players at seconds_since_live==0 and equipment sum >= 20000.",
]}


def boot_weights(M, nb=NB, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M, size=(nb, M))
    W = np.zeros((nb, M))
    np.add.at(W, (np.arange(nb)[:, None], idx), 1.0)
    return W


def ci(a):
    a = np.asarray(a, float)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def pearson_from_sums(N, Sx, Sy, Sxx, Syy, Sxy):
    num = N * Sxy - Sx * Sy
    den = np.sqrt((N * Sxx - Sx ** 2) * (N * Syy - Sy ** 2))
    return num / den


def binary_entropy(q):
    q = np.clip(q, EPS, 1 - EPS)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


# ---------------------------------------------------------------- PART 1
pt = pl.read_parquet(f"{B}/viz/per_tick_states.parquet",
                     columns=["match_key", "round", "tick", "alive_ct", "alive_t",
                              "label_ct_win", "seconds_since_round_live"])
pr = pl.read_parquet(f"{B}/viz/per_tick_test_preds.parquet",
                     columns=["match_key", "round", "tick", "p_xgb_grid", "label_ct_win"])

test_keys = sorted(pr["match_key"].unique().to_list())
M = len(test_keys)
kidx = {k: i for i, k in enumerate(test_keys)}
RES["n_test_matches"] = M

d = pr.join(pt.select(["match_key", "round", "tick", "alive_ct", "alive_t"]),
            on=["match_key", "round", "tick"], how="left")
assert d["alive_ct"].null_count() == 0, "preds rows must all join to grid rows"
ct = d["alive_ct"].to_numpy().astype(np.int64)
tt = d["alive_t"].to_numpy().astype(np.int64)
y = d["label_ct_win"].to_numpy().astype(np.float64)
p_raw = d["p_xgb_grid"].to_numpy().astype(np.float64)
p = np.clip(p_raw, EPS, 1 - EPS)
ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
ad = np.abs(ct - tt).astype(np.float64)
midx = np.array([kidx[k] for k in d["match_key"].to_list()], dtype=np.int64)
state = ct * 6 + tt
RES["p_xgb_grid_min"] = float(p_raw.min())
RES["p_xgb_grid_max"] = float(p_raw.max())
RES["n_test_rows"] = int(len(y))

n_ms = np.zeros((M, K)); sY = np.zeros((M, K)); sP = np.zeros((M, K)); sLL = np.zeros((M, K))
np.add.at(n_ms, (midx, state), 1.0)
np.add.at(sY, (midx, state), y)
np.add.at(sP, (midx, state), p)
np.add.at(sLL, (midx, state), ll)
n_s = n_ms.sum(0)
m_s = (n_ms > 0).sum(0)
qual = (n_s >= 200) & (m_s >= 20)
RES["n_states_with_rows"] = int((n_s > 0).sum())
RES["n_qualifying_states_test"] = int(qual.sum())

W = boot_weights(M)
nb_ = W @ n_ms
yb_ = W @ sY
with np.errstate(invalid="ignore", divide="ignore"):
    rate_b = np.where(nb_ > 0, yb_ / np.where(nb_ > 0, nb_, 1), np.nan)
    pb_ = W @ sP
    meanp_b = np.where(nb_ > 0, pb_ / np.where(nb_ > 0, nb_, 1), np.nan)
    lb_ = W @ sLL
    meanll_b = np.where(nb_ > 0, lb_ / np.where(nb_ > 0, nb_, 1), np.nan)

rows_state = []
for s in range(K):
    if n_s[s] == 0:
        continue
    ctv, tv = s // 6, s % 6
    rate = sY[:, s].sum() / n_s[s]
    rows_state.append({
        "alive_ct": ctv, "alive_t": tv, "rows": int(n_s[s]), "matches": int(m_s[s]),
        "qualifies": bool(qual[s]),
        "ct_win_rate": float(rate),
        "ct_win_rate_ci95_cluster": ci(rate_b[:, s]),
        "mean_pred_pct_ct": float(sP[:, s].sum() / n_s[s]),
        "mean_row_logloss": float(sLL[:, s].sum() / n_s[s]),
        "mean_row_logloss_ci95_cluster": ci(meanll_b[:, s]),
        "absdiff": abs(ctv - tv),
    })
RES["state_table_test"] = rows_state


def find_state(a, b):
    for r in rows_state:
        if r["alive_ct"] == a and r["alive_t"] == b:
            return r
    return None


# Row-level correlation |diff| vs per-row log loss, all test rows
mN = n_ms.sum(1)
mx = np.bincount(midx, weights=ad, minlength=M)
my = np.bincount(midx, weights=ll, minlength=M)
mxx = np.bincount(midx, weights=ad * ad, minlength=M)
myy = np.bincount(midx, weights=ll * ll, minlength=M)
mxy = np.bincount(midx, weights=ad * ll, minlength=M)
r_point_formula = float(pearson_from_sums(mN.sum(), mx.sum(), my.sum(), mxx.sum(), myy.sum(), mxy.sum()))
r_point_numpy = float(np.corrcoef(ad, ll)[0, 1])
rb = pearson_from_sums(W @ mN, W @ mx, W @ my, W @ mxx, W @ myy, W @ mxy)
RES["corr_absdiff_vs_rowlogloss_pearson"] = r_point_numpy
RES["corr_absdiff_vs_rowlogloss_formula_check"] = r_point_formula
RES["corr_absdiff_vs_rowlogloss_ci95_cluster"] = ci(rb)

# Descriptive state-level correlation over qualifying states (unweighted)
qs = [r for r in rows_state if r["qualifies"]]
xs = np.array([r["absdiff"] for r in qs], float)
ys = np.array([r["mean_row_logloss"] for r in qs], float)
RES["corr_qualifying_states_absdiff_vs_meanll_descriptive"] = float(np.corrcoef(xs, ys)[0, 1])
RES["n_qualifying_states_for_corr"] = int(len(qs))

# Row-level means by |diff|
by_ad = {}
for a in range(6):
    msk = ad == a
    by_ad[a] = {"rows": int(msk.sum()), "mean_row_logloss": float(ll[msk].mean()) if msk.any() else None}
RES["row_logloss_by_absdiff"] = by_ad

# Requested state log losses (both orientations reported)
RES["claim_check_state_logloss"] = {
    "5v5 (ct5,t5)": find_state(5, 5)["mean_row_logloss"],
    "4v4 (ct4,t4)": find_state(4, 4)["mean_row_logloss"],
    "5v2 (ct5,t2)": find_state(5, 2)["mean_row_logloss"] if find_state(5, 2) else None,
    "5v1 (ct5,t1)": find_state(5, 1)["mean_row_logloss"] if find_state(5, 1) else None,
    "2v5 (ct2,t5)": find_state(2, 5)["mean_row_logloss"] if find_state(2, 5) else None,
    "1v5 (ct1,t5)": find_state(1, 5)["mean_row_logloss"] if find_state(1, 5) else None,
}

# Diagnostic: state-only baselines (NOT in the claim)
# (a) base rate estimated from NON-test grid rows, applied to test rows (out-of-sample)
test_key_df = pl.DataFrame({"match_key": test_keys})
nontest = pt.join(test_key_df, on="match_key", how="anti")
nt_state = nontest["alive_ct"].to_numpy().astype(np.int64) * 6 + nontest["alive_t"].to_numpy().astype(np.int64)
nt_y = nontest["label_ct_win"].to_numpy().astype(np.float64)
nt_n = np.bincount(nt_state, minlength=K).astype(float)
nt_s = np.bincount(nt_state, weights=nt_y, minlength=K)
base_q = np.where(nt_n > 0, nt_s / np.maximum(nt_n, 1), 0.5)
qb = np.clip(base_q[state], EPS, 1 - EPS)
base_ll = -(y * np.log(qb) + (1 - y) * np.log(1 - qb))
RES["n_nontest_grid_rows"] = int(len(nt_y))
RES["n_nontest_matches"] = int(nontest["match_key"].n_unique())
RES["diag_state_baseline"] = []
for r in qs:
    s = r["alive_ct"] * 6 + r["alive_t"]
    msk = state == s
    q_in = r["ct_win_rate"]
    RES["diag_state_baseline"].append({
        "alive_ct": r["alive_ct"], "alive_t": r["alive_t"], "rows": r["rows"],
        "model_row_logloss": r["mean_row_logloss"],
        "nontest_base_rate_logloss": float(base_ll[msk].mean()),
        "in_sample_state_entropy_nats": float(binary_entropy(np.array(q_in))),
        "nontest_base_rate": float(base_q[s]),
    })
bl = np.array([x["nontest_base_rate_logloss"] for x in RES["diag_state_baseline"]])
ent = np.array([x["in_sample_state_entropy_nats"] for x in RES["diag_state_baseline"]])
RES["diag_corr_absdiff_vs_baseline_logloss_qualifying"] = float(np.corrcoef(xs, bl)[0, 1])
RES["diag_corr_absdiff_vs_state_entropy_qualifying"] = float(np.corrcoef(xs, ent)[0, 1])
RES["diag_row_level_baseline_mean_logloss"] = float(base_ll.mean())
RES["diag_row_level_model_mean_logloss"] = float(ll.mean())


# ---------------------------------------------------------------- PART 2
def load_pages(ch, cols):
    files = sorted(glob.glob(f"{TOME}/basic_{ch}.COMBINED/dataframe_*"))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


re26 = load_pages("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]) \
    .filter(pl.col("match_key").str.starts_with("csds/2026/"))
live = pt.group_by(["match_key", "round"]).agg(
    pl.col("tick").filter(pl.col("seconds_since_round_live") == 0).min().alias("live_tick"),
    pl.col("tick").max().alias("last_grid_tick"),
    pl.len().alias("n_grid_rows"),
)
rt = live.join(re26.rename({"tick": "end_tick"}), on=["match_key", "round"], how="inner").sort(["match_key", "round"])
rt = rt.with_row_index("rix")
RES["n_grid_rounds"] = int(live.height)
RES["n_grid_rounds_with_roundend"] = int(rt.height)
RES["grid_rounds_player_count_10"] = int(rt.filter(pl.col("player_count") == 10).height)

de26 = load_pages("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]) \
    .filter(pl.col("match_key").str.starts_with("csds/2026/"))
pi26 = load_pages("player_info", ["match_key", "round", "player_id", "team_code"]) \
    .filter(pl.col("match_key").str.starts_with("csds/2026/"))
de = de26.join(rt.select(["match_key", "round", "rix", "live_tick", "end_tick"]),
               on=["match_key", "round"], how="inner")
RES["deaths_in_grid_rounds_all"] = int(de.height)
RES["deaths_before_live"] = int(de.filter(pl.col("tick") < pl.col("live_tick")).height)
RES["deaths_after_roundend_tick"] = int(de.filter(pl.col("tick") > pl.col("end_tick")).height)
de = de.filter((pl.col("tick") >= pl.col("live_tick")) & (pl.col("tick") <= pl.col("end_tick")))
RES["deaths_in_window"] = int(de.height)
de = de.join(pi26.select(["match_key", "round", "player_id", "team_code"]),
             on=["match_key", "round", "player_id"], how="left")
both = de.filter(pl.col("player_team_code").is_not_null() & pl.col("team_code").is_not_null())
RES["team_code_mismatch_death_vs_roster"] = int(both.filter(pl.col("player_team_code") != pl.col("team_code")).height)
de = de.with_columns(pl.coalesce(["player_team_code", "team_code"]).alias("team"))
RES["deaths_in_window_null_team"] = int(de.filter(pl.col("team").is_null()).height)
dup = de.group_by(["match_key", "round", "player_id"]).len().filter(pl.col("len") > 1)
RES["duplicate_player_round_keys_in_window"] = int(dup.height)
de = de.unique(["match_key", "round", "player_id"], keep="first")
RES["deaths_in_window_after_dedup"] = int(de.height)

cnt = de.group_by(["rix", "team"]).agg(pl.len().alias("n"))
dct = cnt.filter(pl.col("team") == 3).select(["rix", pl.col("n").alias("d_ct")])
dtt = cnt.filter(pl.col("team") == 2).select(["rix", pl.col("n").alias("d_t")])
rr = rt.join(dct, on="rix", how="left").join(dtt, on="rix", how="left").with_columns(
    pl.col("d_ct").fill_null(0), pl.col("d_t").fill_null(0))
dct_np = rr["d_ct"].to_numpy().astype(np.int64)
dt_np = rr["d_t"].to_numpy().astype(np.int64)
win_ct = (rr["winner_team_code"].to_numpy() == 3)
R = len(rr)
rmatch = rr["match_key"].to_list()
RES["max_deaths_one_team_one_round"] = int(max(dct_np.max(), dt_np.max()))
RES["rounds_with_team_deaths_gt5"] = int(((dct_np > 5) | (dt_np > 5)).sum())

win_d = np.where(win_ct, dct_np, dt_np)
lose_d = np.where(win_ct, dt_np, dct_np)
RES["n_rounds_part2"] = int(R)
RES["n_ct_win_rounds"] = int(win_ct.sum())
RES["n_t_win_rounds"] = int((~win_ct).sum())
RES["hist_winner_deaths_0to5"] = np.bincount(np.minimum(win_d, 5), minlength=6).tolist()
RES["hist_loser_deaths_0to5"] = np.bincount(np.minimum(lose_d, 5), minlength=6).tolist()
RES["mean_winner_deaths"] = float(win_d.mean())
RES["mean_loser_deaths"] = float(lose_d.mean())
RES["mean_ct_deaths_ct_win_rounds"] = float(dct_np[win_ct].mean())
RES["mean_t_deaths_ct_win_rounds"] = float(dt_np[win_ct].mean())
RES["mean_ct_deaths_t_win_rounds"] = float(dct_np[~win_ct].mean())
RES["mean_t_deaths_t_win_rounds"] = float(dt_np[~win_ct].mean())
RES["mean_winner_deaths_ct_win_rounds"] = float(win_d[win_ct].mean())
RES["mean_loser_deaths_ct_win_rounds"] = float(lose_d[win_ct].mean())
RES["mean_winner_deaths_t_win_rounds"] = float(win_d[~win_ct].mean())
RES["mean_loser_deaths_t_win_rounds"] = float(lose_d[~win_ct].mean())
# round-level means split CT/T win, plus match-cluster CI for winner/loser means
mi_r = np.array([kidx.get(k, -1) for k in rmatch])  # -1 means not a test match
match_list = sorted(set(rmatch))
mi_all = {k: i for i, k in enumerate(match_list)}
mi_all_arr = np.array([mi_all[k] for k in rmatch], dtype=np.int64)
MM = len(match_list)
rN = np.bincount(mi_all_arr, minlength=MM).astype(float)
rW = np.bincount(mi_all_arr, weights=win_d.astype(float), minlength=MM)
rL = np.bincount(mi_all_arr, weights=lose_d.astype(float), minlength=MM)
Wr = boot_weights(MM)
bw = (Wr @ rW) / (Wr @ rN)
bl2 = (Wr @ rL) / (Wr @ rN)
RES["ci95_cluster_mean_winner_deaths"] = ci(bw)
RES["ci95_cluster_mean_loser_deaths"] = ci(bl2)
RES["n_match_clusters_part2"] = MM

# Consistency: grid alive counts vs cumulative deaths in the same round
BIG = 10 ** 9
dw_ct = de.filter(pl.col("team") == 3)
dw_t = de.filter(pl.col("team") == 2)


def cum_key(dfx):
    rix = dfx["rix"].to_numpy().astype(np.int64)
    tk = dfx["tick"].to_numpy().astype(np.int64)
    return np.sort(rix * BIG + tk)


kct = cum_key(dw_ct)
kt = cum_key(dw_t)
gp = pt.join(rt.select(["match_key", "round", "rix"]), on=["match_key", "round"], how="inner")
grix = gp["rix"].to_numpy().astype(np.int64)
gtick = gp["tick"].to_numpy().astype(np.int64)
base_key = grix * BIG
dead_ct = np.searchsorted(kct, grix * BIG + gtick, side="right") - np.searchsorted(kct, base_key, side="left")
dead_t = np.searchsorted(kt, grix * BIG + gtick, side="right") - np.searchsorted(kt, base_key, side="left")
exp_ct = 5 - dead_ct
exp_t = 5 - dead_t
ga_ct = gp["alive_ct"].to_numpy().astype(np.int64)
ga_t = gp["alive_t"].to_numpy().astype(np.int64)
RES["alive_consistency_grid_rows"] = int(len(ga_ct))
RES["alive_consistency_ct_exact_match_rate"] = float((exp_ct == ga_ct).mean())
RES["alive_consistency_t_exact_match_rate"] = float((exp_t == ga_t).mean())
RES["alive_consistency_ct_mismatch_rows"] = int((exp_ct != ga_ct).sum())
RES["alive_consistency_t_mismatch_rows"] = int((exp_t != ga_t).sum())

# Part 2 round-level: distribution by final alive counts (grid last row) for comparison
last = pt.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("alive_ct").last(), pl.col("alive_t").last())
RES["grid_last_row_alive_ct_mean"] = float(last["alive_ct"].mean())
RES["grid_last_row_alive_t_mean"] = float(last["alive_t"].mean())

# ---------------------------------------------------------------- PART 3
p1 = pl.read_parquet(f"{B}/state_sample/players_1hz.parquet",
                     columns=["match_key", "round", "seconds_since_live", "team_code",
                              "equipment_value_calc", "player_id"])
s0 = p1.filter(pl.col("seconds_since_live") == 0)
agg = s0.group_by(["match_key", "round", "team_code"]).agg(
    pl.len().alias("n"), pl.col("equipment_value_calc").sum().alias("eq"))
ct3 = agg.filter(pl.col("team_code") == 3).select(["match_key", "round", pl.col("n").alias("n_ct"), pl.col("eq").alias("eq_ct")])
t2 = agg.filter(pl.col("team_code") == 2).select(["match_key", "round", pl.col("n").alias("n_t"), pl.col("eq").alias("eq_t")])
fbr = ct3.join(t2, on=["match_key", "round"], how="full", coalesce=True).with_columns(
    pl.col("n_ct").fill_null(0), pl.col("n_t").fill_null(0),
    pl.col("eq_ct").fill_null(0), pl.col("eq_t").fill_null(0))
RES["state_sample_rounds"] = int(fbr.height)
RES["sample_team_rounds_with_5_players_at_sec0"] = int((fbr["n_ct"] == 5).sum() + (fbr["n_t"] == 5).sum())
RES["sample_team_rounds_with_not5_players_at_sec0"] = int(((fbr["n_ct"] != 5) | (fbr["n_t"] != 5)).sum())
fbr = fbr.with_columns(
    ((pl.col("eq_ct") >= FULLBUY) & (pl.col("eq_t") >= FULLBUY)).alias("eqfull_both"),
    ((pl.col("n_ct") == 5) & (pl.col("n_t") == 5)).alias("five_each"),
)
fbr = fbr.with_columns((pl.col("eqfull_both") & pl.col("five_each")).alias("full_buy"))
RES["rounds_both_teams_eq_sum_ge_20000_ignoring_5man"] = int(fbr.filter(pl.col("eqfull_both")).height)
RES["rounds_full_buy_qualify"] = int(fbr.filter(pl.col("full_buy")).height)
RES["rounds_full_buy_matches"] = int(fbr.filter(pl.col("full_buy"))["match_key"].n_unique())
RES["rounds_not_full_buy"] = int(fbr.filter(~pl.col("full_buy")).height)
rd = pl.read_parquet(f"{B}/state_sample/rounds.parquet", columns=["match_key", "round", "ct_win"])
RES["rounds_parquet_ct_win_vs_grid_label_agree"] = None
gl = pt.group_by(["match_key", "round"]).agg(pl.col("label_ct_win").first())
cmpj = rd.join(gl, on=["match_key", "round"], how="inner")
RES["rounds_parquet_ct_win_vs_grid_label_agree"] = [int((cmpj["ct_win"] == cmpj["label_ct_win"]).sum()), int(cmpj.height)]

fbk = fbr.select(["match_key", "round", "full_buy"])
gfb = pt.join(fbk, on=["match_key", "round"], how="inner")


def chain_table(gdf, label):
    # gdf: grid rows of a set of rounds (one row per 1 Hz second)
    mkeys = sorted(gdf["match_key"].unique().to_list())
    MM2 = len(mkeys)
    if MM2 == 0:
        return {"label": label, "matches": 0}
    mi = {k: i for i, k in enumerate(mkeys)}
    mid = np.array([mi[k] for k in gdf["match_key"].to_list()], dtype=np.int64)
    ctv = gdf["alive_ct"].to_numpy().astype(np.int64)
    tv = gdf["alive_t"].to_numpy().astype(np.int64)
    yv = gdf["label_ct_win"].to_numpy().astype(np.float64)
    stv = ctv * 6 + tv
    nM = np.zeros((MM2, K)); sYM = np.zeros((MM2, K))
    np.add.at(nM, (mid, stv), 1.0)
    np.add.at(sYM, (mid, stv), yv)
    # rounds per state
    rk = gdf.select(["match_key", "round", "alive_ct", "alive_t"]).with_columns(
        (pl.col("alive_ct") * 6 + pl.col("alive_t")).alias("st"))
    rounds_per_state = rk.group_by(["st"]).agg(pl.struct(["match_key", "round"]).n_unique().alias("nr")).to_dicts()
    rps = {int(x["st"]): int(x["nr"]) for x in rounds_per_state}
    mps = (nM > 0).sum(0)
    Wc = boot_weights(MM2)
    nbb = Wc @ nM
    ybb = Wc @ sYM
    out = []
    for (a, b_) in [(5, 5), (5, 3), (5, 2), (3, 5), (2, 5)]:
        s = a * 6 + b_
        n_rows = int(nM[:, s].sum())
        if n_rows == 0:
            out.append({"state_ct_vs_t": f"{a}v{b_}", "rows": 0, "rounds": 0, "matches": 0})
            continue
        ctw = sYM[:, s].sum() / nM[:, s].sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            rb_s = np.where(nbb[:, s] > 0, ybb[:, s] / np.where(nbb[:, s] > 0, nbb[:, s], 1), np.nan)
        # team with 5 alive: CT if a==5, T if b_==5
        five_is_ct = (a == 5)
        win5 = ctw if five_is_ct else (1 - ctw)
        win5_ci = ci(rb_s) if five_is_ct else ci(1 - rb_s)
        out.append({
            "state_ct_vs_t": f"{a}v{b_}", "rows": n_rows, "rounds": rps.get(s, 0), "matches": int(mps[s]),
            "ct_win_rate": float(ctw), "ct_win_rate_ci95_cluster": ci(rb_s),
            "five_alive_team": "CT" if five_is_ct else "T",
            "five_alive_team_win_rate": float(win5),
            "five_alive_team_win_rate_ci95_cluster": win5_ci,
            "small_n_flag": bool(rps.get(s, 0) < 30),
        })
    return {"label": label, "matches": MM2, "rounds": int(gdf.select(["match_key", "round"]).unique().height),
            "states": out}


RES["chain_full_buy"] = chain_table(gfb.filter(pl.col("full_buy")), "full_buy_vs_full_buy (sample rounds)")
nonfb_keys = fbk.filter(~pl.col("full_buy")).select(["match_key", "round"])
RES["chain_not_full_buy_context"] = chain_table(
    pt.join(nonfb_keys, on=["match_key", "round"], how="inner"), "not full buy (sample rounds, context)")
RES["chain_all_grid_rounds_context"] = chain_table(pt, "all grid rounds (context; no equipment filter)")

# ---------------------------------------------------------------- WRITE
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(RES, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

print("WROTE", OUT)
print("== part1 headline")
for k in ["n_test_matches", "n_test_rows", "n_qualifying_states_test",
          "corr_absdiff_vs_rowlogloss_pearson", "corr_absdiff_vs_rowlogloss_formula_check",
          "corr_absdiff_vs_rowlogloss_ci95_cluster", "corr_qualifying_states_absdiff_vs_meanll_descriptive",
          "n_qualifying_states_for_corr", "claim_check_state_logloss", "row_logloss_by_absdiff",
          "diag_corr_absdiff_vs_baseline_logloss_qualifying", "diag_corr_absdiff_vs_state_entropy_qualifying",
          "diag_row_level_model_mean_logloss", "diag_row_level_baseline_mean_logloss"]:
    print(k, "=", RES[k])
print("== qualifying state table (test)")
for r in rows_state:
    if r["qualifies"]:
        print(f"ct{r['alive_ct']} t{r['alive_t']} rows={r['rows']} m={r['matches']} "
              f"winCT={r['ct_win_rate']:.4f} ci={[round(x,4) for x in r['ct_win_rate_ci95_cluster']]} "
              f"predCT={r['mean_pred_pct_ct']:.4f} LL={r['mean_row_logloss']:.4f}")
print("== all test states (with rows>0), short")
for r in rows_state:
    print(f"ct{r['alive_ct']} t{r['alive_t']} rows={r['rows']} m={r['matches']} q={r['qualifies']} LL={r['mean_row_logloss']:.4f} winCT={r['ct_win_rate']:.3f}")
print("== part2")
for k in ["n_grid_rounds", "n_grid_rounds_with_roundend", "grid_rounds_player_count_10", "deaths_in_grid_rounds_all",
          "deaths_before_live", "deaths_after_roundend_tick", "deaths_in_window", "team_code_mismatch_death_vs_roster",
          "deaths_in_window_null_team", "duplicate_player_round_keys_in_window", "deaths_in_window_after_dedup",
          "max_deaths_one_team_one_round", "rounds_with_team_deaths_gt5", "n_ct_win_rounds", "n_t_win_rounds",
          "hist_winner_deaths_0to5", "hist_loser_deaths_0to5", "mean_winner_deaths", "mean_loser_deaths",
          "mean_winner_deaths_ct_win_rounds", "mean_loser_deaths_ct_win_rounds",
          "mean_winner_deaths_t_win_rounds", "mean_loser_deaths_t_win_rounds",
          "ci95_cluster_mean_winner_deaths", "ci95_cluster_mean_loser_deaths",
          "alive_consistency_ct_exact_match_rate", "alive_consistency_t_exact_match_rate",
          "alive_consistency_ct_mismatch_rows", "alive_consistency_t_mismatch_rows",
          "grid_last_row_alive_ct_mean", "grid_last_row_alive_t_mean"]:
    print(k, "=", RES[k])
print("== part3")
for k in ["state_sample_rounds", "sample_team_rounds_with_5_players_at_sec0", "sample_team_rounds_with_not5_players_at_sec0",
          "rounds_both_teams_eq_sum_ge_20000_ignoring_5man", "rounds_full_buy_qualify", "rounds_full_buy_matches",
          "rounds_not_full_buy", "rounds_parquet_ct_win_vs_grid_label_agree"]:
    print(k, "=", RES[k])
for key in ["chain_full_buy", "chain_not_full_buy_context", "chain_all_grid_rounds_context"]:
    ch = RES[key]
    print("--", ch["label"], "matches", ch.get("matches"), "rounds", ch.get("rounds"))
    for s in ch.get("states", []):
        if s.get("rows", 0) > 0:
            print("   ", s)
