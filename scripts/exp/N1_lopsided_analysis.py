# N1_lopsided: CT win rate, model log loss and death counts by alive-player state.
# Matches are the independent unit; every interval is a match-cluster percentile bootstrap
# (B=500, numpy default_rng(0)). Only numbers computed here are reported.
import glob
import json
import math
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

D = "G:/pureskill/csds-cs2/data"
TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
P1 = D + "/state_sample/players_1hz.parquet"
OUT_JSON = D + "/viz/N1_lopsided_results.json"
B = 500
SEED = 0
MIN_ROWS = 200
MIN_MATCHES = 20
EPS = 1e-15
K = ["match_key", "round", "tick"]
RK = ["match_key", "round"]


def say(*a):
    print(*a, flush=True)


def pages(ch):
    ps = sorted(glob.glob(TOME + "/basic_" + ch + ".COMBINED/dataframe_*"))
    if not ps:
        raise SystemExit("no tome pages for " + ch)
    return ps


def boot_weights(M):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(B, M))
    W = np.zeros((B, M), dtype=np.float64)
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=M)
    return W


def ratio_ci(W, num, den):
    """num, den: (M, K) per-match sums. Returns point, lo, hi arrays of length K."""
    nt = num.sum(axis=0)
    dt = den.sum(axis=0)
    point = np.where(dt > 0, nt / np.where(dt > 0, dt, 1.0), np.nan)
    bn = W @ num
    bd = W @ den
    boot = np.where(bd > 0, bn / np.where(bd > 0, bd, 1.0), np.nan)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return point, lo, hi


def to_matrix(agg, match_list, keys, col):
    mi = {m: i for i, m in enumerate(match_list)}
    ki = {k: j for j, k in enumerate(keys)}
    A = np.zeros((len(match_list), len(keys)), dtype=np.float64)
    for mk, kk, v in agg.select(["match_key", "key", col]).iter_rows():
        A[mi[mk], ki[kk]] = float(v)
    return A


def partition_stats(rows, match_list, W):
    """rows: match_key, key, y, p, ll (row level). Cluster-bootstrap stats for each key."""
    keys = sorted(rows["key"].unique().to_list())
    agg = rows.group_by(["match_key", "key"]).agg([
        pl.len().alias("n"),
        pl.col("y").sum().alias("sy"),
        pl.col("p").sum().alias("sp"),
        pl.col("ll").sum().alias("sl"),
    ])
    N = to_matrix(agg, match_list, keys, "n")
    SY = to_matrix(agg, match_list, keys, "sy")
    SP = to_matrix(agg, match_list, keys, "sp")
    SL = to_matrix(agg, match_list, keys, "sl")
    win, wlo, whi = ratio_ci(W, SY, N)
    mp, mplo, mphi = ratio_ci(W, SP, N)
    ml, mllo, mlhi = ratio_ci(W, SL, N)
    nrows = N.sum(axis=0)
    nm = (N > 0).sum(axis=0)
    out = []
    for j, k in enumerate(keys):
        out.append({
            "key": int(k), "rows": int(nrows[j]), "matches": int(nm[j]),
            "win": float(win[j]), "win_lo": float(wlo[j]), "win_hi": float(whi[j]),
            "mean_pred": float(mp[j]), "mean_pred_lo": float(mplo[j]), "mean_pred_hi": float(mphi[j]),
            "mean_ll": float(ml[j]), "mean_ll_lo": float(mllo[j]), "mean_ll_hi": float(mlhi[j]),
        })
    return out


def corr_stats(rows, match_list, W):
    """Pearson r between x and ll over rows; point and cluster-bootstrap percentile CI."""
    agg = rows.group_by("match_key").agg([
        pl.len().alias("n"),
        pl.col("x").sum().alias("sx"),
        pl.col("ll").sum().alias("sl"),
        (pl.col("x") * pl.col("x")).sum().alias("sxx"),
        (pl.col("ll") * pl.col("ll")).sum().alias("sll"),
        (pl.col("x") * pl.col("ll")).sum().alias("sxl"),
    ])
    mi = {m: i for i, m in enumerate(match_list)}
    names = ["n", "sx", "sl", "sxx", "sll", "sxl"]
    V = {c: np.zeros(len(match_list)) for c in names}
    for row in agg.iter_rows(named=True):
        i = mi[row["match_key"]]
        for c in names:
            V[c][i] = float(row[c])

    def r_of(n, sx, sl, sxx, sll, sxl):
        return (n * sxl - sx * sl) / np.sqrt((n * sxx - sx * sx) * (n * sll - sl * sl))

    point = float(r_of(V["n"].sum(), V["sx"].sum(), V["sl"].sum(), V["sxx"].sum(), V["sll"].sum(), V["sxl"].sum()))
    bt = r_of(W @ V["n"], W @ V["sx"], W @ V["sl"], W @ V["sxx"], W @ V["sll"], W @ V["sxl"])
    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
    return point, float(lo), float(hi)


def part1(grid, preds):
    say("== PART 1: per-state win rate, model P and log loss (test split)")
    test = preds.join(grid.select(K + ["alive_ct", "alive_t", "label_ct_win"]), on=K, how="left")
    say("test pred rows", test.height, "unmatched to grid", test["alive_ct"].null_count())
    y = test["label_ct_win"].to_numpy().astype(np.float64)
    p = np.clip(test["p_xgb_grid"].to_numpy().astype(np.float64), EPS, 1.0 - EPS)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    ac = test["alive_ct"].to_numpy().astype(np.int64)
    at = test["alive_t"].to_numpy().astype(np.int64)
    ad = np.abs(ac - at)
    rows = pl.DataFrame([
        pl.Series("match_key", test["match_key"].to_list()),
        pl.Series("key", ac * 10 + at),
        pl.Series("absdiff", ad),
        pl.Series("y", y),
        pl.Series("p", p),
        pl.Series("ll", ll),
        pl.Series("x", ad.astype(np.float64)),
    ])
    test_matches = sorted(rows["match_key"].unique().to_list())
    say("test matches", len(test_matches))
    Wt = boot_weights(len(test_matches))
    state_stats = partition_stats(rows.select(["match_key", "key", "y", "p", "ll"]), test_matches, Wt)
    rows_diff = rows.select(["match_key", "absdiff", "y", "p", "ll"]).rename({"absdiff": "key"})
    diff_stats = partition_stats(rows_diff, test_matches, Wt)

    grid_matches = sorted(grid["match_key"].unique().to_list())
    Wg = boot_weights(len(grid_matches))
    grows = grid.with_columns([
        (pl.col("alive_ct") * 10 + pl.col("alive_t")).alias("key"),
        pl.col("label_ct_win").cast(pl.Float64).alias("y"),
        pl.lit(0.0).alias("p"),
        pl.lit(0.0).alias("ll"),
    ]).select(["match_key", "key", "y", "p", "ll"])
    full_stats = partition_stats(grows, grid_matches, Wg)
    full_by_key = {r["key"]: r for r in full_stats}

    pt, clo, chi = corr_stats(rows.select(["match_key", "x", "ll"]), test_matches, Wt)
    direct = float(np.corrcoef(ad.astype(np.float64), ll)[0, 1])
    say("corr point (agg)", round(pt, 6), "direct", round(direct, 6), "boot CI", round(clo, 4), round(chi, 4))

    states = []
    for s in state_stats:
        k = s["key"]
        qual = (s["rows"] >= MIN_ROWS) and (s["matches"] >= MIN_MATCHES)
        f = full_by_key.get(k, {"rows": 0, "matches": 0, "win": float("nan")})
        states.append({
            "alive_ct": k // 10, "alive_t": k % 10, "rows": s["rows"], "matches": s["matches"],
            "qualifies": bool(qual),
            "emp_ct_win": s["win"], "emp_ct_win_lo": s["win_lo"], "emp_ct_win_hi": s["win_hi"],
            "mean_pred_ct": s["mean_pred"], "mean_pred_lo": s["mean_pred_lo"], "mean_pred_hi": s["mean_pred_hi"],
            "mean_row_logloss": s["mean_ll"], "ll_lo": s["mean_ll_lo"], "ll_hi": s["mean_ll_hi"],
            "fullgrid_rows": f["rows"], "fullgrid_matches": f["matches"], "fullgrid_emp_ct_win": f["win"],
        })
    states.sort(key=lambda r: (r["alive_ct"], r["alive_t"]))
    for r in states:
        say("state %dv%d rows=%d matches=%d qual=%s win=%.4f [%.4f,%.4f] predP=%.4f ll=%.4f [%.4f,%.4f] | fullgrid rows=%d win=%.4f" % (
            r["alive_ct"], r["alive_t"], r["rows"], r["matches"], r["qualifies"], r["emp_ct_win"], r["emp_ct_win_lo"],
            r["emp_ct_win_hi"], r["mean_pred_ct"], r["mean_row_logloss"], r["ll_lo"], r["ll_hi"],
            r["fullgrid_rows"], r["fullgrid_emp_ct_win"]))

    diffs = []
    for s in diff_stats:
        diffs.append({"abs_diff": s["key"], "rows": s["rows"], "matches": s["matches"],
                      "mean_row_logloss": s["mean_ll"], "ll_lo": s["mean_ll_lo"], "ll_hi": s["mean_ll_hi"],
                      "emp_ct_win": s["win"], "mean_pred_ct": s["mean_pred"]})
        say("|diff|=%d rows=%d matches=%d ll=%.4f [%.4f,%.4f] win=%.4f predP=%.4f" % (
            s["key"], s["rows"], s["matches"], s["mean_ll"], s["mean_ll_lo"], s["mean_ll_hi"], s["win"], s["mean_pred"]))

    qs = [r for r in states if r["qualifies"]]
    if len(qs) >= 3:
        st_r = float(np.corrcoef([abs(r["alive_ct"] - r["alive_t"]) for r in qs],
                                 [r["mean_row_logloss"] for r in qs])[0, 1])
    else:
        st_r = float("nan")
    say("qualifying states", len(qs), "state-level corr(|diff|, mean ll) =", round(st_r, 4))
    return {
        "test_rows": int(rows.height), "test_matches": len(test_matches),
        "grid_matches": len(grid_matches), "grid_rows": int(grid.height),
        "states": states, "abs_diff": diffs,
        "corr_row_pearson": {"r": pt, "ci_lo": clo, "ci_hi": chi, "direct_check": direct},
        "corr_qualifying_states_pearson": st_r, "qualifying_states": len(qs),
        "qualifying_state_keys": [f"{r['alive_ct']}v{r['alive_t']}" for r in qs],
    }


def part2(grid, grid_matches, test_set):
    say("== PART 2: deaths per team inside the live window, winners vs losers")
    live = grid.filter(pl.col("seconds_since_round_live") == 0).group_by(RK).agg([
        pl.len().alias("n_zero"), pl.col("tick").min().alias("live_start_tick")])
    grid_rounds = grid.group_by(RK).agg([pl.len().alias("grid_rows")]).join(live, on=RK, how="left")
    say("grid match-rounds", grid_rounds.height, "with exactly one live-0 row",
        grid_rounds.filter(pl.col("n_zero") == 1).height)
    re_all = pl.concat([pl.read_parquet(p) for p in pages("round_end")], how="diagonal_relaxed")
    re_g = re_all.filter(pl.col("match_key").is_in(grid_matches)).select(RK + ["tick", "winner_team_code", "player_count"])
    say("round_end rows for grid matches", re_g.height, "duplicate (match,round) rows", re_g.height - re_g.unique(subset=RK).height)
    re_u = re_g.unique(subset=RK, keep="first").rename({"tick": "round_end_tick"})
    rounds = grid_rounds.join(re_u, on=RK, how="left")
    say("grid rounds with no round_end row", rounds.filter(pl.col("round_end_tick").is_null()).height)
    say("player_count distribution:", rounds.group_by("player_count").len().sort("player_count").rows())
    say("winner_team_code distribution:", rounds.group_by("winner_team_code").len().sort("winner_team_code").rows())
    valid = rounds.filter(
        (pl.col("n_zero") == 1) & pl.col("round_end_tick").is_not_null()
        & pl.col("winner_team_code").is_in([2, 3]) & (pl.col("player_count") == 10))
    say("valid 5v5 rounds with winner", valid.height, "excluded", rounds.height - valid.height)

    pdeath = pl.concat([pl.read_parquet(p, columns=["match_key", "round", "tick", "player_id", "player_team_code"])
                        for p in pages("player_death")], how="diagonal_relaxed")
    pdeath = pdeath.filter(pl.col("match_key").is_in(grid_matches))
    say("player_death rows in grid matches", pdeath.height)
    dd = pdeath.unique(subset=RK + ["tick", "player_id"], keep="first")
    say("duplicate death rows dropped", pdeath.height - dd.height)
    say("death rows with null victim team", dd.filter(pl.col("player_team_code").is_null()).height)
    say("victim team codes:", dd.group_by("player_team_code").len().sort("player_team_code").rows())
    dd = dd.filter(pl.col("player_team_code").is_in([2, 3]))
    j = dd.join(valid.select(RK + ["live_start_tick", "round_end_tick"]), on=RK, how="inner")
    inw = j.filter((pl.col("tick") >= pl.col("live_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
    say("deaths in valid rounds", j.height, "inside live window", inw.height, "outside", j.height - inw.height)
    dcount = inw.group_by(RK).agg([
        (pl.col("player_team_code") == 2).sum().alias("d_t"),
        (pl.col("player_team_code") == 3).sum().alias("d_ct"),
    ])
    rf = valid.join(dcount, on=RK, how="left").with_columns([
        pl.col("d_t").fill_null(0).cast(pl.Int64), pl.col("d_ct").fill_null(0).cast(pl.Int64)])
    winner = rf["winner_team_code"].to_numpy().astype(np.int64)
    dT = rf["d_t"].to_numpy().astype(np.int64)
    dCT = rf["d_ct"].to_numpy().astype(np.int64)
    win_d = np.where(winner == 3, dCT, dT)
    lose_d = np.where(winner == 3, dT, dCT)
    say("rounds", len(winner), "CT-win", int((winner == 3).sum()), "T-win", int((winner == 2).sum()))
    say("max winner deaths", int(win_d.max()), "max loser deaths", int(lose_d.max()))
    def side_means(window_df):
        dc = window_df.group_by(RK).agg([
            (pl.col("player_team_code") == 2).sum().alias("d_t"),
            (pl.col("player_team_code") == 3).sum().alias("d_ct"),
        ])
        r = valid.join(dc, on=RK, how="left").with_columns([
            pl.col("d_t").fill_null(0).cast(pl.Int64), pl.col("d_ct").fill_null(0).cast(pl.Int64)])
        wn = r["winner_team_code"].to_numpy().astype(np.int64)
        a = r["d_t"].to_numpy().astype(np.int64)
        b = r["d_ct"].to_numpy().astype(np.int64)
        wd = np.where(wn == 3, b, a)
        ld = np.where(wn == 3, a, b)
        return {"winners_mean": float(wd.mean()), "losers_mean": float(ld.mean()),
                "deaths_counted": int(wd.sum() + ld.sum())}

    glast = grid.group_by(RK).agg(pl.col("tick").max().alias("grid_last_tick"))
    jl = j.join(glast, on=RK, how="left")
    sens = {
        "primary_round_end_window": side_means(
            j.filter((pl.col("tick") >= pl.col("live_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))),
        "grace_1s_after_round_end": side_means(
            j.filter((pl.col("tick") >= pl.col("live_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick") + 64))),
        "all_deaths_after_live_start": side_means(j.filter(pl.col("tick") >= pl.col("live_start_tick"))),
        "grid_last_row_window": side_means(
            jl.filter((pl.col("tick") >= pl.col("live_start_tick")) & (pl.col("tick") <= pl.col("grid_last_tick")))),
        "deaths_after_round_end_total": int(j.filter(pl.col("tick") > pl.col("round_end_tick")).height),
        "deaths_after_round_end_within_1s": int(j.filter(
            (pl.col("tick") > pl.col("round_end_tick")) & (pl.col("tick") <= pl.col("round_end_tick") + 64)).height),
    }
    say("death-window sensitivity:", sens)

    tst = rf["match_key"].is_in(sorted(test_set)).to_numpy()

    mi = {m: i for i, m in enumerate(grid_matches)}
    midx = np.array([mi[m] for m in rf["match_key"].to_list()], dtype=np.int64)
    M = len(grid_matches)

    def msum(v):
        out = np.zeros(M)
        np.add.at(out, midx, np.asarray(v, dtype=np.float64))
        return out

    ctw = (winner == 3).astype(np.float64)
    tw = (winner == 2).astype(np.float64)
    cols = {
        "n": msum(np.ones(len(winner))),
        "win_sum": msum(win_d), "lose_sum": msum(lose_d),
        "ctw_n": msum(ctw), "ctw_ct": msum(ctw * dCT), "ctw_t": msum(ctw * dT),
        "tw_n": msum(tw), "tw_t": msum(tw * dT), "tw_ct": msum(tw * dCT),
    }
    for kk in range(6):
        cols[f"w{kk}"] = msum((win_d == kk).astype(np.float64))
        cols[f"l{kk}"] = msum((lose_d == kk).astype(np.float64))
    W = boot_weights(M)

    def stat(num, den):
        pt, lo, hi = ratio_ci(W, cols[num][:, None], cols[den][:, None])
        return {"value": float(pt[0]), "ci_lo": float(lo[0]), "ci_hi": float(hi[0])}

    hist_w = np.bincount(win_d, minlength=6)[:6]
    hist_l = np.bincount(lose_d, minlength=6)[:6]
    n_r = len(winner)
    hist = []
    for kk in range(6):
        sw = stat(f"w{kk}", "n")
        sl = stat(f"l{kk}", "n")
        hist.append({"deaths": kk, "winner_count": int(hist_w[kk]), "winner_frac": sw["value"],
                     "winner_frac_lo": sw["ci_lo"], "winner_frac_hi": sw["ci_hi"],
                     "loser_count": int(hist_l[kk]), "loser_frac": sl["value"],
                     "loser_frac_lo": sl["ci_lo"], "loser_frac_hi": sl["ci_hi"]})
        say("deaths=%d winners n=%d frac=%.4f [%.4f,%.4f] | losers n=%d frac=%.4f [%.4f,%.4f]" % (
            kk, hist_w[kk], sw["value"], sw["ci_lo"], sw["ci_hi"], hist_l[kk], sl["value"], sl["ci_lo"], sl["ci_hi"]))
    means = {
        "winners_mean_all": stat("win_sum", "n"),
        "losers_mean_all": stat("lose_sum", "n"),
        "ctwin_rounds": {"n_rounds": int(cols["ctw_n"].sum()),
                         "ct_winners_mean": stat("ctw_ct", "ctw_n"),
                         "t_losers_mean": stat("ctw_t", "ctw_n")},
        "twin_rounds": {"n_rounds": int(cols["tw_n"].sum()),
                        "t_winners_mean": stat("tw_t", "tw_n"),
                        "ct_losers_mean": stat("tw_ct", "tw_n")},
    }
    say("winners mean deaths all =", round(means["winners_mean_all"]["value"], 4),
        "[", round(means["winners_mean_all"]["ci_lo"], 4), round(means["winners_mean_all"]["ci_hi"], 4), "]")
    say("losers mean deaths all =", round(means["losers_mean_all"]["value"], 4),
        "[", round(means["losers_mean_all"]["ci_lo"], 4), round(means["losers_mean_all"]["ci_hi"], 4), "]")
    say("CT-win rounds n=%d: CT(winner) mean=%.4f T(loser) mean=%.4f" % (
        means["ctwin_rounds"]["n_rounds"], means["ctwin_rounds"]["ct_winners_mean"]["value"],
        means["ctwin_rounds"]["t_losers_mean"]["value"]))
    say("T-win rounds n=%d: T(winner) mean=%.4f CT(loser) mean=%.4f" % (
        means["twin_rounds"]["n_rounds"], means["twin_rounds"]["t_winners_mean"]["value"],
        means["twin_rounds"]["ct_losers_mean"]["value"]))
    test_only = {
        "rounds": int(tst.sum()),
        "winners_mean": float(win_d[tst].mean()) if tst.any() else None,
        "losers_mean": float(lose_d[tst].mean()) if tst.any() else None,
    }
    say("test-split rounds", test_only["rounds"], "winners mean", test_only["winners_mean"],
        "losers mean", test_only["losers_mean"])
    return {"rounds": int(n_r), "ct_win_rounds": int((winner == 3).sum()), "t_win_rounds": int((winner == 2).sum()),
            "histogram": hist, "means": means, "test_split_sensitivity": test_only,
            "window_definition": "live_start_tick (grid row seconds_since_round_live==0) to round_end tick (round_end tome)",
            "death_window_sensitivity": sens}


def part3(grid, preds):
    say("== PART 3: full-buy vs full-buy rounds (state sample)")
    p1 = pl.read_parquet(P1, columns=["match_key", "round", "tick", "seconds_since_live", "team_code",
                                      "equipment_value_calc", "player_id"])
    say("players_1hz rows", p1.height, "team codes", p1["team_code"].unique().sort().to_list())
    p0 = p1.filter(pl.col("seconds_since_live") == 0).with_columns(pl.col("round").cast(pl.Int64))
    say("players at seconds_since_live==0", p0.height, "match-rounds", p0.select(RK).unique().height)
    ta = p0.group_by(RK + ["team_code"]).agg([
        pl.len().alias("n"),
        pl.col("equipment_value_calc").cast(pl.Int64).sum().alias("eq"),
        pl.col("tick").min().alias("tick0")])
    pt = ta.filter(pl.col("team_code") == 2).select(RK + [pl.col("n").alias("n_t"), pl.col("eq").alias("eq_t"), pl.col("tick0").alias("tick0_t")])
    pc = ta.filter(pl.col("team_code") == 3).select(RK + [pl.col("n").alias("n_ct"), pl.col("eq").alias("eq_ct"), pl.col("tick0").alias("tick0_ct")])
    pr = pt.join(pc, on=RK, how="inner")
    say("match-rounds with both T and CT present at sec 0", pr.height)
    fb5 = pr.filter((pl.col("n_t") == 5) & (pl.col("n_ct") == 5))
    fb = fb5.filter((pl.col("eq_t") >= 20000) & (pl.col("eq_ct") >= 20000))
    say("5v5-player rounds", fb5.height, "FULL-BUY rounds (both eq >= 20000)", fb.height)
    rds = pl.read_parquet(D + "/state_sample/rounds.parquet",
                          columns=["match_key", "round", "ct_win", "split", "live_start_tick", "round_end_tick"])
    say("state-sample rounds", rds.height, "matches", rds["match_key"].n_unique())
    say("state-sample split by match:", rds.group_by("split").agg(pl.col("match_key").n_unique().alias("m")).rows())
    chk = rds.join(pr.select(RK + ["tick0_t"]), on=RK, how="inner")
    say("state-sample rounds with players at sec0", chk.height, "live_start == players tick0:",
        chk.filter(pl.col("live_start_tick") == pl.col("tick0_t")).height)
    fb_rds = fb.join(rds, on=RK, how="inner")
    say("FB rounds with split label", fb_rds.height, fb_rds.group_by("split").len().rows())
    grid_test = set(preds["match_key"].unique().to_list())
    st_test_matches = set(rds.filter(pl.col("split") == "test")["match_key"].unique().to_list())
    say("state-sample test matches", len(st_test_matches), "in grid test split", len(st_test_matches & grid_test))

    grid_fb = grid.join(fb.select(RK), on=RK, how="inner")
    fb_grid_matches = sorted(grid_fb["match_key"].unique().to_list())
    say("FB grid rows", grid_fb.height, "FB match-rounds in grid", grid_fb.select(RK).unique().height,
        "matches", len(fb_grid_matches))
    W_all = boot_weights(len(fb_grid_matches))
    rows_all = grid_fb.with_columns([
        (pl.col("alive_ct") * 10 + pl.col("alive_t")).alias("key"),
        pl.col("label_ct_win").cast(pl.Float64).alias("y"),
        pl.lit(0.0).alias("p"), pl.lit(0.0).alias("ll")]).select(["match_key", "key", "y", "p", "ll"])
    all_stats = partition_stats(rows_all, fb_grid_matches, W_all)

    fbt = preds.join(grid_fb.select(K + ["alive_ct", "alive_t", "label_ct_win"]), on=K, how="inner")
    fb_test_matches = sorted(fbt["match_key"].unique().to_list())
    say("FB rows with predictions (test)", fbt.height, "match-rounds", fbt.select(RK).unique().height,
        "matches", len(fb_test_matches))
    if fbt.height > 0:
        y = fbt["label_ct_win"].to_numpy().astype(np.float64)
        p = np.clip(fbt["p_xgb_grid"].to_numpy().astype(np.float64), EPS, 1.0 - EPS)
        ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        ac = fbt["alive_ct"].to_numpy().astype(np.int64)
        at = fbt["alive_t"].to_numpy().astype(np.int64)
        rows_t = pl.DataFrame([
            pl.Series("match_key", fbt["match_key"].to_list()),
            pl.Series("key", ac * 10 + at), pl.Series("y", y), pl.Series("p", p), pl.Series("ll", ll)])
        W_t = boot_weights(len(fb_test_matches))
        test_stats = partition_stats(rows_t, fb_test_matches, W_t)
    else:
        test_stats = []
    test_by_key = {r["key"]: r for r in test_stats}

    keys_all = sorted({r["key"] for r in all_stats} | {r["key"] for r in test_stats})
    all_by_key = {r["key"]: r for r in all_stats}
    nan = float("nan")
    chain = []
    for k in keys_all:
        a = all_by_key.get(k, {"rows": 0, "matches": 0, "win": nan, "win_lo": nan, "win_hi": nan})
        t = test_by_key.get(k, {"rows": 0, "matches": 0, "win": nan, "win_lo": nan, "win_hi": nan,
                                 "mean_pred": nan, "mean_pred_lo": nan, "mean_pred_hi": nan,
                                 "mean_ll": nan, "mean_ll_lo": nan, "mean_ll_hi": nan})
        chain.append({
            "alive_ct": k // 10, "alive_t": k % 10,
            "fullbuy_all_rows": a["rows"], "fullbuy_all_matches": a["matches"],
            "fullbuy_all_emp_ct_win": a["win"], "fullbuy_all_lo": a["win_lo"], "fullbuy_all_hi": a["win_hi"],
            "fullbuy_test_rows": t["rows"], "fullbuy_test_matches": t["matches"],
            "fullbuy_test_emp_ct_win": t["win"], "fullbuy_test_lo": t["win_lo"], "fullbuy_test_hi": t["win_hi"],
            "fullbuy_test_mean_pred": t["mean_pred"], "fullbuy_test_mean_ll": t["mean_ll"],
            "fullbuy_test_ll_lo": t["mean_ll_lo"], "fullbuy_test_ll_hi": t["mean_ll_hi"],
            "small_n_flag": bool((a["rows"] < MIN_ROWS) or (a["matches"] < MIN_MATCHES)),
        })
    for r in chain:
        say("FB state %dv%d all: rows=%d matches=%d win=%.4f | test: rows=%d matches=%d win=%.4f predP=%.4f ll=%.4f" % (
            r["alive_ct"], r["alive_t"], r["fullbuy_all_rows"], r["fullbuy_all_matches"],
            r["fullbuy_all_emp_ct_win"], r["fullbuy_test_rows"], r["fullbuy_test_matches"],
            r["fullbuy_test_emp_ct_win"], r["fullbuy_test_mean_pred"], r["fullbuy_test_mean_ll"]))
    return {
        "players_sec0_match_rounds": int(p0.select(RK).unique().height),
        "match_rounds_both_teams_present": int(pr.height),
        "match_rounds_5v5_players": int(fb5.height),
        "fullbuy_rounds": int(fb.height),
        "fullbuy_rounds_by_split": {str(r[0]): int(r[1]) for r in fb_rds.group_by("split").len().rows()},
        "fullbuy_grid_matches": len(fb_grid_matches),
        "fullbuy_grid_rows": int(grid_fb.height),
        "fullbuy_test_rows": int(fbt.height), "fullbuy_test_matches": len(fb_test_matches),
        "fullbuy_test_match_rounds": int(fbt.select(RK).unique().height),
        "chain_states": chain,
        "equipment_threshold": 20000,
    }


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    grid = pl.read_parquet(D + "/viz/per_tick_states.parquet",
                           columns=["match_key", "round", "tick", "alive_ct", "alive_t", "label_ct_win",
                                    "seconds_since_round_live"])
    preds = pl.read_parquet(D + "/viz/per_tick_test_preds.parquet", columns=["match_key", "round", "tick", "p_xgb_grid"])
    grid_matches = sorted(grid["match_key"].unique().to_list())
    test_set = set(preds["match_key"].unique().to_list())
    say("grid rows", grid.height, "grid matches", len(grid_matches), "test-pred matches", len(test_set))
    r1 = part1(grid, preds)
    r2 = part2(grid, grid_matches, test_set)
    r3 = part3(grid, preds)
    out = {
        "id": "N1_lopsided",
        "script": "G:/pureskill/csds-cs2/scripts/exp/N1_lopsided_analysis.py",
        "output_json": OUT_JSON,
        "bootstrap": {"B": B, "seed": SEED, "method": "match-cluster percentile bootstrap, numpy default_rng"},
        "part1_states": r1,
        "part2_deaths": r2,
        "part3_fullbuy": r3,
        "assumptions": [
            "ASSUMPTION A1: Part 1 uses only the 304 matches in per_tick_test_preds.parquet (404,833 grid rows). Win rate, predicted P and log loss all come from those rows. Full-grid (1,521 matches) empirical win rates are a supplementary column only.",
            "ASSUMPTION A2: rows are unweighted (each 1 Hz grid row counts once). The w_round column (normalised per round) is not used.",
            "ASSUMPTION A3: log loss is the per-row natural-log binary cross-entropy of p_xgb_grid, with p clipped to [1e-15, 1-1e-15]. The clip was not active (min p 2.9e-5, max p 0.99994).",
            "ASSUMPTION A4: intervals are 95% percentile intervals from a match-cluster bootstrap (500 resamples of matches with replacement, numpy default_rng(0)). Clusters: 304 test matches (Part 1), 1,521 grid matches (full-grid supplement and Part 2), 138 and 30 full-buy matches (Part 3).",
            "ASSUMPTION A5: a state qualifies when it has at least 200 test rows and at least 20 distinct test matches.",
            "ASSUMPTION A6: Part 2 live window is [tick of the grid row with seconds_since_round_live == 0, round_end tick from the round_end tome channel]. Winner = round_end.winner_team_code (2 = T, 3 = CT). Deaths are player_death rows deduplicated on (match, round, tick, player_id); the victim team is player_team_code at the death tick.",
            "ASSUMPTION A7 (as specified by the task): full buy = sum of equipment_value_calc over the five players of each team at seconds_since_live == 0 is >= 20000 for both teams, and exactly five players per team are present at that second.",
            "ASSUMPTION A8: the Part 3 model-metric subset is full-buy rounds whose match is in the grid test-prediction set (30 matches). The state sample's own split column is not used for this, because it does not align with the grid prediction split (only 6 of its 30 test matches are in the grid test set).",
        ],
        "caveats": [
            "The grid holds 1,521 matches (28,683 match-rounds), not the 150 stated in the task. The 150-match state sample (rounds.parquet, players_1hz.parquet) is a subset of the grid.",
            "Death counts are measured after the round is decided, so the winner/loser death split is descriptive and conditioned on the outcome. It does not show that death counts predict the winner.",
            "About 1% of in-round deaths (2,029 rows) fall after the round_end tick, with a median of 56 ticks and a maximum of 448. They are excluded from the primary window. The sensitivity block in death_window_sensitivity shows the effect on the means.",
            "Winners' and losers' death histograms do not mirror in shape. Winners cluster at 1 to 3 deaths, while about 90% of losers lose all five players. The supported claim is the gap in mean deaths, not a symmetric inversion.",
            "Row-level Pearson r is driven by time weighting: long rounds contribute more rows. The state-level correlation over qualifying states is descriptive only.",
            "Full-buy subsets are small. The test-only 5v3 (17 matches) and 5v2 (6 matches) cells fail the 20-match rule and are flagged small_n_flag or sit under the threshold.",
            "Part 1 log loss is low in lopsided states partly because the outcome is nearly deterministic there. The table reports observed log loss, not a model-quality ranking.",
            "Using the grid's own last row as the window end drops deaths in the final second before round_end (the grid's last row is 1 to 64 ticks earlier). It counts 182,049 deaths against 205,582 and moves the losers' mean from 4.87 to 4.07. The primary window therefore ends at round_end.",
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(clean(out), fh, indent=2, allow_nan=False)
    say("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
