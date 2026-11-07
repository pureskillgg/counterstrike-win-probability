"""V1 independent verification of full-buy vs full-buy CT win claims.

Built from raw state-sample parquet only (players_1hz.parquet, rounds.parquet).
Does not read the analyst's script or output.
"""
import os
import json
import numpy as np
import polars as pl

ID = "V1_fullfull_verify"
BASE = r"G:\pureskill\csds-cs2\data"
PLAYERS = os.path.join(BASE, "state_sample", "players_1hz.parquet")
ROUNDS = os.path.join(BASE, "state_sample", "rounds.parquet")
OUT_JSON = os.path.join(BASE, "viz", ID + ".json")

N_BOOT = 500
SEED = 0
THRESHOLDS = [17000, 20000, 23000]
HEADLINE_THR = 20000
MIN_MAP_N = 40
GAP = 3000
T_CODE = 2   # ASSUMPTION from task statement; cross-checked below
CT_CODE = 3


def log(*a):
    print(*a, flush=True)


out = {
    "id": ID,
    "assumptions": [
        "ASSUMPTION: team_code 2 = T, 3 = CT (task statement). Cross-checked with has_defuser (CT only) and inv_c4 (T only).",
        "ASSUMPTION: team equipment = sum of equipment_value_calc over the five players of that team at seconds_since_live == 0.",
        "ASSUMPTION: clean round = exactly 10 player rows at seconds_since_live==0, 10 distinct player_ids, one tick, 5 team-2 and 5 team-3 rows.",
        "ASSUMPTION: match-cluster bootstrap resamples whole matches with replacement from all clean matches; numpy default_rng(0); 500 replicates; percentile 95% interval; same draws reused for every statistic.",
        "ASSUMPTION: Spearman rho = Pearson correlation of average (tie-averaged) ranks of CT-minus-T equipment difference and CT win indicator.",
        "ASSUMPTION: quintile edges = sample quantiles 0.2/0.4/0.6/0.8 of the difference among full-vs-full rounds; bins assigned with searchsorted(side=right).",
        "ASSUMPTION: ct_win = 1 means CT won the round (rounds.parquet).",
    ],
    "checks": {},
    "numbers": {},
    "tables": {},
}


# ----------------------------------------------------------------- helpers
def rank_avg(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, first, inv, counts = np.unique(xs, return_index=True, return_inverse=True, return_counts=True)
    avg = first + (counts + 1) / 2.0  # 0-based first position + mean offset -> 1-based average rank
    r = np.empty(n, dtype=float)
    r[order] = avg[inv.ravel()]
    return r


def spearman(x, y):
    rx = rank_avg(x)
    ry = rank_avg(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den == 0:
        return float("nan")
    return float((rx * ry).sum() / den)


def pct_ci(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ----------------------------------------------------------------- 1. rounds
rounds = pl.read_parquet(ROUNDS).with_columns(pl.col("round").cast(pl.Int64))
out["checks"]["rounds_rows"] = rounds.height
out["checks"]["rounds_unique_match_round"] = rounds.select(["match_key", "round"]).unique().height
out["checks"]["rounds_n_matches"] = rounds["match_key"].n_unique()
out["checks"]["round_eq_round_number_all"] = bool((rounds["round"] == rounds["round_number"]).all())
out["checks"]["ct_win_values"] = sorted(set(rounds["ct_win"].to_list()))
out["checks"]["all_rounds_ct_win_rate_raw"] = float(rounds["ct_win"].mean())
out["checks"]["map_counts_all_rounds"] = {
    row["map_name"]: row["n"]
    for row in rounds.group_by("map_name").agg(pl.len().alias("n")).to_dicts()
}
out["checks"]["split_counts_rounds"] = {
    row["split"]: row["n"]
    for row in rounds.group_by("split").agg(pl.len().alias("n")).to_dicts()
}

# score-progression consistency: ct_score(next) - ct_score(now) should equal ct_win(now)
r2 = rounds.sort(["match_key", "round"]).with_columns(
    pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"),
    pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
)
r2 = r2.filter(pl.col("ct_next").is_not_null())
ok_ct = (r2["ct_next"] - r2["ct_score"] == r2["ct_win"]).sum()
ok_t = (r2["t_next"] - r2["t_score"] == (1 - r2["ct_win"])).sum()
out["checks"]["score_progression_pairs"] = r2.height
out["checks"]["score_progression_ct_ok"] = int(ok_ct)
out["checks"]["score_progression_t_ok"] = int(ok_t)
log("rounds rows", rounds.height, "| score checks ct_ok", ok_ct, "t_ok", ok_t, "of", r2.height)

# ----------------------------------------------------------------- 2. side-code sanity on all seconds
side_chk = (
    pl.scan_parquet(PLAYERS)
    .select([
        pl.col("team_code"),
        pl.col("has_defuser").cast(pl.Int64).alias("defuser"),
        (pl.col("inv_c4") > 0).cast(pl.Int64).alias("c4"),
    ])
    .group_by("team_code")
    .agg([
        pl.len().alias("n_rows"),
        pl.col("defuser").sum().alias("n_defuser_true"),
        pl.col("c4").sum().alias("n_c4_held"),
    ])
    .sort("team_code")
    .collect()
)
out["checks"]["side_code_table_all_seconds"] = side_chk.to_dicts()
log("side code table:\n", side_chk)

# ----------------------------------------------------------------- 3. team equipment at seconds_since_live == 0
p0 = (
    pl.scan_parquet(PLAYERS)
    .filter(pl.col("seconds_since_live") == 0)
    .select(["match_key", "round", "tick", "player_id", "team_code", "equipment_value_calc"])
    .collect()
    .with_columns(pl.col("round").cast(pl.Int64))
)
out["checks"]["seconds0_rows"] = p0.height
out["checks"]["seconds0_distinct_match_round"] = p0.select(["match_key", "round"]).unique().height
out["checks"]["seconds0_team_codes"] = sorted(set(p0["team_code"].to_list()))

ag = p0.group_by(["match_key", "round"]).agg([
    pl.len().alias("n_rows"),
    pl.col("player_id").n_unique().alias("n_pid"),
    pl.col("tick").n_unique().alias("n_tick"),
    (pl.col("team_code") == CT_CODE).sum().alias("n_ct"),
    (pl.col("team_code") == T_CODE).sum().alias("n_t"),
    pl.col("equipment_value_calc").filter(pl.col("team_code") == CT_CODE).sum().alias("ct_eq"),
    pl.col("equipment_value_calc").filter(pl.col("team_code") == T_CODE).sum().alias("t_eq"),
    pl.col("tick").min().alias("tick0"),
])
j = rounds.join(ag, on=["match_key", "round"], how="left")
rows = j.to_dicts()

reasons = {"no_seconds0_rows": 0, "not_10_players": 0, "not_5v5": 0, "multi_tick": 0, "ok": 0}
recs = []
tick_match = 0
for r in rows:
    if r["n_rows"] is None:
        reasons["no_seconds0_rows"] += 1
        continue
    if r["n_rows"] != 10 or r["n_pid"] != 10:
        reasons["not_10_players"] += 1
        continue
    if r["n_ct"] != 5 or r["n_t"] != 5:
        reasons["not_5v5"] += 1
        continue
    if r["n_tick"] != 1:
        reasons["multi_tick"] += 1
        continue
    reasons["ok"] += 1
    if r["tick0"] == r["live_start_tick"]:
        tick_match += 1
    recs.append(r)
out["checks"]["clean_round_filter_reasons"] = reasons
out["checks"]["seconds0_tick_equals_live_start_tick_among_clean"] = tick_match
log("clean filter:", reasons, "| tick0==live_start_tick:", tick_match)

mkeys = sorted({r["match_key"] for r in recs})
mindex = {k: i for i, k in enumerate(mkeys)}
M = len(mkeys)
R = len(recs)
mi = np.array([mindex[r["match_key"]] for r in recs], dtype=np.int64)
y = np.array([float(r["ct_win"]) for r in recs])
maps = np.array([r["map_name"] for r in recs])
ct_eq = np.array([float(r["ct_eq"]) for r in recs])
t_eq = np.array([float(r["t_eq"]) for r in recs])
d = ct_eq - t_eq
splits = np.array([r["split"] for r in recs])
out["checks"]["clean_rounds"] = R
out["checks"]["clean_matches"] = M
log("clean rounds", R, "matches", M)

# ----------------------------------------------------------------- 4. bootstrap draws (shared by every statistic)
rng = np.random.default_rng(SEED)
draws = rng.integers(0, M, size=(N_BOOT, M))
C = np.empty((N_BOOT, M), dtype=np.float64)
for b in range(N_BOOT):
    C[b] = np.bincount(draws[b], minlength=M)
W = C[:, mi]  # (N_BOOT, R) weight of each round in each replicate


def sub_rate(sub, vals):
    """Point estimate and bootstrap draws of mean(vals) over rounds in sub."""
    sf = sub.astype(float)
    den = W @ sf
    num = W @ (sf * vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = num / den
    point = float((sf * vals).sum() / sf.sum()) if sf.sum() > 0 else float("nan")
    return point, bt


def boot_spearman(fmask):
    idx_f = np.where(fmask)[0]
    Wf = W[:, idx_f].astype(np.int64)
    dd = d[idx_f]
    yy = y[idx_f]
    res = np.empty(N_BOOT)
    for b in range(N_BOOT):
        rep = np.repeat(np.arange(idx_f.size), Wf[b])
        res[b] = spearman(dd[rep], yy[rep])
    return res


# all-round CT rate
all_mask = np.ones(R, dtype=bool)
all_pt, all_bt = sub_rate(all_mask, y)
out["numbers"]["ct_win_all_rounds"] = all_pt
out["numbers"]["ct_win_all_rounds_ci"] = pct_ci(all_bt)
log(f"ALL rounds CT rate {all_pt:.4f} n={R} ci={pct_ci(all_bt)}")

# ----------------------------------------------------------------- 5. per-threshold analysis
thr_out = {}
for thr in THRESHOLDS:
    fvf = (ct_eq >= thr) & (t_eq >= thr)
    idx_f = np.where(fvf)[0]
    n_f = int(fvf.sum())
    pt, bt = sub_rate(fvf, y)
    _, bt_all = sub_rate(all_mask, y)
    diff_bt = bt - bt_all
    diff_pt = pt - all_pt
    spear_pt = spearman(d[idx_f], y[idx_f])
    spear_bt = boot_spearman(fvf)
    rec = {
        "threshold": thr,
        "n_rounds": n_f,
        "n_matches": int(np.unique(mi[fvf]).size),
        "ct_win_fvf": pt,
        "ct_win_fvf_ci": pct_ci(bt),
        "fvf_minus_0.5": pt - 0.5,
        "fvf_minus_0.5_ci": pct_ci(bt - 0.5),
        "fvf_minus_all_rounds": diff_pt,
        "fvf_minus_all_rounds_ci": pct_ci(diff_bt),
        "spearman_diff_vs_ctwin": spear_pt,
        "spearman_ci": pct_ci(spear_bt),
        "n_split_test": int((fvf & (splits == "test")).sum()),
        "n_split_train": int((fvf & (splits == "train")).sum()),
    }
    thr_out[thr] = rec
    log(f"thr {thr}: n={n_f} matches={rec['n_matches']} CT={pt:.4f} ci={rec['ct_win_fvf_ci']} "
        f"fvf-all={diff_pt:+.4f} ci={rec['fvf_minus_all_rounds_ci']} rho={spear_pt:.4f} ci={rec['spearman_ci']}")

    # full-vs-full diff shares
    if thr == HEADLINE_THR or True:
        gt = fvf & (d > GAP)
        lt = fvf & (d < -GAP)
        within = fvf & (np.abs(d) <= GAP)
        gt_pt, gt_bt = sub_rate(gt, np.ones(R))  # share of fvf rounds with gt
        # share relative to fvf denominators via sub_rate on fvf subset of gt indicator
        s_gt_pt = float(gt.sum() / n_f)
        s_gt_bt = (W @ gt.astype(float)) / (W @ fvf.astype(float))
        s_lt_pt = float(lt.sum() / n_f)
        s_lt_bt = (W @ lt.astype(float)) / (W @ fvf.astype(float))
        s_within_pt = float(within.sum() / n_f)
        s_within_bt = (W @ within.astype(float)) / (W @ fvf.astype(float))
        rec["share_ct_outspends_gt3000"] = s_gt_pt
        rec["share_ct_outspends_gt3000_ci"] = pct_ci(s_gt_bt)
        rec["share_t_outspends_gt3000"] = s_lt_pt
        rec["share_t_outspends_gt3000_ci"] = pct_ci(s_lt_bt)
        rec["share_within_3000"] = s_within_pt
        rec["share_within_3000_ci"] = pct_ci(s_within_bt)
        rec["count_ct_gt3000"] = int(gt.sum())
        rec["count_t_gt3000"] = int(lt.sum())
        rec["count_within3000"] = int(within.sum())
        rec["median_abs_diff_fvf"] = float(np.median(np.abs(d[idx_f])))
        rec["max_abs_diff_fvf"] = float(np.max(np.abs(d[idx_f])))
        rec["min_team_eq_fvf"] = float(min(np.min(ct_eq[idx_f]), np.min(t_eq[idx_f])))

    # quintile table (rounds in fvf only)
    if True:
        dfv = d[idx_f]
        edges = np.quantile(dfv, [0.2, 0.4, 0.6, 0.8])
        bins = np.searchsorted(edges, dfv, side="right")
        q_rows = []
        for k in range(5):
            sel = np.zeros(R, dtype=bool)
            sel[idx_f[bins == k]] = True
            if sel.sum() == 0:
                q_rows.append({"bin": k + 1, "n": 0})
                continue
            qpt, qbt = sub_rate(sel, y)
            q_rows.append({
                "bin": k + 1,
                "n_rounds": int(sel.sum()),
                "n_matches": int(np.unique(mi[sel]).size),
                "diff_min": float(d[sel].min()),
                "diff_max": float(d[sel].max()),
                "ct_win_rate": qpt,
                "ct_win_ci": pct_ci(qbt),
            })
        rec["quintile_edges"] = [float(e) for e in edges]
        rec["quintiles"] = q_rows
        rec["quintile_bin_sizes"] = [int(r.get("n_rounds", 0)) for r in q_rows]
        log(f"  thr {thr} quintile edges {edges} sizes {rec['quintile_bin_sizes']}")
        for q in q_rows:
            if q.get("n_rounds"):
                log(f"    Q{q['bin']} n={q['n_rounds']} m={q['n_matches']} diff[{q['diff_min']:.0f},{q['diff_max']:.0f}] "
                    f"CT={q['ct_win_rate']:.4f} ci={q['ct_win_ci']}")

    # map table (fvf only, and all-round map rates)
    if True:
        map_rows = []
        for mname in sorted(set(maps.tolist())):
            mm = maps == mname
            fm = fvf & mm
            n_fm = int(fm.sum())
            row = {"map": mname, "n_fvf_rounds": n_fm, "n_fvf_matches": int(np.unique(mi[fm]).size) if n_fm else 0,
                   "n_all_rounds_map": int(mm.sum())}
            all_map_pt, _ = sub_rate(mm, y)
            row["ct_win_all_rounds_map"] = all_map_pt
            if n_fm >= MIN_MAP_N:
                mpt, mbt = sub_rate(fm, y)
                row["ct_win_fvf"] = mpt
                row["ct_win_fvf_ci"] = pct_ci(mbt)
                row["fvf_minus_0.5"] = mpt - 0.5
                row["fvf_minus_0.5_ci"] = pct_ci(mbt - 0.5)
            else:
                row["ct_win_fvf"] = "n<40, n only"
            map_rows.append(row)
        rec["map_table"] = map_rows
        for mr in map_rows:
            if isinstance(mr.get("ct_win_fvf"), float):
                log(f"    map {mr['map']} n={mr['n_fvf_rounds']} m={mr['n_fvf_matches']} CT={mr['ct_win_fvf']:.4f} ci={mr['ct_win_fvf_ci']}")
            else:
                log(f"    map {mr['map']} n={mr['n_fvf_rounds']} (n<40, rate not reported)")

    # ----- alternative bootstrap: resample only the matches that have full-vs-full rounds
    if thr == HEADLINE_THR:
        fm_list = np.unique(mi[fvf])
        Mf = fm_list.size
        rng2 = np.random.default_rng(SEED)
        dr = rng2.integers(0, Mf, size=(N_BOOT, Mf))
        Calt = np.empty((N_BOOT, Mf))
        for b in range(N_BOOT):
            Calt[b] = np.bincount(dr[b], minlength=Mf)
        loc = {m: i for i, m in enumerate(fm_list)}
        mi_f = np.array([loc[m] for m in mi[fvf]])
        Wf_alt = Calt[:, mi_f]
        alt_bt = (Wf_alt * y[fvf]).sum(1) / Wf_alt.sum(1)
        rec["alt_boot_fvf_matches_only_ci"] = pct_ci(alt_bt)

    # rounds per match among fvf
    if thr == HEADLINE_THR:
        per_match = np.bincount(mi[fvf], minlength=M)
        per_match = per_match[per_match > 0]
        rec["fvf_rounds_per_match_median"] = float(np.median(per_match))
        rec["fvf_rounds_per_match_max"] = int(per_match.max())

    # 1-sided bootstrap check of fvf win > 0.5
    thr_out[thr] = rec

out["numbers"]["by_threshold"] = {str(k): v for k, v in thr_out.items()}

# headline convenience numbers
h = thr_out[HEADLINE_THR]
out["numbers"]["ct_win_fullvsfull_T20000"] = h["ct_win_fvf"]
out["numbers"]["ct_win_fullvsfull_T20000_ci"] = h["ct_win_fvf_ci"]
out["numbers"]["fullvsfull_minus_all_rounds_T20000"] = h["fvf_minus_all_rounds"]
out["numbers"]["fullvsfull_minus_all_rounds_T20000_ci"] = h["fvf_minus_all_rounds_ci"]
out["numbers"]["spearman_rho_diff_vs_ctwin_T20000"] = h["spearman_diff_vs_ctwin"]
out["numbers"]["spearman_rho_diff_vs_ctwin_T20000_ci"] = h["spearman_ci"]
out["numbers"]["spearman_rho_diff_vs_ctwin_T23000"] = thr_out[23000]["spearman_diff_vs_ctwin"]
out["numbers"]["spearman_rho_diff_vs_ctwin_T23000_ci"] = thr_out[23000]["spearman_ci"]
qh = h["quintiles"]
for q in qh:
    if q.get("n_rounds"):
        out["numbers"][f"quintile_Q{q['bin']}_T20000"] = q["ct_win_rate"]
        out["numbers"][f"quintile_Q{q['bin']}_T20000_ci"] = q["ct_win_ci"]
        out["numbers"][f"quintile_Q{q['bin']}_T20000_n"] = q["n_rounds"]

# ----------------------------------------------------------------- 6. write output
out["tables"]["headline_threshold_20000_full"] = h
out["tables"]["threshold_17000_summary"] = {k: v for k, v in thr_out[17000].items() if k not in ("quintiles", "map_table")}
out["tables"]["threshold_23000_summary"] = {k: v for k, v in thr_out[23000].items() if k not in ("quintiles", "map_table")}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("wrote", OUT_JSON)
