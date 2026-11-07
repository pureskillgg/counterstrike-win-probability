"""N2_fullfull: CT win rate in full-buy vs full-buy rounds (state sample, 1 Hz).

Team equipment = sum of equipment_value_calc over alive players at
seconds_since_live == 0 (the first live second). Full buy = team sum >= threshold.
Clusters = match_key. Match-cluster percentile bootstrap, 500 resamples, seed 0, numpy only.
"""
import json
import math
import sys

import numpy as np
import polars as pl

BASE = r"G:\pureskill\csds-cs2\data\state_sample"
OUT_JSON = r"G:\pureskill\csds-cs2\data\viz\N2_fullfull.json"
SCRIPT_PATH = r"G:\pureskill\csds-cs2\scripts\exp\N2_fullfull_analysis.py"

B = 500
SEED = 0
THRESHOLDS = [17000, 20000, 23000]
MAIN_T = 20000
MIN_N_MAP = 40
GAP = 3000
CT_TEAM = 3
T_TEAM = 2


def avg_rank(x):
    x = np.asarray(x, dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    cum = np.cumsum(counts)
    avg = (cum - counts + 1 + cum) / 2.0  # average of the 1-based ranks in each tie block
    return avg[inv]


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if den == 0.0:
        return float("nan")
    return float((a * b).sum() / den)


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return pearson(avg_rank(x), avg_rank(y))


def pct_ci(vals):
    v = np.asarray(vals, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return o


def fmt(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main():
    # ---------- 1. team equipment at the first live second ----------
    players = (
        pl.scan_parquet(BASE + r"\players_1hz.parquet")
        .filter(pl.col("seconds_since_live") == 0)
        .select(
            "match_key",
            pl.col("round").cast(pl.Int64),
            pl.col("team_code").cast(pl.Int64),
            pl.col("equipment_value_calc").cast(pl.Int64),
        )
        .collect()
    )
    print("snapshot-0 player rows:", players.height,
          "| null equipment_value_calc:", players["equipment_value_calc"].null_count())

    team = players.group_by(["match_key", "round", "team_code"]).agg(
        pl.col("equipment_value_calc").sum().alias("eq"),
        pl.len().cast(pl.Int64).alias("n_alive"),
        pl.col("equipment_value_calc").is_not_null().sum().cast(pl.Int64).alias("n_valid"),
    )
    ct = team.filter(pl.col("team_code") == CT_TEAM).select(
        "match_key", "round",
        pl.col("eq").alias("ct_eq"), pl.col("n_alive").alias("ct_n"), pl.col("n_valid").alias("ct_nv"),
    )
    tt = team.filter(pl.col("team_code") == T_TEAM).select(
        "match_key", "round",
        pl.col("eq").alias("t_eq"), pl.col("n_alive").alias("t_n"), pl.col("n_valid").alias("t_nv"),
    )

    # ---------- 2. rounds (outcome, map, clean-round flags) ----------
    rounds = pl.read_parquet(BASE + r"\rounds.parquet").select(
        "match_key",
        pl.col("round").cast(pl.Int64),
        pl.col("ct_win").cast(pl.Int64),
        "map_name",
        "split",
        "alive_gap_snapshots",
        "live_start_tick",
    )
    n_round_dupes = rounds.height - rounds.select("match_key", "round").unique().height
    print("rounds rows:", rounds.height, "| duplicate (match_key, round):", n_round_dupes)
    print("team-side orphans (no rounds row): CT", ct.join(rounds.select("match_key", "round"),
          on=["match_key", "round"], how="anti").height,
          "| T", tt.join(rounds.select("match_key", "round"), on=["match_key", "round"], how="anti").height)

    df = (
        rounds.join(ct, on=["match_key", "round"], how="left")
        .join(tt, on=["match_key", "round"], how="left")
    )
    n_no_ct = df.filter(pl.col("ct_eq").is_null()).height
    n_no_t = df.filter(pl.col("t_eq").is_null()).height
    print("rounds with no CT snapshot-0 rows:", n_no_ct, "| no T snapshot-0 rows:", n_no_t)

    df = df.with_columns(
        pl.col(["ct_eq", "t_eq", "ct_n", "t_n", "ct_nv", "t_nv"]).fill_null(0),
    ).with_columns(
        ((pl.col("ct_n") == 5) & (pl.col("t_n") == 5)
         & (pl.col("ct_nv") == 5) & (pl.col("t_nv") == 5)).alias("complete"),
    ).sort(["match_key", "round"])

    gap_rows = df.filter(pl.col("alive_gap_snapshots") > 0).select(
        "match_key", "round", "alive_gap_snapshots", "complete", "ct_n", "t_n")
    print("rounds with alive_gap_snapshots > 0:")
    print(gap_rows)
    n_incomplete = df.filter(~pl.col("complete")).height
    print("rounds NOT complete at snapshot 0 (excluded from team-level analysis):", n_incomplete)

    # ---------- 3. arrays ----------
    match_keys = sorted(df["match_key"].unique().to_list())
    M = len(match_keys)
    idx_map = {m: i for i, m in enumerate(match_keys)}
    mi = np.array([idx_map[m] for m in df["match_key"].to_list()], dtype=np.int64)
    n_rows = df.height
    ctwin = df["ct_win"].to_numpy().astype(np.float64)
    map_arr = np.asarray(df["map_name"].to_list())
    ct_eq = df["ct_eq"].to_numpy().astype(np.float64)
    t_eq = df["t_eq"].to_numpy().astype(np.float64)
    complete = df["complete"].to_numpy().astype(bool)
    diff_full = ct_eq - t_eq
    absd_full = np.abs(diff_full)
    print("clusters (matches):", M, "| rounds:", n_rows, "| CT win share all rounds:", round(float(ctwin.mean()), 6))

    # ---------- 4. bootstrap draws (shared across all statistics) ----------
    rng = np.random.default_rng(SEED)
    DRAWS = rng.integers(0, M, size=(B, M))

    def group_boot(mask, outcome):
        n_m = np.bincount(mi[mask], minlength=M).astype(np.float64)
        w_m = np.bincount(mi[mask], weights=outcome[mask], minlength=M).astype(np.float64)
        n_tot = float(n_m.sum())
        point = float(w_m.sum() / n_tot) if n_tot > 0 else float("nan")
        nb = n_m[DRAWS].sum(axis=1)
        wb = w_m[DRAWS].sum(axis=1)
        rb = np.where(nb > 0, wb / np.where(nb > 0, nb, 1.0), np.nan)
        lo, hi = pct_ci(rb)
        res = {
            "rate": point,
            "ci_low": lo,
            "ci_high": hi,
            "n_rounds": int(n_tot),
            "n_matches": int((n_m > 0).sum()),
            "boot_nan_draws": int(np.isnan(rb).sum()),
        }
        return res, rb

    def boot_spearman(x, y, mi_sub):
        n_sub = len(x)
        rhos = np.full(B, np.nan)
        for b in range(B):
            counts_b = np.bincount(DRAWS[b], minlength=M)
            w = counts_b[mi_sub]
            rep = np.repeat(np.arange(n_sub), w)
            if rep.size >= 3:
                rhos[b] = spearman(x[rep], y[rep])
        return pct_ci(rhos), rhos

    numbers = []
    table = []
    tables_by_t = {}

    def add_number(name, res, n_key="n_rounds"):
        numbers.append({
            "name": name,
            "value": res["rate"],
            "ci95_low": res["ci_low"],
            "ci95_high": res["ci_high"],
            "n_matches": res["n_matches"],
            "n_rows": res["n_rounds"],
        })

    def add_row(section, group, res):
        table.append({
            "section": section,
            "group": group,
            "n_rounds": res["n_rounds"],
            "n_matches": res["n_matches"],
            "ct_win_rate": res["rate"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
        })

    all_mask = np.ones(n_rows, dtype=bool)
    all_res, all_rb = group_boot(all_mask, ctwin)

    results_by_t = {}
    for T in THRESHOLDS:
        print("\n" + "=" * 100)
        print(f"THRESHOLD full buy >= {T} (both teams)")
        print("=" * 100)
        ff = complete & (ct_eq >= T) & (t_eq >= T)
        out = {"threshold": T}

        # ---- (1) overall full-vs-full CT win rate
        ff_res, ff_rb = group_boot(ff, ctwin)
        out["item1_overall_fullvsfull"] = ff_res
        out["item1_all_rounds_comparator"] = all_res
        dlo, dhi = pct_ci(ff_rb - all_rb)
        out["item1_fullvsfull_minus_all_rounds"] = {
            "diff": ff_res["rate"] - all_res["rate"], "ci_low": dlo, "ci_high": dhi}
        out["item1_fullvsfull_vs_50"] = {
            "diff": ff_res["rate"] - 0.5, "ci_low": ff_res["ci_low"] - 0.5, "ci_high": ff_res["ci_high"] - 0.5,
            "ci_includes_0.5": bool(ff_res["ci_low"] <= 0.5 <= ff_res["ci_high"])}
        print(f"(1) full-vs-full: CT win {fmt(ff_res['rate'])} [{fmt(ff_res['ci_low'])}, {fmt(ff_res['ci_high'])}] "
              f"n_rounds={ff_res['n_rounds']} n_matches={ff_res['n_matches']} boot_nan={ff_res['boot_nan_draws']}")
        print(f"    all rounds: CT win {fmt(all_res['rate'])} [{fmt(all_res['ci_low'])}, {fmt(all_res['ci_high'])}] "
              f"n_rounds={all_res['n_rounds']}")
        print(f"    full-vs-full minus all: {fmt(ff_res['rate'] - all_res['rate'])} [{fmt(dlo)}, {fmt(dhi)}]")
        ci_inc = out["item1_fullvsfull_vs_50"]["ci_includes_0.5"]
        print(f"    CI includes 0.5: {ci_inc}")
        add_number(f"ct_win_fullvsfull_T{T}", ff_res)
        add_number(f"ct_win_all_rounds_T{T}", all_res)
        add_row("overall", f"full-vs-full T{T}", ff_res)
        add_row("overall", "all rounds (comparator)", all_res)

        # ---- (1) and (3): by map
        maps_rows = []
        print("    by map (full-vs-full):")
        for m in sorted(set(map_arr.tolist())):
            mm = ff & (map_arr == m)
            r_res, r_rb = group_boot(mm, ctwin)
            if r_res["n_rounds"] < MIN_N_MAP:
                maps_rows.append({"map_name": m, "n_rounds": r_res["n_rounds"],
                                  "n_matches": r_res["n_matches"],
                                  "note": "n only (fewer than 40 qualifying rounds)"})
                print(f"      {m}: n_rounds={r_res['n_rounds']} n_matches={r_res['n_matches']} (n only)")
                continue
            dmlo, dmhi = pct_ci(r_rb - ff_rb)
            row = {"map_name": m, **r_res,
                   "adv_vs_50": r_res["rate"] - 0.5,
                   "diff_vs_fullvsfull_overall": r_res["rate"] - ff_res["rate"],
                   "diff_ci_low": dmlo, "diff_ci_high": dmhi}
            maps_rows.append(row)
            add_number(f"ct_win_fullvsfull_T{T}_{m}", r_res)
            add_row("by_map", f"{m} T{T}", r_res)
            print(f"      {m}: CT win {fmt(r_res['rate'])} [{fmt(r_res['ci_low'])}, {fmt(r_res['ci_high'])}] "
                  f"n_rounds={r_res['n_rounds']} n_matches={r_res['n_matches']} "
                  f"minus overall {fmt(row['diff_vs_fullvsfull_overall'])} [{fmt(dmlo)}, {fmt(dmhi)}]")
        out["item1_by_map"] = maps_rows

        # ---- (2) quintiles of CT-minus-T equipment difference within full-vs-full
        dff = diff_full[ff]
        ctw_ff = ctwin[ff]
        mi_ff = mi[ff]
        edges = np.quantile(dff, [0.2, 0.4, 0.6, 0.8])
        qb = np.full(n_rows, -1, dtype=np.int64)
        qb[ff] = np.searchsorted(edges, dff, side="right")
        quint_rows = []
        print("    quintiles of (CT eq - T eq) within full-vs-full:")
        for q in range(5):
            mq = ff & (qb == q)
            q_res, _ = group_boot(mq, ctwin)
            row = {
                "bin": q + 1,
                "diff_min": float(diff_full[mq].min()) if mq.any() else None,
                "diff_max": float(diff_full[mq].max()) if mq.any() else None,
                "diff_mean": float(diff_full[mq].mean()) if mq.any() else None,
                **q_res,
            }
            quint_rows.append(row)
            add_row("quintile", f"Q{q + 1} T{T}", q_res)
            print(f"      Q{q + 1} diff [{row['diff_min']}, {row['diff_max']}] mean={fmt(row['diff_mean'])}: "
                  f"CT win {fmt(q_res['rate'])} [{fmt(q_res['ci_low'])}, {fmt(q_res['ci_high'])}] "
                  f"n_rounds={q_res['n_rounds']} n_matches={q_res['n_matches']}")
        out["item2_quintiles"] = quint_rows
        rho = spearman(dff, ctw_ff)
        (rlo, rhi), _ = boot_spearman(dff, ctw_ff, mi_ff)
        out["item2_spearman_rank_corr"] = {"rho": rho, "ci_low": rlo, "ci_high": rhi,
                                           "n_rounds": int(ff.sum()), "n_matches": int(len(np.unique(mi_ff)))}
        print(f"    Spearman (avg ranks) diff vs CT win: rho={fmt(rho)} [{fmt(rlo)}, {fmt(rhi)}]")
        bal = ff & (absd_full <= GAP)
        bal_res, _ = group_boot(bal, ctwin)
        out["extra_balanced_abs_diff_le_3000"] = bal_res
        add_row("balanced", f"abs diff <= 3000 T{T}", bal_res)
        print(f"    balanced (|diff| <= 3000): CT win {fmt(bal_res['rate'])} "
              f"[{fmt(bal_res['ci_low'])}, {fmt(bal_res['ci_high'])}] n_rounds={bal_res['n_rounds']}")

        # ---- (4) how often the two full-buy teams differ by more than 3000
        gap_flag = (absd_full > GAP).astype(np.float64)
        g_res, _ = group_boot(ff, gap_flag)
        up_res, _ = group_boot(ff, (diff_full > GAP).astype(np.float64))
        dn_res, _ = group_boot(ff, (diff_full < -GAP).astype(np.float64))
        absff = absd_full[ff]
        out["item4_gap_gt_3000"] = {
            **g_res,
            "ct_higher_by_gt_3000": up_res,
            "t_higher_by_gt_3000": dn_res,
            "abs_diff_quantiles": {str(p): float(np.quantile(absff, p)) for p in [0.25, 0.5, 0.75, 0.9]},
            "abs_diff_mean": float(absff.mean()),
        }
        add_row("gap", f"share |diff|>3000 T{T}", g_res)
        print(f"(4) |CT eq - T eq| > 3000 in full-vs-full: {fmt(g_res['rate'])} "
              f"[{fmt(g_res['ci_low'])}, {fmt(g_res['ci_high'])}] n_rounds={g_res['n_rounds']}; "
              f"CT higher by >3000: {fmt(up_res['rate'])}; T higher by >3000: {fmt(dn_res['rate'])}; "
              f"median |diff|={out['item4_gap_gt_3000']['abs_diff_quantiles']['0.5']}")

        results_by_t[str(T)] = out

    # ---------- 5. cross-check against round_buy_types (already in the sample folder) ----------
    rbt = pl.read_parquet(BASE + r"\round_buy_types.parquet").select(
        "match_key", pl.col("round").cast(pl.Int64), "ct_equipment", "t_equipment",
        "ct_buy", "t_buy", pl.col("ct_win").cast(pl.Int64).alias("rbt_ct_win"),
    )
    xc = df.join(rbt, on=["match_key", "round"], how="left")
    xc_c = xc.filter(pl.col("complete"))
    n_c = xc_c.height
    eq_exact = xc_c.filter(
        (pl.col("ct_eq") == pl.col("ct_equipment")) & (pl.col("t_eq") == pl.col("t_equipment"))).height
    rbt_missing = xc.filter(pl.col("ct_equipment").is_null()).height
    win_agree = xc.filter(pl.col("ct_win") == pl.col("rbt_ct_win")).height
    cls_mine = (pl.col("ct_eq") >= MAIN_T) & (pl.col("t_eq") >= MAIN_T)
    cls_rbt = (pl.col("ct_buy") == "full") & (pl.col("t_buy") == "full")
    cls_agree = xc_c.filter(cls_mine == cls_rbt).height
    cross = {
        "complete_rounds_checked": n_c,
        "team_equipment_exact_match_with_round_buy_types": eq_exact,
        "rounds_missing_in_round_buy_types": rbt_missing,
        "ct_win_agree_with_rounds_all_rounds": win_agree,
        "full_vs_full_T20000_agree_with_round_buy_types_full_labels": cls_agree,
        "full_vs_full_T20000_count_mine": xc_c.filter(cls_mine).height,
        "full_vs_full_count_round_buy_types_labels": xc_c.filter(cls_rbt).height,
    }
    print("\n== cross-check vs round_buy_types ==")
    print(json.dumps(cross, indent=1))

    # ---------- 6. write JSON ----------
    main_out = results_by_t[str(MAIN_T)]
    payload = {
        "analysis_id": "N2_fullfull",
        "script": SCRIPT_PATH,
        "inputs": {
            "players": BASE + r"\players_1hz.parquet (seconds_since_live == 0)",
            "rounds": BASE + r"\rounds.parquet",
            "crosscheck": BASE + r"\round_buy_types.parquet",
        },
        "bootstrap": {"method": "match-cluster percentile bootstrap", "resamples": B, "seed": SEED,
                      "rng": "numpy.random.default_rng(0)", "n_clusters": M},
        "data_counts": {
            "rounds_total": n_rows,
            "rounds_complete_snapshot0_5v5": int(complete.sum()),
            "rounds_incomplete_snapshot0": n_incomplete,
            "rounds_with_alive_gap": gap_rows.height,
            "duplicate_round_keys": n_round_dupes,
            "matches": M,
        },
        "crosscheck_round_buy_types": cross,
        "results_by_threshold": results_by_t,
        "table_main_T20000": [r for r in table
                              if "T20000" in r["group"] or r["group"] == "all rounds (comparator)"],
        "table_all": table,
        "numbers_main_T20000": [n for n in numbers if "T20000" in n["name"]],
        "numbers_all": numbers,
        "assumptions": [
            "ASSUMPTION: team equipment = sum of equipment_value_calc over the team's alive players at seconds_since_live == 0 (the first live second, tick == live_start_tick).",
            "ASSUMPTION: CT win = rounds.ct_win == 1.",
            "ASSUMPTION: 'full buy' is a team sum >= threshold; the main threshold is 20000 and 17000 / 23000 are sensitivity checks.",
            "ASSUMPTION: a round is team-complete only if each side has 5 alive players with a valid equipment value at snapshot 0; the single round with a gap (match 22e9c3e7..., round 16, T has 4 at snapshot 0) is excluded from team-level analyses and kept in the all-rounds comparator.",
            "ASSUMPTION: clusters are match_key; both train and test splits are pooled (150 matches, 2673 rounds).",
            "ASSUMPTION: 95% interval = 2.5/97.5 percentiles of 500 match-cluster resamples with replacement (numpy default_rng(0)); the same resamples are reused for every statistic.",
            "ASSUMPTION: Spearman-like correlation = Pearson correlation of average ranks (ties averaged), computed in numpy; its interval uses the same 500 resamples.",
            "ASSUMPTION: quintile edges are np.quantile of (CT eq - T eq) within full-vs-full rounds at each threshold; bins assigned with searchsorted (right).",
            "ASSUMPTION: maps with fewer than 40 qualifying rounds report n only, for both the overall-by-map and the difference rows.",
            "ASSUMPTION: 'differ by more than 3000' means |CT team eq - T team eq| > 3000 (strict).",
        ],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(jsonable(payload), fh, indent=2)
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()
