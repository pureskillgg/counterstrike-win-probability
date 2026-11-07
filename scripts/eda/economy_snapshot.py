"""Economy snapshot prototype (TASK D). CS2 round win probability, pre-round economy.

For 40 clean 5v5 matches (2026 days 01-07, not wingman, >= 24 rounds, every round
player_count = 10; sample = random.Random(0).sample of the sorted eligible keys), take the
player_status rows at the round_freeze_end tick of every round (nearest player_status tick if
no exact row), aggregate per side, join the round winner, and evaluate univariate AUCs and
small logistic models with a match-level split.

Secondary comparison (same pass, same E read): the same features at the buytime_ended tick
(end of buy time), restricted to rounds that are kept in both snapshots.

Reads:
  G (tomes):  header/dataframe_00000 (selection), basic_round_end (winner, count),
              basic_round_state (round_freeze_end and buytime_ended rows, scores),
              basic_player_info (team per round and player; needed because
              player_status has no team column).
  E (raw):    <E:\\csds_final\\csds\\2026\\10\\<DD>\\<id>\\player_status> for the 40 sampled
              matches only, column projection, one read per match.
Writes (under G, logs\\eda):
  economy_snapshot_matches.csv, economy_snapshot_e_timing.csv,
  economy_snapshot_rounds.csv, economy_snapshot_rounds_buyend.csv,
  economy_snapshot_results.json

Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I scripts\\eda\\economy_snapshot.py
"""

import glob
import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
E_CSDS = r"E:\csds_final\csds"
LOG_DIR = os.path.join(ROOT, "logs", "eda")

SEED = 0
N_MATCHES = 40
MIN_ROUNDS = 24
DAYS = {"01", "02", "03", "04", "05", "06", "07"}
MATCH_KEY_RE = re.compile(r"^csds/2026/10/(\d{2})/([0-9a-f-]{36})/csds$")
N_HOLDOUT_MATCHES = 10
N_BOOT = 1000

# player_status.inv_primary codes treated as "rifle or AWP". Names verified against
# basic_weapon_fire.weapon_name for the 40 sampled matches (economy_probe_weapon_codes.py):
# 7 ak47, 8 aug, 9 awp, 10 famas, 11 g3sg1, 13 galilar, 16 m4a1, 38 scar20, 39 sg556, 60 m4a1_silencer.
# SSG 08 (40) is a scout rifle and is excluded from this definition.
RIFLE_OR_AWP = {7, 8, 9, 10, 11, 13, 16, 38, 39, 60}

E_COLS = [
    "tick", "round", "player_id", "money", "armor", "has_helmet", "has_defuser",
    "inv_primary", "equipment_value_calc", "round_start_equipment_cost",
    "freezetime_end_equipment_cost", "current_equipment_cost", "health",
]
METRICS = ["eq", "money", "armor_n", "helmet_n", "defuser_n", "rifle_n", "rse", "ffe"]


def tome_files(channel):
    dirs = [d for d in glob.glob(os.path.join(TOME, "basic_%s.*" % channel)) if os.path.isdir(d)]
    assert len(dirs) == 1, (channel, dirs)
    return sorted(glob.glob(os.path.join(dirs[0], "dataframe_*")))


def e_path(match_key):
    # csds/2026/10/01/<id>/csds  ->  E:\csds_final\csds\2026\10\01\<id>\player_status
    parts = match_key.split("/")
    return os.path.join(E_CSDS, parts[1], parts[2], parts[3], parts[4], "player_status")


def load_selection():
    hdr = pq.read_table(
        os.path.join(TOME, "header", "dataframe_00000"),
        columns=["key", "is_wingman", "max_rounds", "match_date", "match_id"],
    ).to_pandas()
    rend = pq.read_table(
        tome_files("round_end")[0],
        columns=["match_key", "round", "winner_team_code", "player_count"],
    ).to_pandas()
    per = rend.groupby("match_key").agg(
        n_rounds=("round", "size"),
        n_unique_rounds=("round", "nunique"),
        all_pc10=("player_count", lambda s: bool((s == 10).all())),
    )
    rows = []
    for _, r in hdr.iterrows():
        m = MATCH_KEY_RE.match(r["key"])
        rows.append({
            "match_key": r["key"],
            "day": m.group(1) if m else None,
            "is_wingman": bool(r["is_wingman"]),
            "max_rounds": int(r["max_rounds"]),
        })
    cand = pd.DataFrame(rows).merge(per, left_on="match_key", right_index=True, how="left")
    cand["n_rounds"] = cand["n_rounds"].fillna(0).astype(int)
    cand["n_unique_rounds"] = cand["n_unique_rounds"].fillna(0).astype(int)
    cand["all_pc10"] = cand["all_pc10"].fillna(False).astype(bool)
    cand["in_days_01_07"] = cand["day"].isin(DAYS)
    cand["eligible"] = (
        cand["in_days_01_07"] & (~cand["is_wingman"]) & (cand["n_rounds"] >= MIN_ROUNDS)
        & cand["all_pc10"] & (cand["n_unique_rounds"] == cand["n_rounds"])
    )
    eligible = sorted(cand.loc[cand["eligible"], "match_key"].tolist())
    chosen = sorted(random.Random(SEED).sample(eligible, N_MATCHES))
    return cand, eligible, chosen


def read_player_status(match_key):
    path = e_path(match_key)
    t0 = time.perf_counter()
    pf = pq.ParquetFile(path)
    t_open = time.perf_counter() - t0
    t1 = time.perf_counter()
    tbl = pf.read(columns=E_COLS)
    t_read = time.perf_counter() - t1
    t2 = time.perf_counter()
    df = tbl.to_pandas()
    t_pd = time.perf_counter() - t2
    md = pf.metadata
    proj_bytes = 0
    for rg in range(md.num_row_groups):
        row = md.row_group(rg)
        for c in range(row.num_columns):
            col = row.column(c)
            if col.path_in_schema in E_COLS:
                proj_bytes += col.total_compressed_size
    timing = {
        "match_key": match_key,
        "file_bytes": os.path.getsize(path),
        "projected_compressed_bytes": int(proj_bytes),
        "rows": int(md.num_rows),
        "open_s": round(t_open, 4),
        "read_s": round(t_read, 4),
        "to_pandas_s": round(t_pd, 4),
        "total_s": round(t_open + t_read + t_pd, 4),
    }
    return df, timing


def snapshot_at(sub, F, mt_round):
    """Nearest-tick snapshot of one round. Returns (snap, tick, gap, dup_rows)."""
    ticks = np.unique(sub["tick"].to_numpy())
    i = int(np.searchsorted(ticks, F))
    cands = [int(ticks[j]) for j in (i - 1, i) if 0 <= j < len(ticks)]
    t_star = min(cands, key=lambda x: (abs(x - F), x))
    snap = sub[sub["tick"] == t_star].copy()
    n_dup = int(snap.duplicated(["player_id"]).sum())
    snap = snap.drop_duplicates(["player_id"], keep="first")
    snap = snap.merge(mt_round, on="player_id", how="left")
    return snap, t_star, abs(t_star - F), n_dup


def split_ok(snap):
    n_null = int(snap["team_code"].isna().sum())
    n_t = int((snap["team_code"] == 2).sum())
    n_ct = int((snap["team_code"] == 3).sum())
    if n_null > 0:
        return False, "null_team", n_t, n_ct
    if n_t != 5 or n_ct != 5:
        return False, "team_split_not_5v5", n_t, n_ct
    return True, "", n_t, n_ct


def side_table(snaps):
    """Per (match, round) side aggregates from kept snapshot rows."""
    allsnap = pd.concat(snaps, ignore_index=True)
    allsnap["armor_flag"] = allsnap["armor"] > 0
    allsnap["rifle_flag"] = allsnap["inv_primary"].isin(RIFLE_OR_AWP)
    agg = allsnap.groupby(["match_key", "round", "team_code"]).agg(
        eq=("equipment_value_calc", "sum"),
        money=("money", "sum"),
        armor_n=("armor_flag", "sum"),
        helmet_n=("has_helmet", "sum"),
        defuser_n=("has_defuser", "sum"),
        rifle_n=("rifle_flag", "sum"),
        rse=("round_start_equipment_cost", "sum"),
        ffe=("freezetime_end_equipment_cost", "sum"),
        n=("player_id", "size"),
    ).reset_index()
    wide = agg.pivot(index=["match_key", "round"], columns="team_code")
    out = pd.DataFrame(index=wide.index)
    for m in METRICS:
        out["t_" + m] = wide[(m, 2)]
        out["ct_" + m] = wide[(m, 3)]
    out = out.reset_index()
    for m in METRICS:
        out["d_" + m] = out["ct_" + m] - out["t_" + m]
    return out


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece10": ece(y, p),
    }


def fit_predict(Xtr, ytr, Xte, const=False):
    if const:
        return np.full(len(Xte), ytr.mean())
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def cluster_boot(fns, keys, idx_by_key, rng, B=N_BOOT):
    """Match-cluster bootstrap. fns: name -> f(row_index_array) -> float or None."""
    draws = {k: [] for k in fns}
    for _ in range(B):
        ks = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([idx_by_key[k] for k in ks])
        for name, f in fns.items():
            v = f(idx)
            if v is not None and np.isfinite(v):
                draws[name].append(v)
    out = {}
    for name, vals in draws.items():
        vals = np.asarray(vals)
        out[name] = {"ci95_lo": float(np.percentile(vals, 2.5)),
                     "ci95_hi": float(np.percentile(vals, 97.5)),
                     "n_valid": int(len(vals))}
    return out


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    t_start = time.perf_counter()

    # 1. Selection (header + round_end, both G-side)
    cand, eligible, chosen = load_selection()
    print("eligible matches (days 01-07, not wingman, >=24 rounds, all rounds 10 players):", len(eligible))
    print("chosen matches:", len(chosen), "seed", SEED)
    cand[cand["match_key"].isin(chosen)].to_csv(
        os.path.join(LOG_DIR, "economy_snapshot_matches.csv"), index=False)
    chosen_set = set(chosen)

    # 2. Round winners (G-side round_end)
    rend = pq.read_table(tome_files("round_end")[0],
                         columns=["match_key", "round", "winner_team_code", "player_count"]).to_pandas()
    rend = rend[rend["match_key"].isin(chosen_set)].copy()
    rend["ct_win"] = (rend["winner_team_code"] == 3).astype(int)

    # 3. Freeze-end and buy-end rows and scores (G-side round_state)
    rs_parts, rb_parts = [], []
    for f in tome_files("round_state"):
        rs_parts.append(pq.read_table(
            f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
            filters=[("event_type", "==", "round_freeze_end")]).to_pandas())
        rb_parts.append(pq.read_table(
            f, columns=["match_key", "round", "tick", "event_type"],
            filters=[("event_type", "==", "buytime_ended")]).to_pandas())
    rs = pd.concat(rs_parts, ignore_index=True)
    rs = rs[rs["match_key"].isin(chosen_set)].copy()
    rs_dups = int(rs.duplicated(["match_key", "round"]).sum())
    rs = rs.drop_duplicates(["match_key", "round"], keep="first")
    rb = pd.concat(rb_parts, ignore_index=True)
    rb = rb[rb["match_key"].isin(chosen_set)].drop_duplicates(["match_key", "round"], keep="first")
    buy_map = {(m, int(r)): int(t) for m, r, t in zip(rb["match_key"], rb["round"], rb["tick"])}

    # 4. Team membership (G-side basic_player_info; player_status has no team column)
    pi = pq.read_table(
        tome_files("player_info")[0],
        columns=["match_key", "round", "player_id", "team_code"],
        filters=[("match_key", "in", chosen)]).to_pandas()
    pi_dups = int(pi.duplicated(["match_key", "round", "player_id"]).sum())
    pi = pi.drop_duplicates(["match_key", "round", "player_id"], keep="first")
    pi = pi[pi["team_code"].isin([2, 3])]

    # 5. Per match: read player_status once from E, snapshot at freeze end and buy end
    snaps, snaps_buy, round_diag, timings = [], [], [], []
    for mk in chosen:
        try:
            df, timing = read_player_status(mk)
        except Exception as exc:  # record and continue
            timings.append({"match_key": mk, "error": repr(exc)})
            continue
        timings.append(timing)
        fz = rs[rs["match_key"] == mk].set_index("round")
        mt = pi[pi["match_key"] == mk][["round", "player_id", "team_code"]]
        grp = {r: g for r, g in df.groupby("round")}
        for r, row in fz.iterrows():
            r = int(r)
            F = int(row["tick"])
            diag = {"match_key": mk, "round": r, "freeze_tick": F,
                    "t_score": int(row["t_score"]), "ct_score": int(row["ct_score"])}
            if r not in grp:
                diag.update(status="dropped", reason="no_player_status_rows_in_round")
                round_diag.append(diag)
                continue
            sub = grp[r]
            mt_r = mt[mt["round"] == r][["player_id", "team_code"]]
            snap, t_star, gap, n_dup = snapshot_at(sub, F, mt_r)
            ok, reason, n_t, n_ct = split_ok(snap)
            diag.update(snapshot_tick=t_star, tick_gap=int(gap), players=len(snap),
                        n_T=n_t, n_CT=n_ct, dup_player_rows=n_dup)
            if ok:
                diag.update(status="kept", reason="")
                snap = snap.assign(match_key=mk, round=r)
                snaps.append(snap)
            else:
                diag.update(status="dropped", reason=reason)
            # buy-end snapshot (same round rows, tick = buytime_ended)
            B = buy_map.get((mk, r))
            if B is not None:
                snapB, tB, gapB, _ = snapshot_at(sub, B, mt_r)
                okB, reasonB, nBt, nBct = split_ok(snapB)
                diag.update(buy_tick=B, buy_minus_freeze_ticks=int(B - F),
                            buy_snapshot_tick=tB, buy_tick_gap=int(gapB),
                            buy_status="kept" if okB else "dropped", buy_reason=reasonB,
                            buy_players=len(snapB), buy_n_T=nBt, buy_n_CT=nBct)
                if okB:
                    snapsB = snapB.assign(match_key=mk, round=r)
                    snaps_buy.append(snapsB)
            round_diag.append(diag)
        print(mk[-40:], "read_s", timing["read_s"], "total_s", timing["total_s"], "rows", timing["rows"])

    diag_df = pd.DataFrame(round_diag)
    diag_df.to_csv(os.path.join(LOG_DIR, "economy_snapshot_round_diag.csv"), index=False)
    timing_df = pd.DataFrame(timings)
    timing_df.to_csv(os.path.join(LOG_DIR, "economy_snapshot_e_timing.csv"), index=False)

    # 6. Side aggregates and labels
    out = side_table(snaps)
    out = out.merge(rend[["match_key", "round", "winner_team_code", "ct_win"]],
                    on=["match_key", "round"], how="left")
    out = out.merge(rs[["match_key", "round", "t_score", "ct_score"]],
                    on=["match_key", "round"], how="left")
    out["d_score"] = out["ct_score"] - out["t_score"]
    out = out.sort_values(["match_key", "round"]).reset_index(drop=True)
    out.to_csv(os.path.join(LOG_DIR, "economy_snapshot_rounds.csv"), index=False)

    out_buy = side_table(snaps_buy)
    out_buy = out_buy.merge(rend[["match_key", "round", "ct_win"]], on=["match_key", "round"], how="left")
    out_buy = out_buy.merge(rs[["match_key", "round", "t_score", "ct_score"]], on=["match_key", "round"], how="left")
    out_buy["d_score"] = out_buy["ct_score"] - out_buy["t_score"]
    out_buy = out_buy.sort_values(["match_key", "round"]).reset_index(drop=True)
    out_buy.to_csv(os.path.join(LOG_DIR, "economy_snapshot_rounds_buyend.csv"), index=False)

    # 7. Analysis (freeze-end snapshot is the primary)
    y = out["ct_win"].to_numpy()
    groups = out["match_key"].to_numpy()
    keys = sorted(set(groups.tolist()))
    idx_by_key = {k: np.where(groups == k)[0] for k in keys}
    rng = np.random.default_rng(SEED)
    kept = diag_df[diag_df["status"] == "kept"]

    results = {
        "selection": {
            "eligible_matches_days_01_07": len(eligible),
            "chosen_matches": len(chosen),
            "seed": SEED,
            "min_rounds": MIN_ROUNDS,
            "max_rounds_in_sample": sorted(cand.loc[cand["match_key"].isin(chosen), "max_rounds"].unique().tolist()),
        },
        "rounds": {
            "round_freeze_rows": int(len(rs)),
            "round_freeze_dup_rows": rs_dups,
            "player_info_dup_rows": pi_dups,
            "round_attempts": int(len(diag_df)),
            "status_counts": diag_df["status"].value_counts().to_dict(),
            "reason_counts": diag_df["reason"].value_counts().to_dict(),
            "kept_rounds": int(len(out)),
            "kept_matches": int(out["match_key"].nunique()),
            "ct_win_rate": float(y.mean()),
            "rounds_per_kept_match_mean": float(out.groupby("match_key").size().mean()),
        },
        "tick_gap_freeze_kept": {
            "exact_share": float((kept["tick_gap"] == 0).mean()),
            "median": float(kept["tick_gap"].median()),
            "max": int(kept["tick_gap"].max()),
        },
    }

    uni = {
        "d_eq (equipment_value_calc)": out["d_eq"],
        "d_money": out["d_money"],
        "d_armor_n": out["d_armor_n"],
        "d_helmet_n (extra)": out["d_helmet_n"],
        "d_defuser_n (extra)": out["d_defuser_n"],
        "d_rifle_n (extra)": out["d_rifle_n"],
        "d_score (round-start score diff, control)": out["d_score"],
        "d_round_start_equipment_cost (robustness)": out["d_rse"],
        "d_freezetime_end_equipment_cost (robustness)": out["d_ffe"],
    }
    univ = {}
    for name, x in uni.items():
        xv = x.to_numpy(dtype=float)
        fns = {name: (lambda idx, xv=xv: safe_auc(y[idx], xv[idx]))}
        univ[name] = {"auc": safe_auc(y, xv), **cluster_boot(fns, keys, idx_by_key, rng)[name]}
    results["univariate_auc_ct_win_freeze"] = univ

    feature_sets = {
        "const_alive_diff_0": None,
        "score_only": ["d_score"],
        "econ3": ["d_eq", "d_money", "d_armor_n"],
        "econ3_plus_score": ["d_eq", "d_money", "d_armor_n", "d_score"],
        "econ_full6": ["d_eq", "d_money", "d_armor_n", "d_helmet_n", "d_defuser_n", "d_rifle_n"],
    }

    # Holdout: 10 matches chosen with random.Random(SEED)
    test_keys = set(random.Random(SEED).sample(keys, N_HOLDOUT_MATCHES))
    test_mask = np.array([g in test_keys for g in groups])
    holdout = {}
    gkf = GroupKFold(n_splits=5)
    oof, oof_fold_auc, oof_pred = {}, {}, {}
    for name, cols in feature_sets.items():
        X = np.zeros((len(out), 1)) if cols is None else out[cols].to_numpy(dtype=float)
        const = cols is None
        p_hold = fit_predict(X[~test_mask], y[~test_mask], X[test_mask], const=const)
        holdout[name] = {"auc": safe_auc(y[test_mask], p_hold), **metrics(y[test_mask], p_hold)}
        p_oof = np.zeros(len(out))
        fold_aucs = []
        for tr, te in gkf.split(X, y, groups):
            p_oof[te] = fit_predict(X[tr], y[tr], X[te], const=const)
            fold_aucs.append(safe_auc(y[te], p_oof[te]))
        oof_pred[name] = p_oof
        oof[name] = metrics(y, p_oof)
        valid = [a for a in fold_aucs if a is not None]
        oof_fold_auc[name] = {"fold_auc_mean": float(np.mean(valid)), "fold_auc_sd": float(np.std(valid)),
                              "fold_auc": [round(a, 4) if a is not None else None for a in fold_aucs]}
    results["holdout_10_matches"] = {"test_matches": sorted(test_keys), "test_rounds": int(test_mask.sum()),
                                     "models": holdout}
    results["oof_5fold_group"] = {"models": {k: {**oof[k], **oof_fold_auc[k]} for k in oof}}

    def auc_diff(a, b):
        def f(idx):
            va = safe_auc(y[idx], oof_pred[a][idx])
            vb = safe_auc(y[idx], oof_pred[b][idx])
            return None if (va is None or vb is None) else va - vb
        return f

    def ll_gain(base, new):
        def f(idx):
            return float(log_loss(y[idx], np.clip(oof_pred[base][idx], 1e-6, 1 - 1e-6))
                         - log_loss(y[idx], np.clip(oof_pred[new][idx], 1e-6, 1 - 1e-6)))
        return f

    fns = {}
    for name in ["const_alive_diff_0", "score_only", "econ3", "econ3_plus_score"]:
        fns[name + " | pooled OOF AUC"] = (lambda p: (lambda idx: safe_auc(y[idx], p[idx])))(oof_pred[name])
        fns[name + " | pooled OOF logloss"] = (
            lambda p: (lambda idx: float(log_loss(y[idx], np.clip(p[idx], 1e-6, 1 - 1e-6)))))(oof_pred[name])
    fns["econ3 minus const | AUC"] = auc_diff("econ3", "const_alive_diff_0")
    fns["econ3 minus const | logloss gain"] = ll_gain("const_alive_diff_0", "econ3")
    fns["econ3 minus score_only | AUC"] = auc_diff("econ3", "score_only")
    fns["econ3 minus score_only | logloss gain"] = ll_gain("score_only", "econ3")
    fns["econ3_plus_score minus score_only | AUC"] = auc_diff("econ3_plus_score", "score_only")
    fns["econ3_plus_score minus score_only | logloss gain"] = ll_gain("score_only", "econ3_plus_score")
    results["oof_cluster_bootstrap_95ci"] = cluster_boot(fns, keys, idx_by_key, rng)

    p3 = oof_pred["econ3"]
    bins = np.linspace(0, 1, 6)
    idx_b = np.clip(np.digitize(p3, bins[1:-1]), 0, 4)
    results["econ3_oof_calibration_5bins"] = [
        {"bin": f"{bins[b]:.1f}-{bins[b + 1]:.1f}", "n": int((idx_b == b).sum()),
         "mean_pred": float(p3[idx_b == b].mean()) if (idx_b == b).any() else None,
         "obs_ct_win": float(y[idx_b == b].mean()) if (idx_b == b).any() else None}
        for b in range(5)]

    coefs = {}
    for name in ["score_only", "econ3", "econ3_plus_score"]:
        cols = feature_sets[name]
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        pipe.fit(out[cols].to_numpy(dtype=float), y)
        coefs[name] = {c: float(v) for c, v in zip(cols, pipe.named_steps["logisticregression"].coef_[0])}
    results["standardized_coef_all_rounds_descriptive"] = coefs

    # 8. Buy-end comparison on rounds kept in both snapshots
    paired = out[["match_key", "round"]].merge(out_buy[["match_key", "round"]], on=["match_key", "round"])
    fz_p = out.merge(paired, on=["match_key", "round"])
    by_p = out_buy.merge(paired, on=["match_key", "round"])
    fz_p = fz_p.sort_values(["match_key", "round"]).reset_index(drop=True)
    by_p = by_p.sort_values(["match_key", "round"]).reset_index(drop=True)
    yp = fz_p["ct_win"].to_numpy()
    gp = fz_p["match_key"].to_numpy()
    kp = sorted(set(gp.tolist()))
    ibk = {k: np.where(gp == k)[0] for k in kp}
    buy_cmp = {"paired_rounds": int(len(paired)), "paired_matches": int(paired["match_key"].nunique()),
               "buy_minus_freeze_ticks_median": float(diag_df["buy_minus_freeze_ticks"].median())
               if "buy_minus_freeze_ticks" in diag_df else None,
               "buy_minus_freeze_seconds_p10_p90": [
                   float(diag_df["buy_minus_freeze_ticks"].quantile(0.1) / 64),
                   float(diag_df["buy_minus_freeze_ticks"].quantile(0.9) / 64)]
               if "buy_minus_freeze_ticks" in diag_df else None,
               "buy_tick_gap_median": float(diag_df["buy_tick_gap"].median()) if "buy_tick_gap" in diag_df else None,
               "buy_status_counts": diag_df["buy_status"].value_counts().to_dict() if "buy_status" in diag_df else None,
               "uni_auc_freeze_vs_buyend_paired": {}, "econ3_oof_paired": {}}
    for name in ["d_eq", "d_money", "d_armor_n", "d_score"]:
        buy_cmp["uni_auc_freeze_vs_buyend_paired"][name] = {
            "freeze": safe_auc(yp, fz_p[name].to_numpy(dtype=float)),
            "buy_end": safe_auc(yp, by_p[name].to_numpy(dtype=float))}
    # econ3 OOF on paired rounds, freeze vs buy-end features, same folds
    for label, frame in [("freeze", fz_p), ("buy_end", by_p)]:
        Xp = frame[feature_sets["econ3"]].to_numpy(dtype=float)
        p_oof = np.zeros(len(frame))
        fa = []
        for tr, te in GroupKFold(n_splits=5).split(Xp, yp, gp):
            p_oof[te] = fit_predict(Xp[tr], yp[tr], Xp[te])
            fa.append(safe_auc(yp[te], p_oof[te]))
        buy_cmp["econ3_oof_paired"][label] = {
            "fold_auc_mean": float(np.mean([a for a in fa if a is not None])),
            **metrics(yp, p_oof)}
    results["freeze_vs_buy_end"] = buy_cmp

    # 9. E read timing and extrapolation
    if "total_s" in timing_df.columns:
        ok = timing_df.dropna(subset=["total_s"]).reset_index(drop=True)
        rest = ok.iloc[1:]
        results["e_read_timing"] = {
            "matches_read": int(len(ok)),
            "file_bytes_total": int(ok["file_bytes"].sum()),
            "file_bytes_mean": float(ok["file_bytes"].mean()),
            "projected_bytes_mean": float(ok["projected_compressed_bytes"].mean()),
            "rows_mean": float(ok["rows"].mean()),
            "total_s_first_match": float(ok["total_s"].iloc[0]),
            "total_s_mean_all": float(ok["total_s"].mean()),
            "total_s_mean_excluding_first": float(rest["total_s"].mean()),
            "total_s_median": float(ok["total_s"].median()),
            "total_s_p90": float(ok["total_s"].quantile(0.9)),
            "extrapolated_1752_hours_median_per_match": float(ok["total_s"].median() * 1752 / 3600),
            "extrapolated_1752_hours_mean_excl_first": float(rest["total_s"].mean() * 1752 / 3600),
            "extrapolated_1752_file_gb": float(ok["file_bytes"].mean() * 1752 / 1e9),
        }
    results["wall_clock_s"] = round(time.perf_counter() - t_start, 1)

    with open(os.path.join(LOG_DIR, "economy_snapshot_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
