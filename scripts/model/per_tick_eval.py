"""Per-second (1 Hz) win-probability evaluation, following the advisor's rules.

Rules applied (from the advisor's note):
  * No random train/test split: the bake-off primary split is by MATCH (seed 0), so no round or match is on both sides.
  * The model must run at every tick, so it is trained and scored on a 1 Hz grid of ticks (freeze_end + 64k, k >= 0,
    strictly before round_end), not only on event-anchored states. Between events the alive counts and bomb state are
    constant; the two time features advance every second.
  * Target metric: log loss. Also Brier, AUC, ECE (10 equal-width bins) and equal-mass reliability tables.
  * Ticks inside one round share one winner, so they are NOT independent. Every interval here is a match-cluster
    bootstrap (resample matches, 1000 resamples, seed 0). The naive i.i.d. interval is printed beside it to show the gap.
  * Round-weighted log loss (each round counts once) is reported beside the all-ticks number.

Reuses (read only): scripts/eda/extract_state_1hz.py (bakeoff_kept) and scripts/model/bakeoff.py (build_states, FEATS, MONO,
make_splits, metrics). Writes under data/viz and notes/model only.

Run from the project root:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\model\\per_tick_eval.py
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")  # BLAS crashes natively otherwise (see sitecustomize.py)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import collections
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
VIZ = ROOT / "data" / "viz"
NOTE = ROOT / "notes" / "model" / "per_tick_eval.md"
OUT_JSON = VIZ / "per_tick_eval.json"
SEED = 0
N_BOOT = 1000
TICK = 64
EPS = 1e-15

spec = importlib.util.spec_from_file_location("extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
bk = ex.bk
FEATS, MONO = bk.FEATS, bk.MONO


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def xy(df):
    return df.select(FEATS).to_numpy().astype(np.float64), df["label_ct_win"].to_numpy().astype(np.int64)


def fit_xgb(tr, va):
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
                          tree_method="hist", device="cpu", monotone_constraints=tuple(MONO),
                          early_stopping_rounds=100, eval_metric="logloss", random_state=SEED)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, int(m.best_iteration)


def pred_xgb(m, best, df):
    X, _ = xy(df)
    return m.predict_proba(X, iteration_range=(0, best + 1))[:, 1]


class RidgeLogit:
    """Standardised L2 logistic regression (C = 1, intercept unpenalised), Newton steps, numpy only.
    Same objective as sklearn's LogisticRegression(C=1.0); written out because the sklearn/scipy BLAS call
    crashes natively (Windows error 0xc06d007f) in this environment."""

    def fit(self, X, y, C=1.0, iters=30):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        lam = np.full(Z.shape[1], 1.0 / C)
        lam[0] = 0.0
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ w)))
            g = Z.T @ (p - y) + lam * w
            H = (Z * (p * (1.0 - p))[:, None]).T @ Z + np.diag(lam)
            step = np.linalg.solve(H, g)
            w = w - step
            if np.abs(step).max() < 1e-9:
                break
        self.w = w
        return self

    def predict(self, X):
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return 1.0 / (1.0 + np.exp(-(Z @ self.w)))


def fit_lr(tr, cols):
    X = tr.select(cols).to_numpy().astype(np.float64)
    y = tr["label_ct_win"].to_numpy().astype(np.int64)
    return RidgeLogit().fit(X, y)


def pred_lr(model, df, cols):
    return model.predict(df.select(cols).to_numpy().astype(np.float64))


def expand_round(k, rr, states, plants_k):
    """1 Hz grid for one round as numpy columns. Returns (cols or None, death_events, plant_sec)."""
    frz, end = rr["freeze"], rr["end"]
    in_win = [t for t in plants_k if frz <= t <= end]
    plant = min(in_win) if in_win else None
    st = sorted(states, key=lambda s: s["tick"])
    first = st[0]
    ts = np.arange(frz, end, TICK, dtype=np.int64)
    cols = None
    if len(ts) > 0:
        st_ticks = np.array([s["tick"] for s in st], dtype=np.int64)
        j = np.clip(np.searchsorted(st_ticks, ts, side="right") - 1, 0, len(st) - 1)
        ac = np.array([s["alive_ct"] for s in st], dtype=np.int64)[j]
        at = np.array([s["alive_t"] for s in st], dtype=np.int64)[j]
        n = len(ts)
        planted = (ts >= plant) if plant is not None else np.zeros(n, dtype=bool)
        cols = {
            "round": np.full(n, k[1], dtype=np.int64), "tick": ts,
            "alive_ct": ac, "alive_t": at, "ct_alive_minus_t": ac - at,
            "bomb_planted": planted.astype(np.int64),
            "seconds_since_round_live": (ts - frz) / TICK,
            "seconds_since_plant": np.where(planted, (ts - (plant if plant is not None else 0)) / TICK, -1.0),
            "round_number": np.full(n, first["round_number"], dtype=np.int64),
            "ct_score_minus_t_score": np.full(n, first["ct_score_minus_t_score"], dtype=np.int64),
            "is_pistol_round": np.full(n, first["is_pistol_round"], dtype=np.int64),
            "is_overtime": np.full(n, first["is_overtime"], dtype=np.int64),
            "label_ct_win": np.full(n, first["label_ct_win"], dtype=np.int64),
        }
    deaths, prev = [], (first["alive_ct"], first["alive_t"])
    for s in st[1:]:
        if s["alive_ct"] < prev[0]:
            deaths.append({"sec": (s["tick"] - frz) / TICK, "side": "CT"})
        elif s["alive_t"] < prev[1]:
            deaths.append({"sec": (s["tick"] - frz) / TICK, "side": "T"})
        prev = (s["alive_ct"], s["alive_t"])
    psec = ((plant - frz) / TICK) if plant is not None else None
    return cols, deaths, psec


def match_index(df):
    mk = df["match_key"].to_list()
    uniq = sorted(set(mk))
    pos = {m: i for i, m in enumerate(uniq)}
    return np.array([pos[m] for m in mk], dtype=np.int64), len(uniq)


def cluster_ratio(num, den, mid, M, idx):
    sn = np.bincount(mid, weights=num, minlength=M)
    sd = np.bincount(mid, weights=den, minlength=M)
    return sn[idx].sum(axis=1) / np.maximum(sd[idx].sum(axis=1), 1e-12), float(sn.sum() / sd.sum())


def pct(a):
    a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def reliability(p, y, mid, M, idx, edges):
    """Reliability bins with cluster-bootstrap interval on the observed rate."""
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    out = []
    for i in range(len(edges) - 1):
        m = b == i
        n = int(m.sum())
        if n == 0:
            out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": 0})
            continue
        reps, rate = cluster_ratio(y * m, m.astype(np.float64), mid, M, idx)
        out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n, "mean_pred": float(p[m].mean()),
                    "observed": float(y[m].mean()), "obs_ci95": pct(reps)})
    return out


def main():
    t0 = time.time()
    VIZ.mkdir(parents=True, exist_ok=True)
    kept = ex.bakeoff_kept()
    H, rd, live, plants, players, deaths = (kept[x] for x in ("H", "rd", "live", "plants", "players", "deaths"))
    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    print(f"split matches: train {len(sp['train'])}, val {len(sp['val'])}, test {len(sp['test'])}", flush=True)

    event_rows = []
    chunks, mkeys = [], []
    death_ev, plant_sec = {}, {}
    for k in kept["rounds_ok"]:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        st = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                             H[k[0]]["max_rounds"], collections.Counter())
        assert st is not None
        event_rows.extend(st)
        cols, dev, psec = expand_round(k, rr, st, plants.get(k, []))
        if cols is not None:
            chunks.append(cols)
            mkeys.extend([k[0]] * len(cols["tick"]))
        death_ev[k] = dev
        plant_sec[k] = psec
    names = list(chunks[0].keys())
    G = pl.DataFrame({"match_key": mkeys, **{n_: np.concatenate([c[n_] for c in chunks]) for n_ in names}})
    del chunks, mkeys
    E = pl.DataFrame(event_rows)
    assert E.height == kept["got"]["states"]
    print(f"event states {E.height}; 1 Hz grid rows {G.height}; rounds {len(kept['rounds_ok'])} "
          f"({time.time() - t0:.0f}s)", flush=True)
    n_pts = G.group_by(["match_key", "round"]).len().rename({"len": "n_pts"})
    G = G.join(n_pts, on=["match_key", "round"])
    G = G.with_columns((1.0 / pl.col("n_pts")).alias("w_round"))

    parts = {n: G.filter(pl.col("match_key").is_in(sorted(sp[n]))) for n in ("train", "val", "test")}
    parts_e = {n: E.filter(pl.col("match_key").is_in(sorted(sp[n]))) for n in ("train", "val", "test")}
    for n in parts:
        print(f"  {n}: grid rows {parts[n].height}, event states {parts_e[n].height}", flush=True)

    # ---- models on the 1 Hz grid
    tr, va, te = parts["train"], parts["val"], parts["test"]
    rate = float(tr["label_ct_win"].mean())
    lr1 = fit_lr(tr, ["ct_alive_minus_t"])
    lrall = fit_lr(tr, FEATS)
    print(f"LR fitted ({time.time() - t0:.0f}s)", flush=True)
    mg, best_g = fit_xgb(tr, va)
    print(f"XGBoost grid fitted: best_iteration {best_g} ({time.time() - t0:.0f}s)", flush=True)
    # ---- the bake-off model: trained on event states only
    me, best_e = fit_xgb(parts_e["train"], parts_e["val"])
    print(f"XGBoost event-trained fitted: best_iteration {best_e} ({time.time() - t0:.0f}s)", flush=True)

    y = te["label_ct_win"].to_numpy().astype(np.int64)
    P = {
        "constant": np.full(len(y), rate),
        "LR alive difference": pred_lr(lr1, te, ["ct_alive_minus_t"]),
        "LR all features": pred_lr(lrall, te, FEATS),
        "XGBoost (trained on 1 Hz grid)": pred_xgb(mg, best_g, te),
        "XGBoost (trained on event states)": pred_xgb(me, best_e, te),
    }
    wr = te["w_round"].to_numpy()
    sec = te["seconds_since_round_live"].to_numpy()
    mid, M = match_index(te)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))
    one = np.ones(len(y))

    models = {}
    for name, p in P.items():
        ll = ll_vec(p, y)
        reps_s, pt_s = cluster_ratio(ll, one, mid, M, idx)
        reps_r, pt_r = cluster_ratio(ll * wr, wr, mid, M, idx)
        mt = bk.metrics(y, p)
        naive_se = float(ll.std(ddof=1) / np.sqrt(len(ll)))
        clus_se = float(reps_s.std(ddof=1))
        models[name] = {
            "logloss_all_ticks": pt_s, "logloss_all_ticks_ci95": pct(reps_s),
            "logloss_round_weighted": pt_r, "logloss_round_weighted_ci95": pct(reps_r),
            "brier": mt["brier"], "auc": mt["auc"], "ece_10_equal_width": mt["ece"],
            "naive_iid_se": naive_se, "cluster_se": clus_se, "cluster_over_naive_se": clus_se / naive_se,
            "naive_iid_ci95": [pt_s - 1.96 * naive_se, pt_s + 1.96 * naive_se],
        }
        print(f"{name:<36} LL all {pt_s:.5f} {models[name]['logloss_all_ticks_ci95']} | round-weighted {pt_r:.5f} | "
              f"Brier {mt['brier']:.5f} AUC {mt['auc']:.4f} ECE {mt['ece']:.4f} | SE naive {naive_se:.5f} cluster {clus_se:.5f}",
              flush=True)

    # ---- paired differences
    def paired(a, b):
        d = ll_vec(P[a], y) - ll_vec(P[b], y)
        rs, pt = cluster_ratio(d, one, mid, M, idx)
        rr_, ptr = cluster_ratio(d * wr, wr, mid, M, idx)
        return {"a_minus_b_all_ticks": pt, "ci95": pct(rs), "round_weighted": ptr, "round_weighted_ci95": pct(rr_)}
    paired_out = {
        "XGB grid minus LR alive difference": paired("XGBoost (trained on 1 Hz grid)", "LR alive difference"),
        "XGB grid minus LR all features": paired("XGBoost (trained on 1 Hz grid)", "LR all features"),
        "LR all features minus LR alive difference": paired("LR all features", "LR alive difference"),
        "XGB event-trained minus XGB grid-trained": paired("XGBoost (trained on event states)", "XGBoost (trained on 1 Hz grid)"),
    }
    for k_, v in paired_out.items():
        print(f"paired {k_}: {v['a_minus_b_all_ticks']:+.5f} CI {v['ci95']}; round-weighted {v['round_weighted']:+.5f}", flush=True)

    # ---- by time bin
    tb = np.minimum((sec // 10).astype(int), 11)
    bins = []
    px = P["XGBoost (trained on 1 Hz grid)"]
    pl1 = P["LR alive difference"]
    for b in range(12):
        m = tb == b
        if m.sum() == 0:
            continue
        llx, ll1 = ll_vec(px, y), ll_vec(pl1, y)
        rx, ptx = cluster_ratio(llx * m, m.astype(np.float64), mid, M, idx)
        r1, pt1 = cluster_ratio(ll1 * m, m.astype(np.float64), mid, M, idx)
        bins.append({"bin": f"{b * 10}-{b * 10 + 10}" if b < 11 else "110+", "n": int(m.sum()),
                     "xgb_logloss": ptx, "xgb_ci95": pct(rx), "lr_alive_logloss": pt1, "lr_ci95": pct(r1),
                     "mean_pred": float(px[m].mean()), "observed": float(y[m].mean())})

    # ---- calibration
    eq_w = np.linspace(0, 1, 11)
    cal = {}
    for name in ("XGBoost (trained on 1 Hz grid)", "LR all features", "LR alive difference", "XGBoost (trained on event states)"):
        p = P[name]
        eq_m = np.quantile(p, np.linspace(0, 1, 11))
        eq_m[0], eq_m[-1] = 0.0, 1.0
        eq_m = np.unique(eq_m)
        cal[name] = {"equal_width": reliability(p, y, mid, M, idx, eq_w),
                     "equal_mass": reliability(p, y, mid, M, idx, eq_m)}

    # ---- terminal-state inflation in the event table
    te_e = parts_e["test"]
    ye = te_e["label_ct_win"].to_numpy().astype(np.int64)
    pe = pred_xgb(me, best_e, te_e)
    term = ((te_e["alive_ct"] == 0) | (te_e["alive_t"] == 0)).to_numpy()
    lle = ll_vec(pe, ye)
    terminal = {"event_test_states": int(len(ye)), "terminal_states": int(term.sum()),
                "terminal_share": float(term.mean()),
                "logloss_all_event_states": float(lle.mean()),
                "logloss_non_terminal_event_states": float(lle[~term].mean()),
                "logloss_terminal_event_states": float(lle[term].mean()) if term.any() else None,
                "grid_rows_with_a_side_dead": int(((te["alive_ct"] == 0) | (te["alive_t"] == 0)).sum())}
    print(f"terminal states in event test table: {terminal}", flush=True)

    # ---- what the grid model learned: probability vs time for fixed states
    curves = []
    secs = np.arange(0, 111)
    for (a_ct, a_t) in ((5, 5), (5, 4), (4, 5), (3, 3), (2, 1), (1, 2), (1, 1)):
        for planted_at in (None, 45):
            Xc = np.zeros((len(secs), len(FEATS)))
            for j, f in enumerate(FEATS):
                col = {"alive_ct": a_ct, "alive_t": a_t, "ct_alive_minus_t": a_ct - a_t,
                       "round_number": 10, "ct_score_minus_t_score": 0, "is_pistol_round": 0, "is_overtime": 0}.get(f)
                if col is not None:
                    Xc[:, j] = col
                elif f == "seconds_since_round_live":
                    Xc[:, j] = secs
                elif f == "bomb_planted":
                    Xc[:, j] = (secs >= planted_at) if planted_at is not None else 0
                elif f == "seconds_since_plant":
                    Xc[:, j] = np.where(secs >= planted_at, secs - planted_at, -1.0) if planted_at is not None else -1.0
            pc = mg.predict_proba(Xc, iteration_range=(0, best_g + 1))[:, 1]
            curves.append({"alive_ct": a_ct, "alive_t": a_t, "plant_at_sec": planted_at, "sec": secs.tolist(),
                           "p_ct_win": [float(v) for v in pc]})

    # ---- example traces from the test matches
    te_rounds = te.select(["match_key", "round", "label_ct_win"]).unique()
    r2 = np.random.default_rng(SEED)
    pick = []
    for lab in (1, 0):
        cand = te_rounds.filter(pl.col("label_ct_win") == lab).to_dicts()
        r2.shuffle(cand)
        n_ok = 0
        for c in cand:
            sub = te.filter((pl.col("match_key") == c["match_key"]) & (pl.col("round") == c["round"]))
            k = (c["match_key"], int(c["round"]))
            if 45 <= sub.height <= 110 and len(death_ev.get(k, [])) >= 4:
                pick.append(k)
                n_ok += 1
            if n_ok == 3:
                break
    traces = []
    for k in pick:
        m = (te["match_key"] == k[0]) & (te["round"] == k[1])
        sub = te.filter(m)
        sidx = np.where(m.to_numpy())[0]
        order = np.argsort(sub["seconds_since_round_live"].to_numpy())
        traces.append({"match": k[0][-40:], "round": k[1], "ct_won": int(sub["label_ct_win"][0]),
                       "sec": sub["seconds_since_round_live"].to_numpy()[order].tolist(),
                       "p_xgb": [float(v) for v in px[sidx][order]],
                       "p_lr_alive": [float(v) for v in pl1[sidx][order]],
                       "deaths": death_ev[k], "plant_sec": plant_sec[k]})

    out = {
        "split": {"train_matches": len(sp["train"]), "val_matches": len(sp["val"]), "test_matches": len(sp["test"]),
                  "train_grid_rows": tr.height, "val_grid_rows": va.height, "test_grid_rows": te.height,
                  "test_rounds": int(te.select(["match_key", "round"]).unique().height), "rule": "match-level, seed 0"},
        "xgb_best_iteration": {"grid": best_g, "event": best_e},
        "models": models, "paired": paired_out, "by_time_bin": bins, "calibration": cal,
        "terminal_state_inflation": terminal, "state_curves": curves, "traces": traces,
        "notes": ["Grid = freeze_end + 64k ticks strictly before round_end; 1 Hz stands in for every tick because the "
                  "features are piecewise constant between events except the two time features.",
                  "All intervals are match-cluster bootstrap (1000 resamples, seed 0).",
                  "Round-weighted log loss gives every round weight 1/(its grid points)."],
        "run_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    G.write_parquet(VIZ / "per_tick_states.parquet", compression="zstd")
    te_out = te.select(["match_key", "round", "tick", "seconds_since_round_live", "label_ct_win"]).with_columns(
        [pl.Series("p_constant", P["constant"]), pl.Series("p_lr_alive", P["LR alive difference"]),
         pl.Series("p_lr_all", P["LR all features"]), pl.Series("p_xgb_grid", P["XGBoost (trained on 1 Hz grid)"]),
         pl.Series("p_xgb_event", P["XGBoost (trained on event states)"])])
    te_out.write_parquet(VIZ / "per_tick_test_preds.parquet", compression="zstd")

    L = ["# Per-second win probability evaluation (advisor's rules)", "",
         "Script: `scripts/model/per_tick_eval.py` (seed 0). Split by match (bake-off primary split). 1 Hz grid of ticks, "
         "strictly before round end. Intervals are match-cluster bootstrap (1000 resamples).", "",
         f"Test: {out['split']['test_matches']} matches, {out['split']['test_rounds']} rounds, "
         f"{out['split']['test_grid_rows']} one-second rows.", "",
         "| Model | LL all ticks [95% CI] | LL round-weighted | Brier | AUC | ECE | SE cluster / naive |",
         "|---|---|---|---|---|---|---|"]
    for n, v in models.items():
        c = v["logloss_all_ticks_ci95"]
        L.append(f"| {n} | {v['logloss_all_ticks']:.4f} [{c[0]:.4f}, {c[1]:.4f}] | {v['logloss_round_weighted']:.4f} | "
                 f"{v['brier']:.4f} | {v['auc']:.4f} | {v['ece_10_equal_width']:.4f} | {v['cluster_over_naive_se']:.1f}x |")
    L += ["", "Paired log loss differences (negative = first model better):", ""]
    for k_, v in paired_out.items():
        L.append(f"- {k_}: {v['a_minus_b_all_ticks']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]; "
                 f"round-weighted {v['round_weighted']:+.4f}")
    L += ["", f"Terminal-state inflation (event table): {terminal}", ""]
    NOTE.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}, {NOTE}; done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
