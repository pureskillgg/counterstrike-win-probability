"""X12 steps 2-5: callout site zones, per-second callout features, and the CT-win test.

Inputs (read only): data\\state_sample\\players_1hz.parquet, rounds.parquet, place_names.parquet
(written by scripts\\exp\\X12_extract_place_names.py).

Site zones (step 2). Per map, the place names that are often inside the bomb zone
(is_in_bomb_zone share >= 0.5 and >= 20 bomb-zone rows) are the site candidates. Their
row-weighted mean positions are split into two clusters with a weighted 2-means (farthest-point
start, heaviest candidate first). S1 is the cluster holding the heaviest bomb-zone place; S2 is the
other. Every place on that map is then labelled by its nearest site centre if its mean position is
within SITE_THR * (site separation) of that centre, and "mid" otherwise (corridors, spawns, far
areas). Rows with no place name are "unk".

Features (step 3), one row per (match, round, live-second snapshot). Only the current snapshot and
earlier ones are used:
  per side counts in S1, S2, mid, unk (the 6 numbers, t_s1 t_s2 t_mid, ct_s1 ct_s2 ct_mid);
  t_site_max, t_site_min, ct_stack (= max CT in S1, S2), ct_site_min, ct_site_total;
  ct_site_delta5 = change in CT site total over the last 5 s; ct_rot5_abs = |change| in S1 + |change| in S2;
  t_rush_10_20 = running max of T count in the larger site zone over seconds 10..min(t, 20) (null before 10 s);
  base: nT, nCT, bomb_planted, seconds_since_plant, seconds_since_live, round_number, score_lead
  (= ct_score - t_score at live start), pistol (round_number in 1, 13).

Model (step 4). 5-fold GroupKFold by match (greedy largest-first assignment, the same scheme as
sklearn GroupKFold, implemented here). XGBoost on CUDA, depth 4, eta 0.05, 300 rounds, subsample 0.8.
Out-of-fold log loss per row, by time bin, with a 1000-resample match-cluster bootstrap (seed 0) on
the paired difference (callout minus base).

Step 5 cost numbers are logged from the extraction timing file.

Run from the project root:
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\exp\\X12_callout_model.py
"""

import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = Path(r"G:\pureskill\csds-cs2")
SAMPLE = ROOT / "data" / "state_sample"
LOG_PATH = ROOT / "logs" / "exp" / "X12_model.log"
T, CT = 2, 3
SEED = 0
N_BOOT = 1000
N_FOLDS = 5
NROUNDS = 300
SITE_THR = 0.35
KEY = ["match_key", "round", "tick", "player_id"]
BINS = [(0, 10, "0-10 s"), (10, 20, "10-20 s"), (20, 30, "20-30 s"), (30, 45, "30-45 s"),
        (45, 60, "45-60 s"), (60, 10 ** 9, "60+ s")]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4, "eta": 0.05,
          "subsample": 0.8, "device": "cuda", "tree_method": "hist", "seed": SEED, "verbosity": 0}
BASE = ["nT", "nCT", "bomb_planted", "seconds_since_plant", "seconds_since_live", "round_number",
        "score_lead", "pistol"]
CALLOUT = ["t_s1", "t_s2", "t_mid", "t_unk", "ct_s1", "ct_s2", "ct_mid", "ct_unk", "t_site_max",
           "t_site_min", "ct_stack", "ct_site_min", "ct_site_total", "ct_site_delta5", "ct_rot5_abs",
           "t_rush_10_20"]

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_logf = open(LOG_PATH, "w", encoding="utf-8")


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s, flush=True)
    _logf.write(s + "\n")
    _logf.flush()


def load_joined():
    ps = pl.read_parquet(SAMPLE / "players_1hz.parquet", columns=[
        "match_key", "round", "tick", "seconds_since_live", "player_id", "team_code", "x_pos", "y_pos"])
    ps = ps.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                         pl.col("player_id").cast(pl.Int64), pl.col("team_code").cast(pl.Int64),
                         pl.col("seconds_since_live").cast(pl.Int64))
    pn = pl.read_parquet(SAMPLE / "place_names.parquet").with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
        pl.col("player_id").cast(pl.Int64))
    ps = ps.join(pn.select(KEY + ["place_name", "is_in_bomb_zone"]), on=KEY, how="left")
    log(f"players_1hz rows {ps.height}; rows with no place_status match "
        f"{ps.filter(pl.col('is_in_bomb_zone').is_null()).height}")
    rd = pl.read_parquet(SAMPLE / "rounds.parquet").with_columns(pl.col("round").cast(pl.Int64))
    ps = ps.join(rd.select(["match_key", "round", "map_name", "ct_win", "plant_tick", "ct_score",
                            "t_score", "round_number", "live_start_tick"]),
                 on=["match_key", "round"], how="left")
    ps = ps.with_columns(
        pl.when(pl.col("place_name").is_not_null() & (pl.col("place_name").str.strip_chars() != ""))
        .then(pl.col("place_name").str.strip_chars()).otherwise(None).alias("place_clean"),
        pl.col("is_in_bomb_zone").fill_null(False))
    return ps, rd


def site_zones(ps):
    log("== 2. Site zones per map ==")
    lookup_rows = []
    for mp in sorted(ps["map_name"].unique().to_list()):
        sub = ps.filter((pl.col("map_name") == mp) & pl.col("place_clean").is_not_null())
        st = sub.group_by("place_clean").agg(
            pl.len().alias("n"), pl.col("x_pos").mean().alias("mx"), pl.col("y_pos").mean().alias("my"),
            pl.col("is_in_bomb_zone").sum().alias("bz_n")).with_columns(
            (pl.col("bz_n") / pl.col("n")).alias("bz_share"))
        # is_in_bomb_zone is a sparse flag (about 7% of BombsiteA rows), so the candidate rule is a
        # minimum count of bomb-zone rows, not a share of the place's rows
        cand = st.filter(pl.col("bz_n") >= 20).sort("bz_n", descending=True)
        if cand.height < 2:
            log(f"  WARNING {mp}: fewer than 2 bomb-zone candidates, using the top 2 by bomb-zone rows")
            cand = st.sort("bz_n", descending=True).head(2)
        P = cand.select(["mx", "my"]).to_numpy().astype(float)
        W = cand["bz_n"].to_numpy().astype(float)
        d0 = ((P - P[0]) ** 2).sum(1)
        c = np.vstack([P[0], P[int(d0.argmax())]])
        for _ in range(100):
            d = ((P[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            new = c.copy()
            for kk in range(2):
                msk = lab == kk
                if msk.any():
                    new[kk] = np.average(P[msk], axis=0, weights=W[msk])
            if np.allclose(new, c):
                break
            c = new
        # orient so S1 = the site holding the callout 'BombsiteA' (A) and S2 = the other (B), so the
        # raw per-site counts mean the same thing on every map; fall back to the heaviest candidate
        d = ((P[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        cand_names = cand["place_clean"].to_list()
        if "BombsiteA" in cand_names:
            a_cluster = int(lab[cand_names.index("BombsiteA")])
        else:
            a_cluster = int(np.argmin(((c - P[0]) ** 2).sum(1)))
        if a_cluster == 1:
            c = c[::-1].copy()
        L = float(np.linalg.norm(c[1] - c[0]))
        names = st["place_clean"].to_list()
        Q = np.column_stack([st["mx"].to_numpy(), st["my"].to_numpy()])
        dq = np.sqrt(((Q[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        near = dq.argmin(1)
        dmin = dq.min(1)
        zone = np.where(dmin <= SITE_THR * L, np.where(near == 0, "S1", "S2"), "mid")
        for nm, z in zip(names, zone.tolist()):
            lookup_rows.append({"map_name": mp, "place_clean": nm, "zone": z})
        zs = st.with_columns(pl.Series("zone", zone))
        share_bz = zs.filter(pl.col("bz_n") > 0).group_by("zone").agg(pl.col("bz_n").sum())
        bz_tot = int(zs["bz_n"].sum())
        bz_in = {r["zone"]: r["bz_n"] for r in share_bz.iter_rows(named=True)}
        log(f"  {mp}: site centres S1 ({c[0][0]:.0f}, {c[0][1]:.0f}), S2 ({c[1][0]:.0f}, {c[1][1]:.0f}), "
            f"separation {L:.0f}, threshold {SITE_THR * L:.0f}")
        log(f"    site candidates (bomb-zone rows): "
            + ", ".join(f"{r['place_clean']}({r['bz_n']})" for r in cand.iter_rows(named=True)))
        for z in ("S1", "S2", "mid"):
            zz = zs.filter(pl.col("zone") == z).sort("n", descending=True)
            top = ", ".join(f"{r['place_clean']}({r['n']})" for r in zz.head(8).iter_rows(named=True))
            log(f"    {z}: {zz.height} places, {int(zz['n'].sum())} rows; top: {top}")
        log(f"    bomb-zone rows on S1 {bz_in.get('S1', 0)}, S2 {bz_in.get('S2', 0)}, mid "
            f"{bz_in.get('mid', 0)} (of {bz_tot} bomb-zone rows with a place name)")
    lookup = pl.DataFrame(lookup_rows)
    return ps.join(lookup, on=["map_name", "place_clean"], how="left")


def build_snapshots(ps):
    z = pl.col("zone")
    t = pl.col("team_code") == T
    c = pl.col("team_code") == CT
    snap = ps.group_by(["match_key", "round", "tick"]).agg(
        t.sum().alias("nT"), c.sum().alias("nCT"),
        (t & (z == "S1")).sum().alias("t_s1"), (t & (z == "S2")).sum().alias("t_s2"),
        (t & (z == "mid")).sum().alias("t_mid"), (t & z.is_null()).sum().alias("t_unk"),
        (c & (z == "S1")).sum().alias("ct_s1"), (c & (z == "S2")).sum().alias("ct_s2"),
        (c & (z == "mid")).sum().alias("ct_mid"), (c & z.is_null()).sum().alias("ct_unk"),
        pl.col("map_name").first(), pl.col("ct_win").first(), pl.col("plant_tick").first(),
        pl.col("ct_score").first(), pl.col("t_score").first(), pl.col("round_number").first(),
        pl.col("seconds_since_live").first())
    snap = snap.sort(["match_key", "round", "tick"])
    # counts come out of sum() as UInt32; cast before any subtraction (UInt32 wraps on negatives)
    snap = snap.with_columns([pl.col(c).cast(pl.Int64) for c in
                              ["nT", "nCT", "t_s1", "t_s2", "t_mid", "t_unk", "ct_s1", "ct_s2",
                               "ct_mid", "ct_unk"]])
    dup_keys = snap.height - snap.select(["match_key", "round", "seconds_since_live"]).n_unique()
    log(f"  duplicate (match, round, second) snapshot keys: {dup_keys}")
    snap = snap.with_columns(
        (pl.col("ct_s1") + pl.col("ct_s2")).alias("ct_site_total"),
        pl.min_horizontal("ct_s1", "ct_s2").alias("ct_site_min"),
        pl.max_horizontal("ct_s1", "ct_s2").alias("ct_stack"),
        pl.min_horizontal("t_s1", "t_s2").alias("t_site_min"),
        pl.max_horizontal("t_s1", "t_s2").alias("t_site_max"),
        (pl.col("plant_tick").is_not_null() & (pl.col("tick") >= pl.col("plant_tick")))
        .cast(pl.Int64).alias("bomb_planted"),
        (pl.col("ct_score") - pl.col("t_score")).alias("score_lead"),
        pl.col("round_number").is_in([1, 13]).cast(pl.Int64).alias("pistol"),
    )
    snap = snap.with_columns(
        pl.when(pl.col("bomb_planted") == 1)
        .then((pl.col("tick") - pl.col("plant_tick")) / 64.0).otherwise(-1.0)
        .alias("seconds_since_plant"))
    lag = snap.select(["match_key", "round", "seconds_since_live", "ct_site_total", "ct_s1", "ct_s2"]) \
        .with_columns(pl.col("seconds_since_live") + 5) \
        .rename({"ct_site_total": "lag_ct_site_total", "ct_s1": "lag_ct_s1", "ct_s2": "lag_ct_s2"})
    snap = snap.join(lag, on=["match_key", "round", "seconds_since_live"], how="left")
    snap = snap.with_columns(
        (pl.col("ct_site_total") - pl.col("lag_ct_site_total")).alias("ct_site_delta5"),
        ((pl.col("ct_s1") - pl.col("lag_ct_s1")).abs() + (pl.col("ct_s2") - pl.col("lag_ct_s2")).abs())
        .alias("ct_rot5_abs"))
    sec = pl.col("seconds_since_live")
    snap = snap.with_columns(
        pl.when((sec >= 10) & (sec <= 20)).then(pl.col("t_site_max")).otherwise(-1).alias("_w"))
    snap = snap.with_columns(pl.col("_w").cum_max().over(["match_key", "round"]).alias("_rm"))
    snap = snap.with_columns(
        pl.when((sec < 10) | (pl.col("_rm") < 0)).then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("_rm")).cast(pl.Float64).alias("t_rush_10_20"))
    return snap


def logloss(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_retry(dtr):
    for attempt in range(4):
        try:
            return xgb.train(PARAMS, dtr, NROUNDS)
        except xgb.core.XGBoostError as e:
            if "memory" in str(e).lower() and attempt < 3:
                log(f"  CUDA out of memory, waiting 30 s and retrying ({attempt + 1})")
                time.sleep(30)
            else:
                raise


def fit_oof(X, y, fold, tag):
    oof = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        bst = train_retry(xgb.DMatrix(X[tr], label=y[tr]))
        oof[te] = bst.predict(xgb.DMatrix(X[te]))
        log(f"  {tag} fold {k}: train rows {int(tr.sum())}, test rows {int(te.sum())}")
    return oof


def main():
    t_all = time.perf_counter()
    ps, rd = load_joined()
    ps = site_zones(ps)
    snap = build_snapshots(ps)
    n_match = snap["match_key"].n_unique()
    n_round = snap.select(["match_key", "round"]).n_unique()
    log(f"== 3. Snapshots: {snap.height} rows, {n_match} matches, {n_round} rounds")
    log(f"  CT win share {snap['ct_win'].mean():.4f}; null t_rush share {snap['t_rush_10_20'].is_null().mean():.4f}; "
        f"null ct_rot5 share {snap['ct_rot5_abs'].is_null().mean():.4f}")
    log("  feature means: " + ", ".join(
        f"{c}={snap[c].cast(pl.Float64).mean():.2f}" for c in BASE + CALLOUT))

    # folds: greedy largest-first match assignment (GroupKFold scheme)
    sizes = snap.group_by("match_key").len().sort(["len", "match_key"], descending=[True, False]).to_dicts()
    fold_n = np.zeros(N_FOLDS)
    fold_of = {}
    for r in sizes:
        k = int(np.argmin(fold_n))
        fold_of[r["match_key"]] = k
        fold_n[k] += r["len"]
    fold_df = pl.DataFrame({"match_key": list(fold_of.keys()), "fold": list(fold_of.values())})
    snap = snap.join(fold_df, on="match_key", how="left")
    fold = snap["fold"].to_numpy()
    y = snap["ct_win"].to_numpy().astype(float)
    sec = snap["seconds_since_live"].to_numpy().astype(float)
    X_base = snap.select([pl.col(c).cast(pl.Float64) for c in BASE]).to_numpy()
    X_call = snap.select([pl.col(c).cast(pl.Float64) for c in BASE + CALLOUT]).to_numpy()

    log("== 4. XGBoost 5-fold GroupKFold by match (cuda, depth 4, eta 0.05, 300 rounds, subsample 0.8) ==")
    t0 = time.perf_counter()
    oof_b = fit_oof(X_base, y, fold, "base")
    oof_c = fit_oof(X_call, y, fold, "callout")
    log(f"  fits done in {time.perf_counter() - t0:.1f} s")

    lb = logloss(oof_b, y)
    lc = logloss(oof_c, y)
    log(f"overall: n rows {len(y)}, matches {n_match}; base {lb.mean():.4f}; callout {lc.mean():.4f}; "
        f"diff (callout - base) {lc.mean() - lb.mean():+.4f}")

    # match index and per-match, per-bin sums for the cluster bootstrap
    mkeys = sorted(snap["match_key"].unique().to_list())
    mi = snap["match_key"].replace_strict({m: i for i, m in enumerate(mkeys)}, return_dtype=pl.Int64).to_numpy()
    M = len(mkeys)
    nb = len(BINS)
    bi = np.full(len(y), -1)
    for j, (lo, hi, _) in enumerate(BINS):
        bi[(sec >= lo) & (sec < hi)] = j
    valid = bi >= 0
    comb = mi[valid] * nb + bi[valid]
    SB = np.bincount(comb, weights=lb[valid], minlength=M * nb).reshape(M, nb)
    SC = np.bincount(comb, weights=lc[valid], minlength=M * nb).reshape(M, nb)
    NN = np.bincount(comb, minlength=M * nb).reshape(M, nb).astype(float)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))

    rows = []
    labels = [b[2] for b in BINS] + ["overall"]
    for j in range(nb + 1):
        if j < nb:
            sb_full, sc_full, nn_full = SB[:, j], SC[:, j], NN[:, j]
        else:
            sb_full, sc_full, nn_full = SB.sum(1), SC.sum(1), NN.sum(1)
        n_rows = int(nn_full.sum())
        n_mat = int((nn_full > 0).sum())
        base_mean = sb_full.sum() / n_rows
        call_mean = sc_full.sum() / n_rows
        sb_b = sb_full[idx].sum(1)
        sc_b = sc_full[idx].sum(1)
        nn_b = nn_full[idx].sum(1)
        ok = nn_b > 0
        diff_b = (sc_b[ok] / nn_b[ok]) - (sb_b[ok] / nn_b[ok])
        base_b = sb_b[ok] / nn_b[ok]
        call_b = sc_b[ok] / nn_b[ok]
        lo_d, hi_d = np.percentile(diff_b, [2.5, 97.5])
        rows.append({"bin": labels[j], "rows": n_rows, "matches": n_mat, "base": base_mean,
                     "callout": call_mean, "diff": call_mean - base_mean, "diff_lo": lo_d,
                     "diff_hi": hi_d, "share_callout_better": float((diff_b < 0).mean()),
                     "base_lo": np.percentile(base_b, 2.5), "base_hi": np.percentile(base_b, 97.5),
                     "call_lo": np.percentile(call_b, 2.5), "call_hi": np.percentile(call_b, 97.5)})
    log("== 4b. Out-of-fold log loss by time bin (match-cluster bootstrap, 1000 resamples, seed 0) ==")
    log(f"{'bin':10s} {'rows':>8s} {'matches':>7s} {'base':>7s} {'[95% CI]':>17s} {'callout':>8s} "
        f"{'[95% CI]':>17s} {'diff':>8s} {'[95% CI]':>19s} {'P(diff<0)':>9s}")
    for r in rows:
        log(f"{r['bin']:10s} {r['rows']:8d} {r['matches']:7d} {r['base']:7.4f} "
            f"[{r['base_lo']:.4f}, {r['base_hi']:.4f}] {r['callout']:8.4f} "
            f"[{r['call_lo']:.4f}, {r['call_hi']:.4f}] {r['diff']:+8.4f} "
            f"[{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}] {r['share_callout_better']:9.3f}")

    # feature gain on the full sample (descriptive, one fit)
    log("== gain importance, callout model fit on all rows (descriptive) ==")
    bst = train_retry(xgb.DMatrix(X_call, label=y, feature_names=BASE + CALLOUT))
    gain = bst.get_score(importance_type="gain")
    for k, v in sorted(gain.items(), key=lambda kv: -kv[1])[:14]:
        log(f"  {k}: {v:.1f}")

    log(f"total wall {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
