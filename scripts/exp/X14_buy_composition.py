# X14: buy composition (hero / full / partial / light / eco) on the freeze-end 1 Hz snapshot.
import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = Path("G:/pureskill/csds-cs2")
DATA = ROOT / "data" / "state_sample"
NB = 1000
SEED = 0
T_START = time.time()

RIFLE_ONLY = [7, 8, 10, 11, 13, 16, 38, 39, 60]  # brief's rifle/AWP codes minus AWP (9)
AWP = 9
SSG08 = 40  # scout: not a rifle/AWP code in the brief, so non-rifle primary in the base variant
SMG = [17, 19, 23, 24, 26, 33, 34]
OTHER = [14, 25, 27, 28, 29, 35]  # M249, XM1014, MAG-7, Sawed-Off, Negev, Nova
TYPES = ["full", "hero", "partial", "light", "eco"]
TORDER = {t: i for i, t in enumerate(TYPES)}
UTIL_COL = {"flash": "n_flash", "smoke": "n_smoke", "he": "n_he", "molinc": "n_molinc", "decoy": "n_decoy"}
COMP_COLS = ["n_full", "n_light", "n_eco"]
CLASS_COLS = ["n_armor", "n_helmet", "n_defuser", "n_rifle", "n_sniper", "n_smg", "n_other"]
SIDE_FEAT = ["team_equip"] + COMP_COLS + list(UTIL_COL.values()) + CLASS_COLS

_BOOT = {}


def log(*a):
    print(*a, flush=True)


def elapsed():
    return f"[{time.time() - T_START:6.1f}s]"


def boot_weights(M):
    if M not in _BOOT:
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, M, size=(NB, M))
        W = np.zeros((NB, M))
        for b in range(NB):
            W[b] = np.bincount(idx[b], minlength=M)
        _BOOT[M] = W
    return _BOOT[M]


def boot_ratio(midx, num, den, M, mask=None):
    """Match-cluster bootstrap of sum(num)/sum(den). Returns point, lo, hi (95%)."""
    midx = np.asarray(midx)
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    if mask is not None:
        midx, num, den = midx[mask], num[mask], den[mask]
    S = np.bincount(midx, weights=num, minlength=M)
    N = np.bincount(midx, weights=den, minlength=M)
    point = S.sum() / N.sum() if N.sum() > 0 else np.nan
    W = boot_weights(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        bd = (W @ S) / (W @ N)
    lo, hi = np.nanpercentile(bd, [2.5, 97.5])
    return point, lo, hi


def pct_ci(t, scale=100.0):
    point, lo, hi = t
    return f"{scale * point:5.1f} [{scale * lo:5.1f}, {scale * hi:5.1f}]"


def load():
    p = pl.read_parquet(DATA / "players_1hz.parquet").with_columns(pl.col("round").cast(pl.Int64))
    r = pl.read_parquet(DATA / "rounds.parquet").with_columns(pl.col("round").cast(pl.Int64))
    bt = pl.read_parquet(DATA / "round_buy_types.parquet").with_columns(pl.col("round").cast(pl.Int64))
    return p, r, bt


def freeze_end(p):
    t0 = p.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t0"))
    return p.join(t0, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("t0")).drop("t0")


def classify(fe, variant):
    rifle_set = RIFLE_ONLY + [AWP] + ([SSG08] if variant == "S1" else [])
    is_rifle = pl.col("inv_primary").is_in(rifle_set)
    has_p = pl.col("inv_primary") != 0
    arm = pl.col("armor") > 0
    eq = pl.col("equipment_value_calc")
    full = is_rifle & arm
    light_mid = arm & (eq >= 1000) & (eq <= 3500)
    if variant == "S2":
        light = (has_p & ~is_rifle) | light_mid
    else:
        light = (has_p & ~is_rifle) | light_mid | (is_rifle & ~arm)
    cls = (
        pl.when(full).then(pl.lit("FULL"))
        .when(light).then(pl.lit("LIGHT"))
        .otherwise(pl.lit("ECO"))
        .alias("cls")
    )
    return fe.with_columns(cls)


def side_rounds(fe, variant):
    c = classify(fe, variant)
    sr = c.group_by(["match_key", "round", "team_code"]).agg([
        pl.len().alias("n_players"),
        (pl.col("cls") == "FULL").sum().alias("n_full"),
        (pl.col("cls") == "LIGHT").sum().alias("n_light"),
        (pl.col("cls") == "ECO").sum().alias("n_eco"),
        pl.col("equipment_value_calc").sum().alias("team_equip"),
        (pl.col("armor") > 0).sum().alias("n_armor"),
        pl.col("has_helmet").sum().alias("n_helmet"),
        pl.col("has_defuser").sum().alias("n_defuser"),
        pl.col("inv_flashbang").sum().alias("n_flash"),
        pl.col("inv_smokegrenade").sum().alias("n_smoke"),
        pl.col("inv_hegrenade").sum().alias("n_he"),
        (pl.col("inv_molotov") + pl.col("inv_incgrenade")).sum().alias("n_molinc"),
        pl.col("inv_decoy").sum().alias("n_decoy"),
        pl.col("inv_primary").is_in(RIFLE_ONLY).sum().alias("n_rifle"),
        pl.col("inv_primary").is_in([AWP, SSG08]).sum().alias("n_sniper"),
        pl.col("inv_primary").is_in(SMG).sum().alias("n_smg"),
        pl.col("inv_primary").is_in(OTHER).sum().alias("n_other"),
    ])
    sr = sr.with_columns(
        pl.when(pl.col("n_full") >= 4).then(pl.lit("full"))
        .when(pl.col("n_full").is_in([1, 2]) & (pl.col("n_eco") >= 2)).then(pl.lit("hero"))
        .when(pl.col("n_full").is_in([2, 3])).then(pl.lit("partial"))
        .when((pl.col("n_full") <= 1) & (pl.col("n_light") >= 3)).then(pl.lit("light"))
        .otherwise(pl.lit("eco"))
        .alias("ttype"),
        pl.when(pl.col("team_code") == 3).then(pl.lit("CT")).otherwise(pl.lit("T")).alias("side"),
    )
    return sr


def train_predict(X_tr, y_tr, X_te):
    params = {
        "device": "cuda", "tree_method": "hist", "max_depth": 3, "eta": 0.05,
        "subsample": 0.8, "objective": "binary:logistic", "eval_metric": "logloss", "seed": SEED,
    }
    for attempt in range(4):
        try:
            bst = xgb.train(params, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=300)
            return bst.predict(xgb.DMatrix(X_te))
        except xgb.core.XGBoostError as e:
            if "memory" in str(e).lower() and attempt < 3:
                log("  CUDA out of memory; waiting 30 s and retrying")
                time.sleep(30)
                continue
            raise


def logloss_rows(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    log(f"X14 buy composition run; polars {pl.__version__}; xgboost {xgb.__version__}")
    p, r, bt = load()
    log(f"players_1hz rows {p.height}; rounds {r.height}; buy-type rows {bt.height}")

    # --- score convention check (pre-round vs post-round) ---
    rs = r.sort(["match_key", "round"]).with_columns([
        pl.col("ct_score").shift(-1).over("match_key").alias("nxt_ct"),
        pl.col("t_score").shift(-1).over("match_key").alias("nxt_t"),
        pl.col("round").shift(-1).over("match_key").alias("nxt_round"),
    ])
    chk = rs.filter(pl.col("nxt_round") == pl.col("round") + 1)
    boundary = pl.col("round").is_in([12, 27])  # halftime (12->13) and first OT half-switch (27->28)
    cont_ok = chk.filter(~boundary & ((pl.col("nxt_ct") - pl.col("ct_score")) == pl.col("ct_win"))).height
    cont_n = chk.filter(~boundary).height
    swap_ok = chk.filter(boundary & (pl.col("nxt_ct") == pl.col("t_score") + (1 - pl.col("ct_win")))
                         & (pl.col("nxt_t") == pl.col("ct_score") + pl.col("ct_win"))).height
    swap_n = chk.filter(boundary).height
    log(f"score convention (pre-round, side-based): non-boundary pairs {cont_ok}/{cont_n} satisfy "
        f"nxt_ct-ct_score==ct_win; side-swap pairs {swap_ok}/{swap_n} satisfy the swapped-side rule "
        f"(post-round scores would fail the first rule everywhere)")

    # --- freeze-end snapshot ---
    fe_all = freeze_end(p)
    dup = fe_all.group_by(["match_key", "round", "player_id"]).agg(pl.len().alias("k")).filter(pl.col("k") > 1).height
    log(f"freeze-end rows {fe_all.height}; match-rounds {fe_all.select(['match_key', 'round']).unique().height}; "
        f"duplicate player rows {dup}; seconds_since_live at freeze-end {fe_all['seconds_since_live'].unique().to_list()}")
    ts = fe_all.group_by(["match_key", "round", "team_code"]).agg(pl.len().alias("n"))
    log("team sizes (size, team-rounds):", ts.group_by("n").agg(pl.len().alias("teams")).sort("n").rows())
    bad = ts.filter(pl.col("n") != 5).select(["match_key", "round"]).unique()
    log("dropped (team size != 5):", bad.rows())

    rA = r.join(bad, on=["match_key", "round"], how="anti")
    n_pistol = rA.filter(pl.col("round").is_in([1, 13])).height
    n_ot = rA.filter(pl.col("round") > 24).height
    rA = rA.filter((~pl.col("round").is_in([1, 13])) & (pl.col("round") <= 24))
    chk_p = rA.join(bt, on=["match_key", "round"], how="left")
    pist_mismatch = chk_p.filter(pl.col("is_pistol") != pl.col("round").is_in([1, 13])).height
    log(f"round filter: pistol rounds (1,13) before filter {n_pistol}; overtime (>24) before filter {n_ot}; "
        f"analysis rounds {rA.height}; is_pistol disagreements {pist_mismatch}; "
        f"rounds without buy-type row {chk_p.filter(pl.col('is_pistol').is_null()).height}")
    matches = sorted(rA["match_key"].unique().to_list())
    midx_of = {m: i for i, m in enumerate(matches)}
    M = len(matches)
    log(f"analysis matches {M}")

    fe = fe_all.join(rA.select(["match_key", "round"]), on=["match_key", "round"], how="inner")
    log(f"analysis freeze-end player rows {fe.height}; players per side-round {fe.height / (2 * rA.height):.3f} (expect 5.000)")

    # --- Step 1: grenade columns ---
    log("\n== STEP 1: utility columns ==")
    for c in ["inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy"]:
        vc = fe[c].value_counts().sort(c).rows()
        log(f"  {c}: values (value, player-rows) {vc}")

    sr = side_rounds(fe, "base")
    sr = sr.join(rA.select(["match_key", "round", "ct_win", "round_number", "ct_score", "t_score"]),
                 on=["match_key", "round"], how="inner")
    sr = sr.with_columns(
        pl.when(((pl.col("team_code") == 3) & (pl.col("ct_win") == 1)) | ((pl.col("team_code") == 2) & (pl.col("ct_win") == 0)))
        .then(1).otherwise(0).alias("win"),
        (pl.col("ct_score") - pl.col("t_score")).alias("score_diff"),
    )
    assert sr.height == 2 * rA.height, (sr.height, rA.height)
    assert (sr["n_full"] + sr["n_light"] + sr["n_eco"] == sr["n_players"]).all()
    midx_sr = np.array([midx_of[m] for m in sr["match_key"].to_list()])
    win = sr["win"].to_numpy().astype(float)
    ones = np.ones(sr.height)
    is_ct = (sr["side"] == "CT").to_numpy()
    ttype = np.array(sr["ttype"].to_list())
    log(f"side-rounds {sr.height} (CT {is_ct.sum()}, T {(~is_ct).sum()}); round-level {rA.height}")

    util_cols = list(UTIL_COL.values())
    util_tot = sum(sr[c].to_numpy() for c in util_cols)
    sr = sr.with_columns(pl.Series("util_total", util_tot))
    log("  mean per side-round (all): " + ", ".join(
        f"{c}={sr[c].mean():.3f}" for c in util_cols + ["util_total"]))
    for t in TYPES:
        sub = sr.filter(pl.col("ttype") == t)
        if sub.height == 0:
            continue
        log(f"  {t:8s} n={sub.height:5d} mean: " + ", ".join(
            f"{c}={sub[c].mean():.3f}" for c in util_cols + ["util_total"]))
    log(f"  utility share with 2 flashes: {(sr['n_flash'] == 2).mean():.3f}; mean flashes per side-round {sr['n_flash'].mean():.3f}")

    # --- Step 2: composition ---
    log("\n== STEP 2: player classes (base variant) ==")
    cls_counts = fe.with_columns(pl.col("inv_primary").is_in(RIFLE_ONLY + [AWP]).alias("rifle")).select(
        [pl.col("rifle").sum().alias("rifle_players"), (pl.col("rifle") & (pl.col("armor") == 0)).sum().alias("rifle_no_armor"),
         (pl.col("inv_primary") == SSG08).sum().alias("ssg08_players")]
    ).row(0)
    log(f"  rifle/AWP holders {cls_counts[0]}; rifle holders with armor 0 (residual -> LIGHT) {cls_counts[1]}; "
        f"SSG 08 holders (non-rifle in base) {cls_counts[2]}")
    log("  n_full distribution (side-rounds):", sr.group_by("n_full").agg(pl.len().alias("k")).sort("n_full").rows())
    log("  n_eco distribution:", sr.group_by("n_eco").agg(pl.len().alias("k")).sort("n_eco").rows())
    combos = sr.group_by(["n_full", "n_light", "n_eco"]).agg(pl.len().alias("k")).sort("k", descending=True).head(12)
    log("  top (n_full, n_light, n_eco) combos:", combos.rows())
    log("  team type counts by side (side-rounds):", sr.group_by(["ttype", "side"]).agg(pl.len().alias("k")).sort(["ttype", "side"]).rows())

    # --- Step 3: team types, win rates, matchups ---
    log("\n== STEP 3: team types and win rates (base) ==")
    hdr = f"  {'type':8s} {'n_all':>6s} {'n_CT':>6s} {'n_T':>6s} {'share%':>7s} {'eqMean':>8s} {'nFull':>6s} {'nLight':>6s} {'nEco':>6s} | {'win% all [95% CI]':>24s} | {'CT-side win% [CI]':>24s} | {'T-side win% [CI]':>24s}"
    log(hdr)
    n_tot = sr.height
    for t in TYPES:
        m = ttype == t
        sub = sr.filter(pl.col("ttype") == t)
        n_ct = int((m & is_ct).sum())
        n_t = int((m & ~is_ct).sum())
        if m.sum() == 0:
            log(f"  {t:8s} none")
            continue
        all_ci = boot_ratio(midx_sr, win, ones, M, mask=m)
        ct_ci = boot_ratio(midx_sr, win, ones, M, mask=m & is_ct)
        t_ci = boot_ratio(midx_sr, win, ones, M, mask=m & ~is_ct)
        log(f"  {t:8s} {m.sum():6d} {n_ct:6d} {n_t:6d} {100*m.sum()/n_tot:7.2f} "
            f"{sub['team_equip'].mean():8.0f} {sub['n_full'].mean():6.2f} {sub['n_light'].mean():6.2f} {sub['n_eco'].mean():6.2f} | "
            f"{pct_ci(all_ci):>24s} | {pct_ci(ct_ci) if n_ct else '-':>24s} | {pct_ci(t_ci) if n_t else '-':>24s}")
    log("  (win% is the win rate of the side whose team is that type; n counts side-rounds)")

    # round-level matchups
    ct_t = sr.filter(pl.col("side") == "CT").select(["match_key", "round", pl.col("ttype").alias("ct_ttype")])
    t_t = sr.filter(pl.col("side") == "T").select(["match_key", "round", pl.col("ttype").alias("t_ttype")])
    rd_t = rA.join(ct_t, on=["match_key", "round"]).join(t_t, on=["match_key", "round"])
    rd_midx = np.array([midx_of[m] for m in rd_t["match_key"].to_list()])
    ctw = rd_t["ct_win"].to_numpy().astype(float)
    cts = rd_t["ct_ttype"].to_list()
    tts = rd_t["t_ttype"].to_list()
    first_name, second_name, first_win = [], [], []
    for a, b, w in zip(cts, tts, ctw):
        if TORDER[a] <= TORDER[b]:
            first_name.append(a)
            second_name.append(b)
            first_win.append(w)  # CT side is first (also mirror case)
        else:
            first_name.append(b)
            second_name.append(a)
            first_win.append(1.0 - w)
    pairs = {}
    for i, (f, s) in enumerate(zip(first_name, second_name)):
        pairs.setdefault((f, s), []).append(i)
    top = sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:8]
    log("\n  eight most common matchups (first-listed = higher in full>hero>partial>light>eco; mirror = CT side):")
    log(f"  {'rank':>4s} {'matchup (first vs second)':30s} {'rounds':>7s} {'first-side win% [CI]':>24s}")
    for k, ((f, s), idx) in enumerate(top, 1):
        mask = np.zeros(rd_t.height, dtype=bool)
        mask[idx] = True
        fw = np.array(first_win)
        ci = boot_ratio(rd_midx, fw, np.ones(len(fw)), M, mask=mask)
        log(f"  {k:4d} {f + ' vs ' + s:30s} {len(idx):7d} {pct_ci(ci):>24s}")

    # hero vs others at matched team equipment
    log("\n  equipment bins: side-round counts by type, and win% of each type (pooled sides)")
    bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 1e9]
    labels = ["<5k", "5-10k", "10-15k", "15-20k", "20-25k", "25-30k", ">=30k"]
    eqv = sr["team_equip"].to_numpy()
    log(f"  {'bin':>7s} {'n':>5s} " + " ".join(f"{t:>8s}" for t in TYPES) + " | win% by type (n): " + " ".join(f"{t:>14s}" for t in TYPES))
    for i in range(len(labels)):
        bm = (eqv >= bins[i]) & (eqv < bins[i + 1])
        cnts = [int((bm & (ttype == t)).sum()) for t in TYPES]
        wins = []
        for t, cn in zip(TYPES, cnts):
            if cn >= 1:
                wins.append(f"{100 * win[bm & (ttype == t)].mean():5.1f}({cn})")
            else:
                wins.append("-")
        log(f"  {labels[i]:>7s} {bm.sum():5d} " + " ".join(f"{c:>8d}" for c in cnts) + " | " + " ".join(f"{w:>14s}" for w in wins))

    # --- Step 3 sensitivity ---
    log("\n== STEP 3 sensitivity ==")
    base_keys = sr.select(["match_key", "round", "team_code", "ttype"]).rename({"ttype": "ttype_base"})
    for variant, desc in [("S1", "SSG 08 counted as rifle/AWP (FULL with armor)"),
                          ("S2", "rifle with no armor -> ECO (not LIGHT)")]:
        sv = side_rounds(fe, variant).select(["match_key", "round", "team_code", "ttype", "team_equip"])
        j = base_keys.join(sv, on=["match_key", "round", "team_code"])
        changed = (j["ttype_base"] != j["ttype"]).mean()
        dist = j.group_by("ttype").agg([pl.len().alias("n"), pl.col("team_equip").mean().alias("eq")]).sort("ttype").rows()
        log(f"  {variant}: {desc}; side-rounds whose type changes {100*changed:.2f}%")
        log(f"    type counts (type, n, mean equip): {[(a, b, round(c)) for a, b, c in dist]}")
        hero_share = (j["ttype"] == "hero").mean()
        log(f"    hero share {100*hero_share:.2f}% of side-rounds")

    # buy-type consistency
    eq_side = sr.select(["match_key", "round", "side", "team_equip"]).pivot(values="team_equip", index=["match_key", "round"], on="side")
    bt_m = eq_side.join(bt.select(["match_key", "round", "ct_equipment", "t_equipment"]), on=["match_key", "round"])
    ct_match = (bt_m["CT"] == bt_m["ct_equipment"]).mean()
    t_match = (bt_m["T"] == bt_m["t_equipment"]).mean()
    log(f"\n  team equipment (sum of equipment_value_calc at freeze-end) equals round_buy_types ct/t_equipment: "
        f"CT {100*ct_match:.2f}%, T {100*t_match:.2f}%")

    # --- Step 4: gain test ---
    log("\n== STEP 4: gain test (round-level ct_win, 5-fold GroupKFold by match, XGBoost cuda) ==")
    feat_ct = sr.filter(pl.col("side") == "CT").select(["match_key", "round"] + [pl.col(c).alias(f"ct_{c}") for c in SIDE_FEAT])
    feat_t = sr.filter(pl.col("side") == "T").select(["match_key", "round"] + [pl.col(c).alias(f"t_{c}") for c in SIDE_FEAT])
    rd = rA.join(feat_ct, on=["match_key", "round"]).join(feat_t, on=["match_key", "round"]).with_columns(
        (pl.col("ct_score") - pl.col("t_score")).alias("score_diff")
    )
    assert rd.height == rA.height
    rd_midx2 = np.array([midx_of[m] for m in rd["match_key"].to_list()])
    y = rd["ct_win"].to_numpy().astype(float)
    ctx = ["round_number", "score_diff"]
    sA = ctx + ["ct_team_equip", "t_team_equip"]
    sB = sA + [f"{s}_{c}" for s in ["ct", "t"] for c in COMP_COLS]
    sC = sB + [f"{s}_{UTIL_COL[u]}" for s in ["ct", "t"] for u in UTIL_COL]
    sD = sC + [f"{s}_{c}" for s in ["ct", "t"] for c in CLASS_COLS]
    sets = [("0 context only", ctx), ("a equipment", sA), ("b + composition", sB), ("c + utility", sC), ("d + armor/helmet/defuser/weapon classes", sD)]
    log(f"  rounds {rd.height}; matches {M}; target CT win rate {y.mean():.4f}")
    oofs = {}
    for name, cols in sets:
        X = rd.select(cols).to_numpy().astype(np.float32)
        oof = np.zeros(len(y))
        base_oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, rd_midx2):
            oof[te] = train_predict(X[tr], y[tr], X[te])
            base_oof[te] = y[tr].mean()
        ll = logloss_rows(y, oof)
        ll_base = logloss_rows(y, base_oof)
        oofs[name] = ll
        ci = boot_ratio(rd_midx2, ll, np.ones(len(ll)), M)
        cib = boot_ratio(rd_midx2, ll_base, np.ones(len(ll)), M)
        log(f"  {name:40s} n_feat={len(cols):2d} OOF logloss {ci[0]:.5f} [{ci[1]:.5f}, {ci[2]:.5f}]; "
            f"fold-base-rate logloss {cib[0]:.5f}; ok {elapsed()}")
    seq = [s[0] for s in sets]
    log("  paired differences (new minus previous; negative = improvement), match-cluster 95% CI:")
    for i in range(1, len(seq)):
        dlt = oofs[seq[i]] - oofs[seq[i - 1]]
        d = boot_ratio(rd_midx2, dlt, np.ones(len(dlt)), M)
        log(f"    {seq[i]:40s} minus {seq[i-1]:32s}: {d[0]:+.5f} [{d[1]:+.5f}, {d[2]:+.5f}]")
    dlt = oofs[seq[-1]] - oofs[seq[1]]
    d = boot_ratio(rd_midx2, dlt, np.ones(len(dlt)), M)
    log(f"    d minus a (total composition+utility+classes): {d[0]:+.5f} [{d[1]:+.5f}, {d[2]:+.5f}]")
    log(f"\nDone {elapsed()}")


if __name__ == "__main__":
    main()
