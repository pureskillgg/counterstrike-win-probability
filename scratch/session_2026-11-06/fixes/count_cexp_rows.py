import sys
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "G:/pureskill/csds-cs2"
WORK = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/fixes"
PARITY = {"c050": f"{WORK}/parity_bucket_c050.parquet", "c060": f"{WORK}/parity_bucket_c060.parquet"}
TOL_DEG, TOL_U = 1e-3, 1e-2
FEATS = ["c_exposed", "t_exposed", "c_sees_enemy", "t_sees_enemy"]
KEYS = ["match_key", "round", "tick"]
HOLD_DATES = {f"2026-10-{d}" for d in range(26, 32)}

with open(f"{WORK}/holdout_b_keys.txt", encoding="utf-8") as fh:
    HB = {ln.strip() for ln in fh if ln.strip()}
print("holdout B keys loaded (count only):", len(HB))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def rule(A, dt):
    cx, cy, cyaw = (A["c"][k].astype(dt) for k in ("x", "y", "yaw"))
    tx, ty, tyaw = (A["t"][k].astype(dt) for k in ("x", "y", "yaw"))
    with np.errstate(all="ignore"):
        dx = tx[:, None, :] - cx[:, :, None]
        dy = ty[:, None, :] - cy[:, :, None]
        D = np.sqrt(dx ** 2 + dy ** 2)
        brg = np.degrees(np.arctan2(dy, dx))
        mc = np.abs(wrap(cyaw[:, :, None] - brg)) - dt(45.0)
        mt = np.abs(wrap(tyaw[:, None, :] - (brg + dt(180.0)))) - dt(45.0)
        c_in_t = mc < 0
        t_in_c = mt < 0
        out = {
            "c_exposed": ((D < 1500.0) & t_in_c & ~c_in_t).any(2).sum(1).astype(np.float64),
            "t_exposed": ((D < 1500.0) & c_in_t & ~t_in_c).any(1).sum(1).astype(np.float64),
            "c_sees_enemy": ((D < 2000.0) & c_in_t).any(2).sum(1).astype(np.float64),
            "t_sees_enemy": ((D < 2000.0) & t_in_c).any(1).sum(1).astype(np.float64),
        }
        near_ang = ((np.abs(mc) < TOL_DEG) | (np.abs(mt) < TOL_DEG)) & (D < 2000.0)
        near_d = (np.abs(D - 1500.0) < TOL_U) | (np.abs(D - 2000.0) < TOL_U)
        near = near_ang.any((1, 2)) | near_d.any((1, 2))
    return out, near


def per_tick(pos):
    K = pos.select(KEYS).unique(maintain_order=True).with_row_index("kid")
    j = pos.join(K, on=KEYS).sort("kid", "side", "player_id")
    j = j.with_columns(pl.int_range(pl.len()).over(["kid", "side"]).alias("slot")).filter(pl.col("slot") < 5)
    n = K.height
    A = {"c": {}, "t": {}}
    for nm, sd in (("c", 3), ("t", 2)):
        s = j.filter(pl.col("side") == sd)
        kid = s["kid"].to_numpy()
        sl = s["slot"].to_numpy()
        for c in ("x", "y", "yaw"):
            a = np.full((n, 5), np.nan, dtype=np.float64)
            a[kid, sl] = s[c].to_numpy().astype(np.float64)
            A[nm][c] = a
    return K, A


tot = {}


def bump(k, v=1):
    tot[k] = tot.get(k, 0) + v


examples = []
for tag, path in PARITY.items():
    par = pl.read_parquet(path, columns=KEYS + FEATS)
    par = par.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                           (pl.col("match_key").str.slice(5, 10).str.replace_all("/", "-")).alias("day"))
    n_all = par.height
    m_date = par["day"].is_in(list(HOLD_DATES))
    m_hb = par["match_key"].is_in(list(HB))
    bump(f"{tag}.rows_total", n_all)
    bump(f"{tag}.rows_excluded_oct26_31", int(m_date.sum()))
    bump(f"{tag}.rows_excluded_holdoutB_not_already", int((m_hb & ~m_date).sum()))
    par = par.filter(~m_date & ~m_hb)
    bump(f"{tag}.rows_used", par.height)
    days = sorted(par["day"].unique().to_list())
    for day in days:
        pd_ = par.filter(pl.col("day") == day).select(KEYS + FEATS)
        pp = f"{ROOT}/data/positions_1hz/{day}.parquet"
        fp = f"{ROOT}/data/pos_features_1hz/every2_sites_v1/{day}.parquet"
        try:
            pos = pl.read_parquet(pp, columns=["match_key", "round", "tick", "side", "player_id", "x", "y", "yaw"])
        except Exception as e:
            bump(f"{tag}.rows_no_laptop_positions", pd_.height)
            continue
        pos = pos.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
        pos = pos.join(pd_.select(KEYS).unique(), on=KEYS, how="semi")
        if pos.height == 0:
            bump(f"{tag}.rows_no_laptop_positions", pd_.height)
            continue
        K, A = per_tick(pos)
        ex, near = rule(A, np.float64)
        f32, _ = rule(A, np.float32)
        R = K.select(KEYS).with_columns(pl.Series("near", near))
        for k in FEATS:
            R = R.with_columns(pl.Series(f"x64_{k}", ex[k]), pl.Series(f"x32_{k}", f32[k]))
        feat = pl.read_parquet(fp, columns=KEYS + FEATS).with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
        feat = feat.join(pd_.select(KEYS).unique(), on=KEYS, how="semi")
        feat = feat.rename({k: f"lap_{k}" for k in FEATS})
        M = pd_.join(R, on=KEYS, how="inner").join(feat, on=KEYS, how="left")
        bump(f"{tag}.rows_covered", M.height)
        bump(f"{tag}.rows_not_covered", pd_.height - M.height)
        for k in FEATS:
            p = M[k].cast(pl.Float64).to_numpy()
            e = M[f"x64_{k}"].to_numpy()
            f = M[f"x32_{k}"].to_numpy()
            lp = M[f"lap_{k}"].cast(pl.Float64).to_numpy()
            def ne(a, b):
                return ~((a == b) | (np.isnan(a) & np.isnan(b)))
            bump(f"{tag}.{k}.parity_vs_exact64", int(ne(p, e).sum()))
            bump(f"{tag}.{k}.parity_vs_laptop_day", int(ne(p, lp).sum()))
            bump(f"{tag}.{k}.laptop_day_vs_exact64", int(ne(lp, e).sum()))
            bump(f"{tag}.{k}.laptop_day_vs_f32recompute", int(ne(lp, f).sum()))
            bump(f"{tag}.{k}.parity_vs_f32laptoprecompute", int(ne(p, f).sum()))
            bump(f"{tag}.{k}.parity_vs_exact64_ROWS_wrong_when_near", int((ne(p, e) & M["near"].to_numpy()).sum()))
        bump(f"{tag}.rows_flagged_near_boundary", int(M["near"].to_numpy().sum()))
        bad = M.filter(pl.Series(M["c_exposed"].cast(pl.Float64).to_numpy() != M["x64_c_exposed"].to_numpy()))
        for r in bad.head(3).iter_rows(named=True):
            if len(examples) < 12:
                examples.append((tag, r["match_key"][-40:], r["round"], r["tick"], r["c_exposed"], r["x64_c_exposed"], r["lap_c_exposed"], r["near"]))
        print(f"{tag} {day}: rows {pd_.height}, covered {M.height}", flush=True)

print("\n==== TOTALS ====")
for k in sorted(tot):
    print(f"{k}: {tot[k]}")
print("\n==== sample rows where parity c_exposed != exact float64 ====")
for e in examples:
    print(e)
