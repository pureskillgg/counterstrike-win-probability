"""A1: armor statistics, non-pistol rounds, 1 Hz freeze-end sample (150 matches).

Unit of analysis: player (armor/helmet/defuser/weapon class) or round (win models).
Freeze-end state = first snapshot per (match_key, round) (seconds_since_live == 0 for all rounds).
Pistol rounds: 1 and 13. Non-pistol (NP) = rounds 2-12 and 14-24. Overtime (25+) reported separately.
Uncertainty: match-cluster bootstrap, B=1000, numpy default_rng(0), resampling matches (not rounds);
percentile 95% intervals. The same resample weights are used for every statistic (paired contrasts).
Logistic models: weighted IRLS on CT win (ct_win from rounds.parquet), intercept included.
Read-only inputs; prints tables to stdout (captured in logs/stats/A1_main.log).
"""
import time

import numpy as np
import polars as pl

T0 = time.time()
SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
ROUNDS = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
B = 1000
SEED = 0

RIFLE = [7, 16, 60, 8, 39, 13, 10]          # AK-47, M4A4, M4A1-S, AUG, SG553, Galil, FAMAS
SNIPER = [9, 11, 38, 40]                    # AWP, G3SG1, SCAR-20, SSG 08
OTHER_PRIMARY = [14, 17, 19, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35]  # LMG, SMG, shotgun
PISTOLS = [1, 2, 3, 4, 30, 32, 36, 61, 63, 64]  # deagle, dualies, fiveseven, glock, tec9, p2000, p250, usp-s, cz75, r8


def log(*a):
    print(*a, flush=True)


# numpy.linalg (LAPACK) crashes in this environment (Windows fatal exception 0xc06d007f inside lstsq),
# so all linear algebra below uses einsum sums and a small pure-numpy Gaussian solver (no BLAS/LAPACK).
def mv(X, b):
    return np.einsum("ni,i->n", X, b)


def solve_small(A, b):
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    n = len(b)
    for i in range(n):
        p = i + int(np.argmax(np.abs(A[i:, i])))
        if p != i:
            A[[i, p]] = A[[p, i]]
            b[[i, p]] = b[[p, i]]
        piv = A[i, i]
        for j in range(i + 1, n):
            f = A[j, i] / piv
            A[j, i:] = A[j, i:] - f * A[i, i:]
            b[j] = b[j] - f * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.sum(A[i, i + 1:] * x[i + 1:])) / A[i, i]
    return x


def ols(X, y):
    A = np.einsum("ni,nj->ij", X, X) + 1e-10 * np.eye(X.shape[1])
    rhs = np.einsum("ni,n->i", X, y)
    return solve_small(A, rhs)


def fmt_p(p, lo, hi):
    return f"{100 * p:6.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}]"


def fmt_x(p, lo, hi, nd=2):
    return f"{p:8.{nd}f} [{lo:7.{nd}f}, {hi:7.{nd}f}]"


def fmt_s(p, lo, hi, nd=4):
    return f"{p:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


# ------------------------------------------------------------------ load
COLS = ["match_key", "round", "tick", "seconds_since_live", "split", "player_id", "team_code",
        "armor", "has_helmet", "has_defuser", "money", "equipment_value_calc", "inv_primary",
        "inv_secondary", "weapon_code", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade",
        "inv_molotov", "inv_incgrenade", "inv_decoy", "inv_c4"]
TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
lf = pl.scan_parquet(SAMPLE).select(COLS)
fs = (lf.with_columns(pl.col("tick").min().over(["match_key", "round"]).alias("t0"))
      .filter(pl.col("tick") == pl.col("t0"))
      .drop("t0")
      .collect())
rd = (pl.scan_parquet(ROUNDS).select(["match_key", "round", "ct_win"]).collect()
      .with_columns(pl.col("round").cast(pl.Int32)))

keys = sorted(fs["match_key"].unique().to_list())
M = len(keys)
midx_df = pl.DataFrame({"match_key": keys, "midx": list(range(M))}).with_columns(pl.col("midx").cast(pl.Int64))
fs = fs.join(midx_df, on="match_key", how="left")
rd = rd.join(midx_df, on="match_key", how="inner")

log("=== A1 armor statistics (1 Hz freeze-end sample) ===")
log(f"first-snapshot player rows: {fs.height}; rounds: {fs.group_by(['match_key', 'round']).agg(pl.len()).height}; matches: {M}")
log("split (matches):", fs.group_by("split").agg(pl.col("match_key").n_unique().alias("m")).sort("split").rows())

# ------------------------------------------------------------------ QA 0: weapon code -> name mapping (weapon_fire)
import glob as _glob
import os as _os
wf_folder = _glob.glob(_os.path.join(TOME, "basic_weapon_fire.*"))[0]
wf_files = sorted(_glob.glob(_os.path.join(wf_folder, "dataframe_*")))
wf_pairs = (pl.scan_parquet(wf_files).select(["player_weapon_code", "weapon_name"])
            .group_by(["player_weapon_code", "weapon_name"]).agg(pl.len().alias("rows")).collect())
wf_code = (wf_pairs.sort(["player_weapon_code", "rows"], descending=[False, True])
           .group_by("player_weapon_code", maintain_order=True)
           .agg(pl.col("rows").sum().alias("code_rows"), pl.col("weapon_name").first().alias("top_name"),
                pl.col("rows").first().alias("top_rows"), pl.col("weapon_name").n_unique().alias("n_names"))
           .with_columns((pl.col("top_rows") / pl.col("code_rows")).round(4).alias("top_share")))
log("\n## QA0. Weapon code -> name (weapon_fire, codes used in this analysis)")
log(f"weapon_fire total rows: {int(wf_pairs['rows'].sum())}; codes with >1 name: {wf_code.filter(pl.col('n_names') > 1).height}")
used_codes = RIFLE + SNIPER + OTHER_PRIMARY + PISTOLS + [0]
qa = (wf_code.filter(pl.col("player_weapon_code").is_in(used_codes))
      .sort("code_rows", descending=True))
for row in qa.iter_rows(named=True):
    log(f"  code {row['player_weapon_code']:4d}  {row['top_name']:28s} rows={row['code_rows']:9d} "
        f"top_share={row['top_share']:.4f} n_names={row['n_names']}")
missing = sorted(set(used_codes) - set(qa["player_weapon_code"].to_list()))
log(f"  codes used but absent from weapon_fire: {missing}")

# ------------------------------------------------------------------ player-level features
pl_df = fs.with_columns(
    pl.lit(1.0).alias("one"),
    (pl.col("armor") == 0).cast(pl.Float64).alias("z0"),
    ((pl.col("armor") >= 1) & (pl.col("armor") <= 50)).cast(pl.Float64).alias("z1"),
    (pl.col("armor") >= 51).cast(pl.Float64).alias("z2"),
    pl.col("armor").cast(pl.Float64).alias("arm"),
    pl.col("has_helmet").cast(pl.Float64).alias("helm"),
    pl.col("has_defuser").cast(pl.Float64).alias("defu"),
    (pl.col("armor") > 0).cast(pl.Float64).alias("armpos"),
    pl.col("equipment_value_calc").cast(pl.Float64).alias("eqv"),
    pl.when(pl.col("round") == 1).then(pl.lit("P1"))
      .when(pl.col("round") == 13).then(pl.lit("P13"))
      .when(pl.col("round").is_between(2, 12)).then(pl.lit("R2_12"))
      .when(pl.col("round").is_between(14, 24)).then(pl.lit("R14_24"))
      .otherwise(pl.lit("OT")).alias("band"),
    (pl.col("round").is_between(2, 12) | pl.col("round").is_between(14, 24)).alias("is_np"),
    pl.when(pl.col("inv_primary").is_in(RIFLE)).then(pl.lit("rifle"))
      .when(pl.col("inv_primary").is_in(SNIPER)).then(pl.lit("sniper"))
      .when(pl.col("inv_primary").is_in(OTHER_PRIMARY)).then(pl.lit("smg_shotgun_lmg"))
      .when((pl.col("inv_primary") == 0) & pl.col("inv_secondary").is_in(PISTOLS)).then(pl.lit("pistol_only"))
      .otherwise(pl.lit("no_primary_no_pistol")).alias("wclass_inv"),
    pl.when(pl.col("weapon_code").is_in(RIFLE)).then(pl.lit("rifle"))
      .when(pl.col("weapon_code").is_in(SNIPER)).then(pl.lit("sniper"))
      .when(pl.col("weapon_code").is_in(PISTOLS)).then(pl.lit("pistol"))
      .otherwise(pl.lit("other_knife_nade")).alias("wclass_active"),
)
pl_df = pl_df.with_columns(
    (pl.col("is_np")).alias("np"),
)

IND_PLAYER = ["one", "z0", "z1", "z2", "arm", "helm", "defu", "armpos", "eqv"]

# ------------------------------------------------------------------ QA 1: does equipment_value_calc include armor/helmet?
top_prim = (fs.group_by("inv_primary").agg(pl.len().alias("n")).sort("n", descending=True).head(12)["inv_primary"].to_list())
top_sec = (fs.group_by("inv_secondary").agg(pl.len().alias("n")).sort("n", descending=True).head(6)["inv_secondary"].to_list())
qa_cols = [np.ones(fs.height), (fs["armor"] > 0).cast(pl.Float64).to_numpy(),
           fs["has_helmet"].cast(pl.Float64).to_numpy(), fs["has_defuser"].cast(pl.Float64).to_numpy(),
           (fs["team_code"] == 3).cast(pl.Float64).to_numpy()]
qa_names = ["const", "armor>0", "helmet", "defuser", "is_CT"]
for c in ["inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy", "inv_c4"]:
    qa_cols.append(fs[c].cast(pl.Float64).fill_null(0).to_numpy())
    qa_names.append(c)
for v in top_prim:
    qa_cols.append((fs["inv_primary"] == v).cast(pl.Float64).to_numpy())
    qa_names.append(f"prim_{v}")
for v in top_sec:
    qa_cols.append((fs["inv_secondary"] == v).cast(pl.Float64).to_numpy())
    qa_names.append(f"sec_{v}")
Xq = np.column_stack(qa_cols)
yq = fs["equipment_value_calc"].cast(pl.Float64).to_numpy()
bq = ols(Xq, yq)
rq = yq - mv(Xq, bq)
log("\n## QA1. OLS equipment_value_calc ~ armor>0 + helmet + defuser + grenades + primary/secondary dummies (all freeze-end players)")
log(f"  R2={1 - rq.var() / yq.var():.4f}, n={len(yq)}")
for nm in ["armor>0", "helmet", "defuser", "is_CT", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_c4"]:
    log(f"  {nm:16s} coef={bq[qa_names.index(nm)]:9.2f}")
# armor points effect within armored players: regress equipment on armor value (armor>0 rows only)
pos = fs["armor"] > 0
Xa = np.column_stack([np.ones(pos.sum()), fs.filter(pos)["armor"].cast(pl.Float64).to_numpy(),
                      fs.filter(pos)["has_helmet"].cast(pl.Float64).to_numpy()] + [Xq[pos.to_numpy()][:, qa_names.index(f"prim_{v}")] for v in top_prim] + [Xq[pos.to_numpy()][:, qa_names.index(f"sec_{v}")] for v in top_sec] + [Xq[pos.to_numpy()][:, qa_names.index(c)] for c in ["inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy", "inv_c4"]])
ya = fs.filter(pos)["equipment_value_calc"].cast(pl.Float64).to_numpy()
ba = ols(Xa, ya)
log(f"  within armored players: equip per armor point = {ba[1]:.2f}, helmet coef = {ba[2]:.2f}  (n={len(ya)})")


def msum(df, mask, cols):
    """Per-match sums (M x k) for rows satisfying mask, full set of matches."""
    agg = (df.filter(mask).group_by("midx").agg([pl.col(c).sum().alias(c) for c in cols]))
    full = midx_df.select("midx").join(agg, on="midx", how="left").fill_null(0).sort("midx")
    return full.select(cols).to_numpy().astype(np.float64)


# bootstrap weights: B x M counts from match resampling, seed 0
rng = np.random.default_rng(SEED)
draws = rng.integers(0, M, size=(B, M))
W = np.zeros((B, M))
np.add.at(W, (np.repeat(np.arange(B), M), draws.ravel()), 1.0)
assert np.allclose(W.sum(1), M)


def ratio_stats(Mx, num, den):
    """Point, lo, hi for sum(num)/sum(den) over matches, bootstrap on match resamples."""
    p = Mx[:, num].sum() / Mx[:, den].sum()
    Sb = np.einsum("bm,mk->bk", W, Mx)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = Sb[:, num] / Sb[:, den]
    lo, hi = np.nanpercentile(rb, [2.5, 97.5])
    return p, lo, hi


def ratio_diff_stats(Mx1, Mx2, num, den):
    p1 = Mx1[:, num].sum() / Mx1[:, den].sum()
    p2 = Mx2[:, num].sum() / Mx2[:, den].sum()
    S1 = np.einsum("bm,mk->bk", W, Mx1)
    S2 = np.einsum("bm,mk->bk", W, Mx2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = S1[:, num] / S1[:, den] - S2[:, num] / S2[:, den]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return p1 - p2, lo, hi


def subset_mask(band=None, side=None, np_only=False):
    m = pl.lit(True)
    if band == "NP":
        m = m & pl.col("is_np")
    elif band is not None:
        m = m & (pl.col("band") == band)
    if side == "CT":
        m = m & (pl.col("team_code") == 3)
    elif side == "T":
        m = m & (pl.col("team_code") == 2)
    return m


# ------------------------------------------------------------------ Table 1: armor distribution by band and side
log("\n## Table 1. Armor at freeze end, per player (share with armor 0 / 1-50 / 51-100; mean armor; helmet; defuser)")
log("95% CI from match-cluster bootstrap (B=1000, seed 0). Defuser share is among CT players only.")
log(f"{'band':10s} {'side':4s} {'n':>6s} {'armor=0':>22s} {'1-50':>22s} {'51-100':>22s} {'mean armor':>22s} {'helmet':>22s} {'defuser':>22s}")
BANDS = [("P1", "Pistol R1"), ("R2_12", "R2-12"), ("P13", "Pistol R13"), ("R14_24", "R14-24"),
         ("NP", "NP R2-24 pooled"), ("OT", "OT R25+")]
T1_ROWS = {}
for band, label in BANDS:
    for side in ["ALL", "CT", "T"]:
        mask = subset_mask(band=band, side=side)
        Mx = msum(pl_df, mask, IND_PLAYER)
        n = Mx[:, 0].sum()
        if n == 0:
            continue
        p0, l0, h0 = ratio_stats(Mx, 1, 0)
        p1, l1, h1 = ratio_stats(Mx, 2, 0)
        p2, l2, h2 = ratio_stats(Mx, 3, 0)
        pm, lm, hm = ratio_stats(Mx, 4, 0)
        ph, lh, hh = ratio_stats(Mx, 5, 0)
        if side in ("CT",):
            pd, ld, hd = ratio_stats(Mx, 6, 0)
            dtxt = fmt_p(pd, ld, hd)
        else:
            dtxt = "-"
        T1_ROWS[(band, side)] = dict(n=n, z0=(p0, l0, h0), z1=(p1, l1, h1), z2=(p2, l2, h2), mean=(pm, lm, hm),
                                     helm=(ph, lh, hh))
        log(f"{label:10s} {side:4s} {int(n):6d} {fmt_p(p0, l0, h0):>22s} {fmt_p(p1, l1, h1):>22s} "
            f"{fmt_p(p2, l2, h2):>22s} {fmt_x(pm, lm, hm, 1):>22s} {fmt_p(ph, lh, hh):>22s} {dtxt:>22s}")

# ------------------------------------------------------------------ Table 2: per-round profile, halftime reset
log("\n## Table 2. Per-round profile at freeze end, all players (armor=0 share; mean armor; helmet share)")
log(f"{'round':>5s} {'n':>5s} {'armor=0':>22s} {'mean armor':>22s} {'helmet':>22s}")
R_IND = ["one", "z0", "arm", "helm"]
round_mats = {}
for r in range(1, 25):
    mask = pl.col("round") == r
    Mx = msum(pl_df, mask, R_IND)
    round_mats[r] = Mx
    n = Mx[:, 0].sum()
    if n == 0:
        continue
    p0, l0, h0 = ratio_stats(Mx, 1, 0)
    pm, lm, hm = ratio_stats(Mx, 2, 0)
    ph, lh, hh = ratio_stats(Mx, 3, 0)
    log(f"{r:5d} {int(n):5d} {fmt_p(p0, l0, h0):>22s} {fmt_x(pm, lm, hm, 1):>22s} {fmt_p(ph, lh, hh):>22s}")

log("\nHalftime reset contrasts (paired bootstrap; difference of rates):")
for ra, rb_, label in [(13, 12, "R13 minus R12"), (13, 1, "R13 minus R1 (both pistol)"), (14, 12, "R14 minus R12"),
                       (12, 11, "R12 minus R11 (no reset)"), (14, 13, "R14 minus R13")]:
    Ma, Mb = round_mats[ra], round_mats[rb_]
    d0, l0, h0 = ratio_diff_stats(Ma, Mb, 1, 0)
    dm, lm, hm = ratio_diff_stats(Ma, Mb, 2, 0)
    dh, lh, hh = ratio_diff_stats(Ma, Mb, 3, 0)
    log(f"  {label:28s} armor=0 share {fmt_s(100 * d0, 100 * l0, 100 * h0, 1):>30s} pp | "
        f"mean armor {fmt_s(dm, lm, hm, 1):>26s} | helmet {fmt_s(100 * dh, 100 * lh, 100 * hh, 1):>30s} pp")

# CT vs T split at rounds 12 and 13
log("\nSide split, armor=0 share and mean armor at R12 and R13:")
for r in [12, 13]:
    for side in ["CT", "T"]:
        Mx = msum(pl_df, (pl.col("round") == r) & (pl.col("team_code") == (3 if side == "CT" else 2)), R_IND)
        p0, l0, h0 = ratio_stats(Mx, 1, 0)
        pm, lm, hm = ratio_stats(Mx, 2, 0)
        log(f"  R{r:2d} {side}: armor=0 {fmt_p(p0, l0, h0)} | mean armor {fmt_x(pm, lm, hm, 1)}")

# ------------------------------------------------------------------ Table 3: helmet and defuser shares by side
log("\n## Table 3. Helmet and defuser share by side (non-pistol R2-24 pooled, and pistol rounds)")
for band, label in [("NP", "NP R2-24 pooled"), ("P1", "Pistol R1"), ("P13", "Pistol R13")]:
    for side in ["CT", "T"]:
        mask = subset_mask(band=band, side=side)
        Mx = msum(pl_df, mask, ["one", "helm", "defu", "armpos"])
        ph, lh, hh = ratio_stats(Mx, 1, 0)
        pa, la, ha = ratio_stats(Mx, 3, 0)
        if side == "CT":
            pd, ld, hd = ratio_stats(Mx, 2, 0)
            dtxt = fmt_p(pd, ld, hd)
        else:
            dtxt = "- (T has no defuser)"
        log(f"  {label:16s} {side}: n={int(Mx[:, 0].sum()):5d} helmet {fmt_p(ph, lh, hh)} | "
            f"armor>0 {fmt_p(pa, la, ha)} | defuser {dtxt}")

# ------------------------------------------------------------------ round-level dataset (NP rounds)
rnd = (pl_df.group_by(["midx", "match_key", "round", "band", "is_np"])
       .agg(
           pl.col("team_code").filter(pl.col("team_code") == 3).count().alias("n_ct"),
           pl.col("team_code").filter(pl.col("team_code") == 2).count().alias("n_t"),
           pl.col("equipment_value_calc").filter(pl.col("team_code") == 3).sum().alias("ct_eq"),
           pl.col("equipment_value_calc").filter(pl.col("team_code") == 2).sum().alias("t_eq"),
           pl.col("armor").filter(pl.col("team_code") == 3).sum().cast(pl.Float64).alias("ct_arm"),
           pl.col("armor").filter(pl.col("team_code") == 2).sum().cast(pl.Float64).alias("t_arm"),
           (pl.col("armor").filter(pl.col("team_code") == 3) > 0).sum().cast(pl.Float64).alias("ct_narm"),
           (pl.col("armor").filter(pl.col("team_code") == 2) > 0).sum().cast(pl.Float64).alias("t_narm"),
           pl.col("has_helmet").filter(pl.col("team_code") == 3).sum().cast(pl.Float64).alias("ct_helm"),
           pl.col("has_helmet").filter(pl.col("team_code") == 2).sum().cast(pl.Float64).alias("t_helm"),
           pl.col("has_defuser").filter(pl.col("team_code") == 3).sum().cast(pl.Float64).alias("ct_def"),
           pl.col("has_defuser").filter(pl.col("team_code") == 2).sum().cast(pl.Float64).alias("t_def"),
       )
       .join(rd.select(["match_key", "round", "ct_win"]), on=["match_key", "round"], how="left"))
log(f"\nround-level rows (all bands): {rnd.height}; with ct_win: {rnd.filter(pl.col('ct_win').is_not_null()).height}")
rnd_np = rnd.filter(pl.col("is_np") & pl.col("ct_win").is_not_null())
n_before = rnd_np.height
rnd_np = rnd_np.filter((pl.col("n_ct") == 5) & (pl.col("n_t") == 5))
log(f"NP rounds with ct_win: {n_before}; kept with 5v5 freeze-end snapshot: {rnd_np.height}; "
    f"matches: {rnd_np['match_key'].n_unique()}")
rnd_np = rnd_np.with_columns(
    ((pl.col("ct_eq") - pl.col("t_eq")) / 1000.0).alias("eq_diff"),
    ((pl.col("ct_arm") - pl.col("t_arm")) / 100.0).alias("arm_diff"),
    (pl.col("ct_narm") - pl.col("t_narm")).alias("narm_diff"),
    (pl.col("ct_helm") - pl.col("t_helm")).alias("helm_diff"),
    (100.0 * (pl.col("ct_arm") / pl.col("ct_eq") - pl.col("t_arm") / pl.col("t_eq"))).alias("ratio_diff_pp"),
    pl.col("ct_win").cast(pl.Float64).alias("y"),
    # equipment_value_calc already contains kevlar/helmet/defuser costs (QA1); strip them at list prices
    # (kevlar 650, helmet 350, defuser 400) to get equipment excluding armor items
    (pl.col("ct_eq") - 650.0 * pl.col("ct_narm") - 350.0 * pl.col("ct_helm") - 400.0 * pl.col("ct_def")).alias("ct_eqx"),
    (pl.col("t_eq") - 650.0 * pl.col("t_narm") - 350.0 * pl.col("t_helm") - 400.0 * pl.col("t_def")).alias("t_eqx"),
)
rnd_np = rnd_np.with_columns(((pl.col("ct_eqx") - pl.col("t_eqx")) / 1000.0).alias("eqx_diff"))
RND_COLS = ["ct_eq", "t_eq", "ct_arm", "t_arm", "ct_narm", "t_narm", "ct_helm", "t_helm"]

# round-level descriptive: side totals (NP rounds)
log("\n## Table 4. Round-level side totals at freeze end, NP rounds (mean per round)")
side_desc = rnd_np.select([
    pl.col("ct_eq").mean().alias("ct_eq"), pl.col("t_eq").mean().alias("t_eq"),
    pl.col("ct_arm").mean().alias("ct_arm"), pl.col("t_arm").mean().alias("t_arm"),
    pl.col("ct_narm").mean().alias("ct_narm"), pl.col("t_narm").mean().alias("t_narm"),
    pl.col("ct_helm").mean().alias("ct_helm"), pl.col("t_helm").mean().alias("t_helm"),
    pl.col("ct_win").mean().alias("ct_win_rate"),
    pl.col("ct_eqx").mean().alias("ct_eqx"), pl.col("t_eqx").mean().alias("t_eqx"),
]).row(0, named=True)
for k, v in side_desc.items():
    log(f"  {k:12s} {v:10.3f}")

# ------------------------------------------------------------------ logistic models
def fit_logit(X, y, w, iters=100, tol=1e-10):
    beta = np.zeros(X.shape[1])
    p0 = float(np.clip((w * y).sum() / w.sum(), 1e-6, 1 - 1e-6))
    beta[0] = np.log(p0 / (1 - p0))
    for _ in range(iters):
        eta = mv(X, beta)
        p = 1.0 / (1.0 + np.exp(-eta))
        g = np.einsum("ni,n->i", X, w * (y - p))
        H = np.einsum("ni,n,nj->ij", X, w * p * (1 - p), X) + 1e-9 * np.eye(X.shape[1])
        step = solve_small(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def wll(X, y, w, beta):
    """Mean negative log-likelihood (log loss, nats per round) under weights w."""
    eta = mv(X, beta)
    ll = y * eta - np.logaddexp(0.0, eta)
    return float(-(w * ll).sum() / w.sum())


def ll_vec(X, y, beta):
    eta = mv(X, beta)
    return y * eta - np.logaddexp(0.0, eta)


y_all = rnd_np["y"].to_numpy()
mid_all = rnd_np["midx"].to_numpy()
eq = rnd_np["eq_diff"].to_numpy()
arm = rnd_np["arm_diff"].to_numpy()
narm = rnd_np["narm_diff"].to_numpy().astype(float)
helm = rnd_np["helm_diff"].to_numpy().astype(float)
ratio = rnd_np["ratio_diff_pp"].to_numpy()
one = np.ones_like(eq)
eqx = rnd_np["eqx_diff"].to_numpy()
COLS_ = {"one": one, "eq": eq, "eqx": eqx, "arm": arm, "narm": narm, "helm": helm, "ratio": ratio}
BASE_A = "Ma equipment (as recorded)"
BASE_X = "Mx equipment excl. armor items"
# name, design columns, coefficient index of the added term (None for bases), base model for dLL
MODEL_SPECS = [
    ("M0 intercept only", ["one"], None, BASE_A),
    (BASE_A, ["one", "eq"], None, None),
    ("Mb equip + armor-sum diff", ["one", "eq", "arm"], 2, BASE_A),
    ("Mb2 equip + armored-count diff", ["one", "eq", "narm"], 2, BASE_A),
    ("Mc equip + helmet-count diff", ["one", "eq", "helm"], 2, BASE_A),
    ("Md equip + armor/equip ratio diff", ["one", "eq", "ratio"], 2, BASE_A),
    ("Me equip + armor + helmet", ["one", "eq", "arm", "helm"], 2, BASE_A),
    ("Mf armor-sum diff only", ["one", "arm"], 1, BASE_A),
    (BASE_X, ["one", "eqx"], None, None),
    ("Mxb eqx + armor-sum diff", ["one", "eqx", "arm"], 2, BASE_X),
    ("Mxb2 eqx + armored-count diff", ["one", "eqx", "narm"], 2, BASE_X),
    ("Mxc eqx + helmet-count diff", ["one", "eqx", "helm"], 2, BASE_X),
]
MODELS = {}
for _name, _cols, _k, _base in MODEL_SPECS:
    MODELS[_name] = (_cols, np.column_stack([COLS_[c] for c in _cols]), _k, _base)

log("\n## Table 5. Logistic regression of CT win (NP rounds), point estimates (in-sample)")
log("LL = mean log loss in nats per round; dLL = LL(Ma) - LL(model) in milli-nats per round (positive = model better)")
point = {}
for name, (cols_, X, k, base) in MODELS.items():
    beta = fit_logit(X, y_all, np.ones_like(y_all))
    point[name] = (beta, wll(X, y_all, np.ones_like(y_all), beta))
for name, (cols_, X, k, base) in MODELS.items():
    beta, ll = point[name]
    dll = (point[base][1] - ll) * 1000 if base is not None else 0.0
    coef = f"coef on added term = {beta[k]:+.4f}" if k is not None else ""
    log(f"  {name:36s} LL={ll:.5f}  dLL vs {('(' + base + ')') if base else '-'}={dll:+.3f} milli-nats  {coef}")

# bootstrap contrasts: every model with a base is refit on each match-cluster resample
log("\nBootstrap (B=1000, match-cluster, seed 0): refit every model on resampled matches; dLL and coef intervals")
rowmatch = mid_all
contrast_names = [n for n, (c_, X_, k_, base_) in MODELS.items() if base_ is not None]
boot_dll = {n: np.zeros(B) for n in contrast_names}
boot_coef = {n: np.zeros(B) for n in contrast_names if MODELS[n][2] is not None}
boot_base_coef = {BASE_A: np.zeros(B), BASE_X: np.zeros(B)}
boot_ll_base = {BASE_A: np.zeros(B), BASE_X: np.zeros(B)}
for b in range(B):
    w = W[b][rowmatch]
    ll_map = {}
    beta_map = {}
    for name, (cols_, X, k, base) in MODELS.items():
        beta_map[name] = fit_logit(X, y_all, w)
        ll_map[name] = wll(X, y_all, w, beta_map[name])
    for n in contrast_names:
        base = MODELS[n][3]
        boot_dll[n][b] = (ll_map[base] - ll_map[n]) * 1000
        if MODELS[n][2] is not None:
            boot_coef[n][b] = beta_map[n][MODELS[n][2]]
    for bn in (BASE_A, BASE_X):
        boot_base_coef[bn][b] = beta_map[bn][1]
        boot_ll_base[bn][b] = ll_map[bn]

log(f"{'contrast (vs base)':78s} {'dLL milli-nats/round [95% CI]':>36s} {'coef [95% CI]':>34s}")
for n in contrast_names:
    base = MODELS[n][3]
    dpt = (point[base][1] - point[n][1]) * 1000
    dlo, dhi = np.percentile(boot_dll[n], [2.5, 97.5])
    k = MODELS[n][2]
    if k is not None:
        cpt = point[n][0][k]
        clo, chi = np.percentile(boot_coef[n], [2.5, 97.5])
        ctxt = f"{cpt:+.4f} [{clo:+.4f}, {chi:+.4f}]"
    else:
        ctxt = "-"
    log(f"  {n + ' vs ' + base:76s} {dpt:+8.3f} [{dlo:+8.3f}, {dhi:+8.3f}]   {ctxt:>34s}")
for bn in (BASE_A, BASE_X):
    cpt = point[bn][0][1]
    clo, chi = np.percentile(boot_base_coef[bn], [2.5, 97.5])
    log(f"  coefficient on equipment diff in {bn} (per $1000): {cpt:+.4f} [{clo:+.4f}, {chi:+.4f}]")
    lo_, hi_ = np.percentile(boot_ll_base[bn], [2.5, 97.5])
    log(f"  {bn} LL bootstrap 95% range: [{lo_:.5f}, {hi_:.5f}]")

# Grouped out-of-fold check (5 folds by match, seed 0), point estimate only
rng_cv = np.random.default_rng(SEED)
perm = rng_cv.permutation(M)
fold_of_match = np.empty(M, dtype=int)
fold_of_match[perm] = np.arange(M) % 5
fold_row = fold_of_match[rowmatch]
log("\n5-fold match-grouped out-of-fold log loss (point estimate, no bootstrap):")
oof = {}
for name, (cols_, X, k, base) in MODELS.items():
    tot = 0.0
    for f in range(5):
        tr = fold_row != f
        te = fold_row == f
        beta = fit_logit(X[tr], y_all[tr], np.ones(tr.sum()))
        tot += float(-ll_vec(X[te], y_all[te], beta).sum())
    oof[name] = tot / len(y_all)
for name, (cols_, X, k, base) in MODELS.items():
    dv = (oof[base] - oof[name]) * 1000 if base is not None else 0.0
    log(f"  {name:36s} OOF LL={oof[name]:.5f}  dLL vs {base if base else '-'}={dv:+.3f} milli-nats")

# ------------------------------------------------------------------ Table 6: CT win rate by armored-count difference
log("\n## Table 6. CT win rate by armored-player difference (CT armored minus T armored), NP rounds")
bins = {"<=-3": (-99, -3), "-2": (-2, -2), "-1": (-1, -1), "0": (0, 0), "+1": (1, 1), "+2": (2, 2), ">=+3": (3, 99)}
rwin = rnd_np.with_columns(pl.col("ct_win").cast(pl.Float64).alias("cw"),
                           pl.lit(1.0).alias("one"))
for lab, (lo_, hi_) in bins.items():
    mask = (pl.col("narm_diff") >= lo_) & (pl.col("narm_diff") <= hi_)
    Mx = msum(rwin, mask, ["one", "cw"])
    n = Mx[:, 0].sum()
    if n == 0:
        continue
    p, l, h = ratio_stats(Mx, 1, 0)
    log(f"  narm_diff {lab:>5s}: rounds={int(n):5d}  CT win {fmt_p(p, l, h)}")

# ------------------------------------------------------------------ Table 7: armor by weapon class
log("\n## Table 6b. CT win rate by equipment difference (CT minus T equipment_value_calc, $1000), NP rounds")
for lab, (lo_, hi_) in {"<-5": (-1e9, -5.0), "-5..-2.5": (-5.0, -2.5), "-2.5..0": (-2.5, 0.0),
                        "0..2.5": (0.0, 2.5), "2.5..5": (2.5, 5.0), ">=5": (5.0, 1e9)}.items():
    mask = (pl.col("eq_diff") >= lo_) & (pl.col("eq_diff") < hi_)
    Mx = msum(rwin, mask, ["one", "cw"])
    n = Mx[:, 0].sum()
    if n == 0:
        continue
    p, l, h = ratio_stats(Mx, 1, 0)
    log(f"  eq_diff {lab:>9s}: rounds={int(n):5d}  CT win {fmt_p(p, l, h)}")

log("\n## Table 7. Armor by weapon class at freeze end (player level)")
log("Class by inventory (inv_primary; pistol-only = no primary and a pistol secondary)")
log(f"{'class':22s} {'band':6s} {'n':>6s} {'armor=0':>22s} {'1-50':>22s} {'51-100':>22s} {'mean armor':>22s} {'helmet':>22s}")
for band_lab, mask_base in [("NP", pl.col("is_np")), ("P1+P13", pl.col("round").is_in([1, 13]))]:
    for cls in ["rifle", "sniper", "smg_shotgun_lmg", "pistol_only", "no_primary_no_pistol"]:
        mask = mask_base & (pl.col("wclass_inv") == cls)
        Mx = msum(pl_df, mask, IND_PLAYER)
        n = Mx[:, 0].sum()
        if n == 0:
            continue
        p0, l0, h0 = ratio_stats(Mx, 1, 0)
        p1, l1, h1 = ratio_stats(Mx, 2, 0)
        p2, l2, h2 = ratio_stats(Mx, 3, 0)
        pm, lm, hm = ratio_stats(Mx, 4, 0)
        ph, lh, hh = ratio_stats(Mx, 5, 0)
        log(f"{cls:22s} {band_lab:6s} {int(n):6d} {fmt_p(p0, l0, h0):>22s} {fmt_p(p1, l1, h1):>22s} "
            f"{fmt_p(p2, l2, h2):>22s} {fmt_x(pm, lm, hm, 1):>22s} {fmt_p(ph, lh, hh):>22s}")

log("\nSensitivity: class by active weapon at freeze end (weapon_code), NP rounds")
for cls in ["rifle", "sniper", "pistol", "other_knife_nade"]:
    mask = pl.col("is_np") & (pl.col("wclass_active") == cls)
    Mx = msum(pl_df, mask, IND_PLAYER)
    n = Mx[:, 0].sum()
    if n == 0:
        continue
    p0, l0, h0 = ratio_stats(Mx, 1, 0)
    pm, lm, hm = ratio_stats(Mx, 4, 0)
    log(f"  {cls:18s} n={int(n):6d} armor=0 {fmt_p(p0, l0, h0)} | mean armor {fmt_x(pm, lm, hm, 1)}")

# rifle vs pistol-only contrast (NP): armor=0 share difference
Mr_ = msum(pl_df, pl.col("is_np") & (pl.col("wclass_inv") == "rifle"), IND_PLAYER)
Mp_ = msum(pl_df, pl.col("is_np") & (pl.col("wclass_inv") == "pistol_only"), IND_PLAYER)
d0, l0, h0 = ratio_diff_stats(Mr_, Mp_, 1, 0)
dz, lz, hz = ratio_diff_stats(Mr_, Mp_, 4, 0)
log(f"\nContrast NP rifle minus pistol-only: armor=0 share {fmt_s(100 * d0, 100 * l0, 100 * h0, 1)} pp; "
    f"mean armor {fmt_s(dz, lz, hz, 1)}")

log(f"\nDone in {time.time() - T0:.1f}s")
