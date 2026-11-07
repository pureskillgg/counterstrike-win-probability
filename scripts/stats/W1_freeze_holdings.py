"""W1 parts 3 and 4 (plus pistol and armor context): freeze-end holdings and round win rates.

Source: data/state_sample/players_1hz.parquet (150 matches, train and test splits pooled),
rounds.parquet (ct_win), round_buy_types.parquet (is_pistol cross-check), and round_end from the
2026 tome (winner cross-check). Freeze-end = every player row at the minimum tick of each
(match_key, round). Pistol rounds = round 1 and round 13 (checked against round_buy_types).
Classes come from inv_primary (CS item definition ids); names are verified against the
modal weapon_fire name for each code.

Uncertainty: match-cluster percentile bootstrap (resample matches with replacement),
1000 resamples, seed 0. Rounds are not resampled.
No BLAS calls (numpy matrix products crash in this Windows build), so the logistic fit
is a hand-written IRLS with a small Gaussian-elimination solver.
"""
from pathlib import Path
import math

import numpy as np
import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
SAMPLE = ROOT / "data" / "state_sample"
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
N_BOOT = 1000
SEED = 0
PISTOL_ROUNDS = [1, 13]

# inv_primary code -> class; NAME_OF is the expected weapon_fire name (verified below)
CLASS_OF = {
    7: "RIFLE", 16: "RIFLE", 60: "RIFLE", 13: "RIFLE", 10: "RIFLE", 8: "RIFLE", 39: "RIFLE",
    34: "SMG", 17: "SMG", 33: "SMG", 19: "SMG", 23: "SMG", 24: "SMG", 26: "SMG",
    9: "SNIPER", 40: "SNIPER",
    38: "HEAVY", 11: "HEAVY", 14: "HEAVY", 28: "HEAVY", 25: "HEAVY", 27: "HEAVY", 35: "HEAVY", 29: "HEAVY",
    0: "NONE",
}
NAME_OF = {
    7: "ak47", 16: "m4a1", 60: "m4a1_silencer", 13: "galilar", 10: "famas", 8: "aug", 39: "sg556",
    34: "mp9", 17: "mac10", 33: "mp7", 19: "p90", 23: "mp5sd", 24: "ump45", 26: "bizon",
    9: "awp", 40: "ssg08",
    38: "scar20", 11: "g3sg1", 14: "m249", 28: "negev", 25: "xm1014", 27: "mag7", 35: "nova", 29: "sawedoff",
    0: "(no primary)",
}
CODES_PRIMARY = [c for c in CLASS_OF if c != 0]
CLASS_LABEL = {
    "RIFLE": "rifle (AK-47, M4A4, M4A1-S, Galil AR, FAMAS, AUG, SG 553)",
    "SMG": "SMG (MP9, MAC-10, MP7, P90, MP5-SD, UMP-45, PP-Bizon)",
    "SNIPER": "sniper (AWP, SSG 08)",
    "HEAVY": "other heavy (G3SG1, SCAR-20, M249, Negev, shotguns)",
    "NONE": "pistol-only (no primary)",
}


def files(channel):
    fs = sorted(TOME.glob(f"basic_{channel}.*/dataframe_*"))
    if not fs:
        raise SystemExit(f"no files for {channel}")
    return [str(f) for f in fs]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def pc(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def boot_matrix(rng, n_match):
    idx = rng.integers(0, n_match, size=(N_BOOT, n_match))
    C = np.empty((N_BOOT, n_match), dtype=np.float64)
    for b in range(N_BOOT):
        C[b] = np.bincount(idx[b], minlength=n_match)
    return C


def boot_sum(C, A):
    out = np.empty((C.shape[0], A.shape[1]), dtype=np.float64)
    for b in range(C.shape[0]):
        out[b] = (C[b][:, None] * A).sum(axis=0)
    return out


def group_matrix(df, key_cols, num_cols, M):
    """Per-match sums of num_cols and row counts for each key tuple.
    Returns keys (list of tuples), NUM (M, K, J), DEN (M, K)."""
    g = df.group_by(["mi"] + key_cols).agg(
        [pl.col(c).cast(pl.Float64).sum().alias(c) for c in num_cols] + [pl.len().alias("_den")]
    )
    keys = sorted(set(tuple(r[k] for k in key_cols) for r in g.iter_rows(named=True)))
    kidx = {k: i for i, k in enumerate(keys)}
    K, J = len(keys), len(num_cols)
    NUM = np.zeros((M, K, J))
    DEN = np.zeros((M, K))
    for r in g.iter_rows(named=True):
        k = kidx[tuple(r[c] for c in key_cols)]
        NUM[r["mi"], k, :] = [r[c] for c in num_cols]
        DEN[r["mi"], k] = r["_den"]
    return keys, NUM, DEN


def ratio_boot(C, NUM, DEN):
    M, K, J = NUM.shape
    pt = NUM.sum(axis=0) / DEN.sum(axis=0)[:, None]
    bn = boot_sum(C, NUM.reshape(M, K * J)).reshape(-1, K, J)
    bd = boot_sum(C, DEN)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bn / bd[:, :, None]
    lo, hi = np.nanpercentile(br, [2.5, 97.5], axis=0)
    return pt, lo, hi


def solve_small(A, b):
    n = len(b)
    Mx = [[float(v) for v in A[i]] + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(Mx[r][c]))
        Mx[c], Mx[p] = Mx[p], Mx[c]
        pv = Mx[c][c]
        for r in range(n):
            if r != c:
                f = Mx[r][c] / pv
                for k in range(c, n + 1):
                    Mx[r][k] -= f * Mx[c][k]
    return [Mx[i][n] / Mx[i][i] for i in range(n)]


def inv_small(A):
    n = len(A)
    cols = [solve_small(A, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def logit_fit(X, y, w):
    k = X.shape[1]
    beta = np.zeros(k)
    for _ in range(100):
        eta = (X * beta).sum(axis=1)
        p = 1.0 / (1.0 + np.exp(-eta))
        g = (X * (w * (y - p))[:, None]).sum(axis=0)
        W = w * p * (1.0 - p)
        H = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                H[i, j] = np.sum(X[:, i] * X[:, j] * W)
        step = np.array(solve_small(H.tolist(), g.tolist()))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = (X * beta).sum(axis=1)
    p = 1.0 / (1.0 + np.exp(-eta))
    W = w * p * (1.0 - p)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            H[i, j] = np.sum(X[:, i] * X[:, j] * W)
    cov = inv_small(H.tolist())
    return beta, np.sqrt(np.array([cov[i][i] for i in range(k)]))


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def main():
    rng = np.random.default_rng(SEED)

    h = pl.read_parquet(
        SAMPLE / "players_1hz.parquet",
        columns=["match_key", "round", "tick", "seconds_since_live", "split", "player_id", "team_code",
                 "armor", "has_helmet", "equipment_value_calc", "inv_primary", "inv_secondary"],
    ).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
        pl.col("team_code").cast(pl.Int64), pl.col("inv_primary").cast(pl.Int64),
        pl.col("inv_secondary").cast(pl.Int64), pl.col("armor").cast(pl.Int64),
        pl.col("equipment_value_calc").cast(pl.Float64), pl.col("has_helmet").cast(pl.Int64),
    )
    rounds = pl.read_parquet(SAMPLE / "rounds.parquet", columns=["match_key", "round", "ct_win", "split"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("ct_win").cast(pl.Int64))
    buy = pl.read_parquet(SAMPLE / "round_buy_types.parquet",
                          columns=["match_key", "round", "is_pistol", "round_class"]).with_columns(
        pl.col("round").cast(pl.Int64))
    rend = pl.read_parquet(files("round_end"), columns=["match_key", "round", "winner_team_code"]).with_columns(
        pl.col("round").cast(pl.Int64))

    matches = sorted(h["match_key"].unique().to_list())
    M = len(matches)
    midx = pl.DataFrame({"match_key": matches, "mi": list(range(M))}, schema={"match_key": pl.String, "mi": pl.Int64})
    C = boot_matrix(rng, M)

    print("# W1 freeze-end holdings, round outcomes (1 Hz sample)\n")
    print(f"- 1 Hz rows: {h.height:,}; matches: {M}; (match, round) pairs: {h.select(['match_key','round']).unique().height:,}")
    spl = h.group_by("split").agg(pl.col("match_key").n_unique().alias("matches"), pl.len().alias("rows"))
    print("- split: " + "; ".join(f"{r['split']}: {r['matches']} matches, {r['rows']:,} rows" for r in spl.iter_rows(named=True)))

    # ---- freeze-end snapshot
    t0 = h.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t0"))
    fz = h.join(t0, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("t0"))
    fz = fz.join(midx, on="match_key", how="left")
    n_mr = fz.select(["match_key", "round"]).unique().height
    dup = fz.select(["match_key", "round", "player_id"]).is_duplicated().sum()
    ssl_vals = sorted(fz["seconds_since_live"].unique().to_list())
    print(f"- freeze-end rows: {fz.height:,} across {n_mr:,} match-rounds; duplicate (match, round, player) rows: {dup}; "
          f"seconds_since_live at freeze-end: {ssl_vals}")
    tc = (fz.group_by(["match_key", "round", "team_code"]).agg(pl.len().alias("k"))
          .group_by("k").agg(pl.len().alias("team-rounds")).sort("k"))
    print("- players per team at freeze-end: " + "; ".join(f"{r['k']} players: {r['team-rounds']:,} team-rounds"
                                                         for r in tc.iter_rows(named=True)))

    # ---- pistol rule vs buy-type flag, winner cross-check
    rr = rounds.join(buy, on=["match_key", "round"], how="left").with_columns(
        pistol=pl.col("round").is_in(PISTOL_ROUNDS))
    no_buy = rr.filter(pl.col("is_pistol").is_null()).select(["match_key", "round"]).to_dicts()
    agree = rr.filter(pl.col("is_pistol").is_not_null())
    mism = agree.filter(pl.col("pistol") != pl.col("is_pistol")).height
    print(f"- rounds: {rr.height:,}; pistol by round number (1 or 13): {rr.filter(pl.col('pistol')).height:,}; "
          f"rounds with no buy-type row: {len(no_buy)} {no_buy}; disagreements with round_buy_types.is_pistol: {mism}")
    print(f"- pistol rounds present by round: round 1 = {rr.filter(pl.col('round') == 1).height} matches, "
          f"round 13 = {rr.filter(pl.col('round') == 13).height} matches; "
          f"round 1 and 13 by buy round_class: " + str(
              buy.filter(pl.col('round').is_in(PISTOL_ROUNDS)).group_by('round_class').agg(pl.len()).to_dicts()))
    rchk = rounds.join(rend, on=["match_key", "round"], how="left")
    rend_n = rend.join(midx, on="match_key", how="inner").height
    chk = rchk.filter(pl.col("winner_team_code").is_not_null())
    wmis = chk.filter(((pl.col("ct_win") == 1) & (pl.col("winner_team_code") != 3)) |
                      ((pl.col("ct_win") == 0) & (pl.col("winner_team_code") != 2))).height
    print(f"- winner cross-check vs tome round_end: {chk.height:,} rounds matched; mismatches: {wmis}; "
          f"sample rounds without round_end match: {rr.height - chk.height}")

    # ---- class mapping and verification against weapon_fire
    cmap = pl.DataFrame(
        {"inv_primary": list(CLASS_OF.keys()), "cls": [CLASS_OF[k] for k in CLASS_OF],
         "wname": [NAME_OF[k] for k in CLASS_OF]},
        schema={"inv_primary": pl.Int64, "cls": pl.String, "wname": pl.String},
    )
    fz = fz.join(cmap, on="inv_primary", how="left")
    unmapped = fz.filter(pl.col("cls").is_null()).height
    codes_present = sorted(fz["inv_primary"].unique().to_list())
    print(f"- inv_primary codes present at freeze-end: {codes_present}; unmapped rows: {unmapped}")

    wf = pl.read_parquet(files("weapon_fire"), columns=["player_weapon_code", "weapon_name"])
    wfc = (wf.filter(pl.col("player_weapon_code").is_not_null())
           .group_by(["player_weapon_code", "weapon_name"]).agg(pl.len().alias("n")))
    tot = wfc.group_by("player_weapon_code").agg(pl.col("n").sum().alias("tot"))
    modal = (wfc.sort(["player_weapon_code", "n"], descending=[False, True])
             .group_by("player_weapon_code", maintain_order=True).first()
             .join(tot, on="player_weapon_code"))
    mrows = {int(r["player_weapon_code"]): (r["weapon_name"], r["n"], r["tot"]) for r in modal.iter_rows(named=True)}
    vrows = []
    bad = []
    for c in sorted(CLASS_OF):
        exp = NAME_OF[c]
        if c == 0:
            vrows.append((c, exp, "-", "-", "-", "not a weapon code (no primary)"))
            continue
        if c in mrows:
            nm, n_m, n_t = mrows[c]
            ok = nm == exp
            if not ok:
                bad.append(c)
            vrows.append((c, exp, nm, f"{n_m:,}", f"{100 * n_m / n_t:.2f}", "match" if ok else "MISMATCH"))
        else:
            vrows.append((c, exp, "not in weapon_fire", "-", "-", "unverified"))
    print("\n## Table 0. Code-to-class mapping verified against weapon_fire modal names\n")
    print(md_table(["inv_primary code", "expected name", "weapon_fire modal name", "rows with modal",
                    "% of code rows", "check"], vrows))
    print(f"\nCodes with a name mismatch: {bad if bad else 'none'}.")

    # ---- player-level and side-round tables
    fz = fz.with_columns(
        helm=pl.col("has_helmet"),
        armor_pos=(pl.col("armor") > 0).cast(pl.Int64),
        armor_f=pl.col("armor").cast(pl.Float64),
        pistol=pl.col("round").is_in(PISTOL_ROUNDS),
        ones=pl.lit(1, dtype=pl.Int64),
    )
    print(f"- freeze-end players with no primary AND no secondary: "
          f"{fz.filter((pl.col('inv_primary') == 0) & (pl.col('inv_secondary') == 0)).height} "
          f"of {fz.height:,}")
    # counts are cast to Int64: polars sums of booleans are UInt32 and a difference of two
    # UInt32 values wraps around (this produced a 1.96e9 mean in the first run)
    aggs = [
        pl.len().cast(pl.Int64).alias("n"),
        (pl.col("cls") == "RIFLE").sum().cast(pl.Int64).alias("n_rifle"),
        (pl.col("cls") == "SMG").sum().cast(pl.Int64).alias("n_smg"),
        (pl.col("cls") == "SNIPER").sum().cast(pl.Int64).alias("n_sniper"),
        (pl.col("cls") == "HEAVY").sum().cast(pl.Int64).alias("n_heavy"),
        (pl.col("cls") == "NONE").sum().cast(pl.Int64).alias("n_none"),
        pl.col("equipment_value_calc").sum().alias("eq"),
        pl.col("helm").sum().cast(pl.Int64).alias("n_helm"),
        pl.col("armor_pos").sum().cast(pl.Int64).alias("n_armor_pos"),
        pl.col("armor_f").sum().alias("armor_sum"),
    ] + [(pl.col("inv_primary") == c).sum().cast(pl.Int64).alias(f"c{c}") for c in CODES_PRIMARY]
    sr = fz.group_by(["mi", "match_key", "round", "team_code"]).agg(aggs).with_columns(
        pistol=pl.col("round").is_in(PISTOL_ROUNDS),
        has_rifle=(pl.col("n_rifle") > 0).cast(pl.Int64),
        has_smg=(pl.col("n_smg") > 0).cast(pl.Int64),
        has_sniper=(pl.col("n_sniper") > 0).cast(pl.Int64),
        has_heavy=(pl.col("n_heavy") > 0).cast(pl.Int64),
        has_none=(pl.col("n_none") > 0).cast(pl.Int64),
        all_none=(pl.col("n_none") == pl.col("n")).cast(pl.Int64),
    ).with_columns(
        side_cls=pl.when(pl.col("n_rifle") > 0).then(pl.lit("RIFLE"))
        .when((pl.col("n_sniper") > 0) | (pl.col("n_heavy") > 0)).then(pl.lit("SNIPER_HEAVY"))
        .otherwise(pl.lit("SMG_PISTOL")),
    )
    sr = sr.with_columns([(pl.col(f"c{c}") > 0).cast(pl.Int64).alias(f"hc{c}") for c in CODES_PRIMARY])
    sr = sr.join(rounds.select(["match_key", "round", "ct_win"]), on=["match_key", "round"], how="left")
    sr = sr.with_columns(
        won=pl.when(pl.col("team_code") == 3).then(pl.col("ct_win")).otherwise(1 - pl.col("ct_win")))
    sr_np = sr.filter(~pl.col("pistol"))
    print(f"- side-rounds (freeze-end): {sr.height:,}; non-pistol side-rounds: {sr_np.height:,}; "
          f"non-pistol rounds: {sr_np.select(['match_key','round']).unique().height:,}; "
          f"overtime (round > 24) non-pistol rounds: {sr_np.filter(pl.col('round') > 24).select(['match_key','round']).unique().height}")

    # ---- Table 1: holdings by side (non-pistol)
    sr_pool = pl.concat([sr_np, sr_np.with_columns(team_code=pl.lit(0, dtype=pl.Int64))])
    flags = ["has_rifle", "has_smg", "has_sniper", "has_heavy", "has_none", "all_none"]
    keys, NUM, DEN = group_matrix(sr_pool, ["team_code"], flags, M)
    pt, lo, hi = ratio_boot(C, NUM, DEN)
    side_name = {0: "both sides", 2: "T", 3: "CT"}
    cnt = {k[0]: DEN.sum(axis=0)[i] for i, k in enumerate(keys)}
    rows = []
    labels = {"has_rifle": "at least one rifle", "has_smg": "at least one SMG",
              "has_sniper": "at least one sniper (AWP/SSG 08)", "has_heavy": "at least one other heavy",
              "has_none": "at least one pistol-only player (no primary)",
              "all_none": "all players pistol-only (no primary)"}
    for j, f in enumerate(flags):
        r = [labels[f]]
        for kk in [3, 2, 0]:
            i = keys.index((kk,))
            r.append(f"{pc(pt[i, j])} [{pc(lo[i, j])}, {pc(hi[i, j])}]")
        rows.append(r)
    print("\n## Table 1. Freeze-end holdings, non-pistol rounds (share of side-rounds, 95% match-cluster CI)\n")
    print(f"Side-rounds: CT {int(cnt[3]):,}, T {int(cnt[2]):,}, both {int(cnt[0]):,}\n")
    print(md_table(["holding (at freeze-end)", "CT % [95% CI]", "T % [95% CI]", "both sides % [95% CI]"], rows))

    # ---- Table 2: per primary weapon held by at least one player
    cflags = [f"hc{c}" for c in CODES_PRIMARY]
    keys2, NUM2, DEN2 = group_matrix(sr_pool, ["team_code"], cflags, M)
    pt2, lo2, hi2 = ratio_boot(C, NUM2, DEN2)
    rows2 = []
    for j, c in enumerate(CODES_PRIMARY):
        i3, i2, i0 = keys2.index((3,)), keys2.index((2,)), keys2.index((0,))
        rows2.append((pt2[i0, j], f"{NAME_OF[c]} ({CLASS_OF[c]})",
                      f"{pc(pt2[i3, j])} [{pc(lo2[i3, j])}, {pc(hi2[i3, j])}]",
                      f"{pc(pt2[i2, j])} [{pc(lo2[i2, j])}, {pc(hi2[i2, j])}]",
                      f"{pc(pt2[i0, j])} [{pc(lo2[i0, j])}, {pc(hi2[i0, j])}]"))
    rows2.sort(key=lambda x: -x[0])
    print("\n## Table 2. Share of non-pistol side-rounds with at least one player holding each primary\n")
    print(md_table(["primary (weapon_fire name, class)", "CT % [95% CI]", "T % [95% CI]", "both % [95% CI]"],
                   [(r[1], r[2], r[3], r[4]) for r in rows2]))

    # rifle count distribution
    rc = (sr_np.with_columns(nr=pl.col("n_rifle").clip(0, 5))
          .group_by(["team_code", "nr"]).agg(pl.len().alias("n")).sort(["team_code", "nr"]))
    print("\nRifle holders per side at freeze-end (non-pistol side-rounds):")
    for team, lab in [(3, "CT"), (2, "T")]:
        sub = rc.filter(pl.col("team_code") == team)
        tot_t = sub["n"].sum()
        print(f"- {lab}: " + ", ".join(f"{r['nr']} rifles: {r['n']:,} ({100 * r['n'] / tot_t:.1f}%)" for r in sub.iter_rows(named=True)))

    # ---- side-class composition, pair types
    def make_pairs(df):
        ct = df.filter(pl.col("team_code") == 3).select(
            ["mi", "match_key", "round", pl.col("side_cls").alias("ct_cls"), pl.col("eq").alias("ct_eq"),
             pl.col("n_rifle").alias("ct_nr"), pl.col("ct_win"), pl.col("n").alias("ct_n")])
        tt = df.filter(pl.col("team_code") == 2).select(
            ["match_key", "round", pl.col("side_cls").alias("t_cls"), pl.col("eq").alias("t_eq"),
             pl.col("n_rifle").alias("t_nr"), pl.col("n").alias("t_n")])
        return ct.join(tt, on=["match_key", "round"], how="inner")

    pairs = make_pairs(sr_np)
    pairs_all = make_pairs(sr)
    print(f"\n- non-pistol rounds with both sides in the freeze-end sample: {pairs.height:,}")
    pk, PN, PD = group_matrix(pairs, ["ct_cls", "t_cls"], ["ct_win"], M)
    pt3, lo3, hi3 = ratio_boot(C, PN, PD)
    rows3 = []
    for i, k in enumerate(pk):
        if PD.sum(axis=0)[i] < 1:
            continue
        rows3.append((-int(PD.sum(axis=0)[i]), f"{k[0]} vs {k[1]}", f"{int(PD.sum(axis=0)[i]):,}",
                      f"{pc(pt3[i, 0])} [{pc(lo3[i, 0])}, {pc(hi3[i, 0])}]"))
    rows3.sort()
    print("\n## Table 3. Pair types, non-pistol rounds (CT class vs T class; CT win % with 95% CI)\n")
    print(md_table(["CT class vs T class", "rounds", "CT win % [95% CI]"], [r[1:] for r in rows3]))

    # ---- Table 4: rifle side vs SMG/pistol-only side
    sub = pairs.filter(((pl.col("ct_cls") == "RIFLE") & (pl.col("t_cls") == "SMG_PISTOL")) |
                       ((pl.col("t_cls") == "RIFLE") & (pl.col("ct_cls") == "SMG_PISTOL")))
    sub = sub.with_columns(
        rifle_ct=(pl.col("ct_cls") == "RIFLE").cast(pl.Int64),
        y=pl.when(pl.col("ct_cls") == "RIFLE").then(pl.col("ct_win")).otherwise(1 - pl.col("ct_win")),
        eq_k=(pl.when(pl.col("ct_cls") == "RIFLE").then(pl.col("ct_eq") - pl.col("t_eq"))
              .otherwise(pl.col("t_eq") - pl.col("ct_eq"))) / 1000.0,
        nr_diff=pl.when(pl.col("ct_cls") == "RIFLE").then(pl.col("ct_nr") - pl.col("t_nr"))
        .otherwise(pl.col("t_nr") - pl.col("ct_nr")).cast(pl.Float64),
        ot=pl.col("round") > 24,
    )
    sub_g = pl.concat([
        sub.with_columns(grp=pl.lit("all")),
        sub.filter(pl.col("rifle_ct") == 1).with_columns(grp=pl.lit("rifle side = CT")),
        sub.filter(pl.col("rifle_ct") == 0).with_columns(grp=pl.lit("rifle side = T")),
    ])
    gk, GN, GD = group_matrix(sub_g, ["grp"], ["y"], M)
    gpt, glo, ghi = ratio_boot(C, GN, GD)
    rows4 = []
    for i, k in enumerate(gk):
        rows4.append((k[0], f"{int(GD.sum(axis=0)[i]):,}", f"{pc(gpt[i, 0])} [{pc(glo[i, 0])}, {pc(ghi[i, 0])}]"))
    n_match_sub = int(sub.select("mi").n_unique())
    ot_n = sub.filter(pl.col("ot")).height
    print("\n## Table 4. Round win rate of the side holding a rifle vs a side holding only SMGs or pistols (non-pistol rounds)\n")
    print(f"Rounds in comparison: {sub.height:,} from {n_match_sub} matches; overtime rounds (>24) inside it: {ot_n}.\n")
    print(md_table(["subset", "rounds", "rifle-side win % [95% CI]"], rows4))

    sub_b = sub.with_columns(
        bin=pl.when(pl.col("eq_k") < -5).then(pl.lit("a: rifle side behind by more than $5k"))
        .when(pl.col("eq_k") < 0).then(pl.lit("b: rifle side behind by $0-5k"))
        .when(pl.col("eq_k") < 5).then(pl.lit("c: rifle side ahead by $0-5k"))
        .otherwise(pl.lit("d: rifle side ahead by more than $5k")))
    bk, BN, BD = group_matrix(sub_b, ["bin"], ["y"], M)
    bpt, blo_, bhi_ = ratio_boot(C, BN, BD)
    rows4c = [(k[0], f"{int(BD.sum(axis=0)[i]):,}", f"{pc(bpt[i, 0])} [{pc(blo_[i, 0])}, {pc(bhi_[i, 0])}]")
              for i, k in enumerate(bk)]
    print("\n## Table 4c. Rifle-side win rate by freeze-end equipment gap (equipment_value_calc, rifle side minus SMG/pistol side)\n")
    print(md_table(["equipment gap bin", "rounds", "rifle-side win % [95% CI]"], rows4c))

    # rifle vs sniper/heavy context
    sub2 = pairs.filter(((pl.col("ct_cls") == "RIFLE") & (pl.col("t_cls") == "SNIPER_HEAVY")) |
                        ((pl.col("t_cls") == "RIFLE") & (pl.col("ct_cls") == "SNIPER_HEAVY")))
    sub2 = sub2.with_columns(y=pl.when(pl.col("ct_cls") == "RIFLE").then(pl.col("ct_win")).otherwise(1 - pl.col("ct_win")),
                             grp=pl.lit("rifle vs sniper/heavy"))
    g2k, G2N, G2D = group_matrix(sub2, ["grp"], ["y"], M)
    g2pt, g2lo, g2hi = ratio_boot(C, G2N, G2D)
    print(f"\nContext: rifle side vs sniper/heavy-only side: {int(G2D.sum()):,} rounds, rifle-side win "
          f"{pc(g2pt[0,0])}% [{pc(g2lo[0,0])}, {pc(g2hi[0,0])}]")

    # side-class mean equipment (confounding context)
    sr_cls = sr_np.with_columns(grp=pl.col("side_cls"))
    ek, EN, ED = group_matrix(sr_cls, ["side_cls"], ["eq", "n_rifle"], M)
    ept, elo, ehi = ratio_boot(C, EN, ED)
    print("\n## Table 4b. Mean freeze-end team equipment value (equipment_value_calc, $) by side class, non-pistol side-rounds\n")
    rows_e = []
    for i, k in enumerate(ek):
        rows_e.append((k[0], f"{int(ED.sum(axis=0)[i]):,}", f"{ept[i,0]:,.0f} [{elo[i,0]:,.0f}, {ehi[i,0]:,.0f}]",
                       f"{ept[i,1]:.2f}"))
    print(md_table(["side class", "side-rounds", "mean team equipment $ [95% CI]", "mean rifles held"], rows_e))

    # ---- Table 5: logistic regressions
    def design(df, cols):
        return np.column_stack([np.ones(df.height)] + [df[c].to_numpy().astype(float) for c in cols])

    def run_model(df, cols, y_col, label):
        X = design(df, cols)
        y = df[y_col].to_numpy().astype(float)
        mi = df["mi"].to_numpy()
        beta, se = logit_fit(X, y, np.ones(len(y)))
        boots = np.empty((N_BOOT, len(beta)))
        for b in range(N_BOOT):
            boots[b], _ = logit_fit(X, y, C[b][mi])
        blo, bhi = np.percentile(boots, [2.5, 97.5], axis=0)
        return label, cols, beta, se, blo, bhi, len(y), df["mi"].n_unique()

    subA = sub.with_columns(pl.col("mi").cast(pl.Int64))
    subB = pairs.with_columns(
        eq_k=(pl.col("ct_eq") - pl.col("t_eq")) / 1000.0,
        nr_diff=(pl.col("ct_nr") - pl.col("t_nr")).cast(pl.Float64),
        rifle_ind_diff=((pl.col("ct_nr") > 0).cast(pl.Int64) - (pl.col("t_nr") > 0).cast(pl.Int64)).cast(pl.Float64),
        y=pl.col("ct_win"),
    ).filter(pl.col("round") > 0)
    models = [
        ("A  rifle vs SMG/pistol subset: y = rifle side won; eq diff + rifle count diff", subA, ["eq_k", "nr_diff"], "y"),
        ("A0 same subset, no equipment control: rifle count diff only", subA, ["nr_diff"], "y"),
        ("B  all non-pistol rounds: y = CT won; eq diff (CT-T) + rifle count diff (CT-T)", subB, ["eq_k", "nr_diff"], "y"),
        ("B0 all non-pistol rounds, no equipment control: rifle indicator diff only", subB, ["rifle_ind_diff"], "y"),
        ("B1 all non-pistol rounds: equipment diff only", subB, ["eq_k"], "y"),
        ("B2 all non-pistol rounds: eq diff (CT-T) + rifle-holding indicator diff (CT-T)", subB, ["eq_k", "rifle_ind_diff"], "y"),
    ]
    print("\n## Table 5. Logistic regression (IRLS, unpenalized); coefficients on log-odds scale; OR = exp(beta)\n")
    print("eq_k = team equipment_value_calc difference in $1,000; nr_diff = rifle holders difference (count); "
          "rifle_ind_diff = rifle-holding indicator difference. 95% CI from match-cluster bootstrap (1000 resamples, seed 0).\n")
    for label, dfm, cols, ycol in models:
        lab, cols_, beta, se, blo, bhi, n_rows, n_m = run_model(dfm, cols, ycol, label)
        rows5 = []
        names = ["intercept"] + cols_
        for j, nm in enumerate(names):
            z = beta[j] / se[j]
            rows5.append((nm, f"{beta[j]:.4f}", f"{se[j]:.4f}", f"{norm_p(z):.2g}",
                          f"[{blo[j]:.4f}, {bhi[j]:.4f}]",
                          f"{math.exp(beta[j]):.3f} [{math.exp(blo[j]):.3f}, {math.exp(bhi[j]):.3f}]"))
        print(f"### {lab}\n")
        print(f"rows={n_rows:,}, matches={n_m}\n")
        print(md_table(["term", "beta", "SE", "p (Wald)", "beta 95% CI (bootstrap)", "OR [95% CI]"], rows5))
        print()

    # ---- Table 6: pistol rounds
    pistol_pairs = pairs_all.filter(pl.col("round").is_in(PISTOL_ROUNDS))
    pk6, P6N, P6D = group_matrix(pistol_pairs, ["round"], ["ct_win"], M)
    pt6, lo6, hi6 = ratio_boot(C, P6N, P6D)
    rows6 = []
    for i, k in enumerate(pk6):
        rows6.append((f"round {k[0]}", f"{int(P6D.sum(axis=0)[i])}", f"{pc(pt6[i,0])} [{pc(lo6[i,0])}, {pc(hi6[i,0])}]",
                      f"{100 - float(pt6[i,0]) * 100:.1f}"))
    tot6 = pistol_pairs.with_columns(grp=pl.lit("both pistol rounds"))
    gk6, G6N, G6D = group_matrix(tot6, ["grp"], ["ct_win"], M)
    gp6, gl6, gh6 = ratio_boot(C, G6N, G6D)
    rows6.append(("both pistol rounds", f"{int(G6D.sum())}", f"{pc(gp6[0,0])} [{pc(gl6[0,0])}, {pc(gh6[0,0])}]", f"{100 - float(gp6[0,0]) * 100:.1f}"))
    print("\n## Table 6. Pistol rounds: CT win rate (95% CI) and T win rate\n")
    print(md_table(["round", "rounds", "CT win % [95% CI]", "T win %"], rows6))

    pst = fz.filter(pl.col("pistol") & (pl.col("ones") == 1))
    pk7, P7N, P7D = group_matrix(pst, ["round", "team_code"], ["helm", "armor_pos", "armor_f"], M)
    pt7, lo7, hi7 = ratio_boot(C, P7N, P7D)
    rows7 = []
    for i, k in enumerate(pk7):
        rows7.append((f"round {k[0]}", "CT" if k[1] == 3 else "T", f"{int(P7D.sum(axis=0)[i])}",
                      f"{pc(pt7[i,0])} [{pc(lo7[i,0])}, {pc(hi7[i,0])}]",
                      f"{pc(pt7[i,1])} [{pc(lo7[i,1])}, {pc(hi7[i,1])}]",
                      f"{pt7[i,2]:.1f} [{lo7[i,2]:.1f}, {hi7[i,2]:.1f}]"))
    print("\n## Table 7. Pistol rounds, freeze-end player equipment by side (player rows)\n")
    print(md_table(["round", "side", "players", "helmet % [95% CI]", "armor>0 % [95% CI]", "mean armor [95% CI]"], rows7))

    # ---- Table 8: armor and helmet by class (non-pistol)
    fzn = fz.filter(~pl.col("pistol")).with_columns(grp=pl.col("cls"))
    fzn_all = pl.concat([fzn, fzn.with_columns(grp=pl.lit("ALL"))])
    ak, AN, AD = group_matrix(fzn_all, ["grp"], ["helm", "armor_pos", "armor_f"], M)
    pt8, lo8, hi8 = ratio_boot(C, AN, AD)
    rows8 = []
    order8 = ["RIFLE", "SMG", "SNIPER", "HEAVY", "NONE", "ALL"]
    for g in order8:
        i = ak.index((g,))
        rows8.append((g, f"{int(AD.sum(axis=0)[i]):,}",
                      f"{pc(pt8[i,0])} [{pc(lo8[i,0])}, {pc(hi8[i,0])}]",
                      f"{pc(pt8[i,1])} [{pc(lo8[i,1])}, {pc(hi8[i,1])}]",
                      f"{pt8[i,2]:.1f} [{lo8[i,2]:.1f}, {hi8[i,2]:.1f}]"))
    print("\n## Table 8. Freeze-end armor and helmet by primary class (non-pistol rounds, player rows)\n")
    print(md_table(["class", "players", "helmet % [95% CI]", "armor>0 % [95% CI]", "mean armor [95% CI]"], rows8))

    # ---- sample sizes for the notes
    print(f"\n- matches in comparison subset (rifle vs SMG/pistol): {n_match_sub} of {M}")


if __name__ == "__main__":
    main()
