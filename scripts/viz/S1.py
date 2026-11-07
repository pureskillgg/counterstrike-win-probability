"""S1: armor and helmet share at freeze end, by regulation round number (1-24).

Reads only:
  data/state_sample/players_1hz.parquet   (projected columns)
  data/state_sample/rounds.parquet        (round_number, live_start_tick cross-checks)
Writes only:
  data/viz/S1.json
  data/viz/S1.svg

Definitions
  Freeze-end snapshot: first snapshot (minimum tick) per (match_key, round).
    Cross-checked: its tick equals rounds.live_start_tick and seconds_since_live == 0.
  Armor share:  players with armor > 0 / players alive at freeze end.
  Helmet share: players with has_helmet / players alive at freeze end.
  Pooled share: sum of numerators / sum of denominators over the matches.

Interval
  Match-cluster bootstrap: resample the matches with replacement, 1000 replicates,
  numpy default_rng(0). Each replicate uses the same resampled matches for every
  round number. 95% percentile interval (2.5th and 97.5th, numpy linear method).
"""

import json
import os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PLAYERS = os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet")
ROUNDS = os.path.join(ROOT, "data", "state_sample", "rounds.parquet")
OUT_JSON = os.path.join(ROOT, "data", "viz", "S1.json")
OUT_SVG = os.path.join(ROOT, "data", "viz", "S1.svg")

N_BOOT = 1000
SEED = 0
RMIN, RMAX = 1, 24
NR = RMAX - RMIN + 1
PISTOL = (1, 13)
HALFTIME_AFTER = 12  # boundary between round 12 and round 13

C_ARMOR = "#475569"   # neutral
C_HELMET = "#059669"  # accent
C_GRID = "#e2e8f0"
C_PISTOL = "#f1f5f9"
C_TEXT = "#334155"


def load():
    base = pl.scan_parquet(PLAYERS).select(
        ["match_key", "round", "tick", "seconds_since_live", "player_id", "team_code", "armor", "has_helmet", "split"]
    )
    first = base.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t0"))
    snap = (
        base.join(first, on=["match_key", "round"], how="inner")
        .filter((pl.col("tick") == pl.col("t0")) & pl.col("round").is_between(RMIN, RMAX))
        .collect()
    )
    fr = (
        first.join(
            pl.scan_parquet(ROUNDS).select(["match_key", "round", "round_number", "live_start_tick"]),
            on=["match_key", "round"],
            how="left",
        )
        .filter(pl.col("round").is_between(RMIN, RMAX))
        .collect()
    )
    total_rounds_sample = pl.scan_parquet(ROUNDS).select(pl.len()).collect().item()
    return snap, fr, total_rounds_sample


def run_checks(snap, fr):
    checks = {}
    checks["snapshot_rows"] = int(snap.height)
    checks["snapshot_rounds_1_24"] = int(snap.select(["match_key", "round"]).unique().height)
    per_round_players = snap.group_by(["match_key", "round"]).agg(pl.len().alias("k"))
    checks["players_per_round_min"] = int(per_round_players["k"].min())
    checks["players_per_round_max"] = int(per_round_players["k"].max())
    checks["rounds_not_10_players"] = [
        {"match_key": str(m), "round": int(r), "players": int(k)}
        for m, r, k in per_round_players.filter(pl.col("k") != 10).iter_rows()
    ]
    checks["duplicate_player_rows"] = int(snap.height - snap.unique(["match_key", "round", "player_id"]).height)
    checks["null_armor"] = int(snap["armor"].null_count())
    checks["null_has_helmet"] = int(snap["has_helmet"].null_count())
    checks["negative_armor_rows"] = int((snap["armor"] < 0).sum())
    checks["seconds_since_live_nonzero_rows"] = int((snap["seconds_since_live"] != 0).sum())
    checks["rounds_in_rounds_parquet_match"] = int(fr["live_start_tick"].null_count() == 0)
    checks["tick_vs_live_start_mismatch"] = int((fr["t0"] != fr["live_start_tick"]).sum())
    checks["round_vs_round_number_mismatch"] = int((fr["round"] != fr["round_number"]).sum())
    checks["helmet_without_armor_rows"] = int(((snap["has_helmet"]) & (snap["armor"] <= 0)).sum())
    checks["team_code_counts"] = {
        str(k): int(v) for k, v in snap.group_by("team_code").agg(pl.len().alias("n")).iter_rows()
    }
    checks["split_counts_rows"] = {
        str(k): int(v) for k, v in snap.group_by("split").agg(pl.len().alias("n")).iter_rows()
    }
    assert checks["duplicate_player_rows"] == 0
    assert checks["null_armor"] == 0 and checks["null_has_helmet"] == 0
    assert checks["negative_armor_rows"] == 0
    assert checks["seconds_since_live_nonzero_rows"] == 0
    assert checks["tick_vs_live_start_mismatch"] == 0
    assert checks["round_vs_round_number_mismatch"] == 0
    # One round has 9 alive players at freeze end; kept with its observed denominator (see checks).
    assert checks["players_per_round_max"] == 10
    assert len(checks["rounds_not_10_players"]) <= 1
    return checks


def ci_stats(boot, point):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    snap, fr, total_rounds_sample = load()
    checks = run_checks(snap, fr)

    per = snap.group_by(["match_key", "round"]).agg(
        pl.len().alias("players"),
        (pl.col("armor") > 0).sum().alias("armor_pos"),
        pl.col("has_helmet").sum().alias("helmet"),
        (pl.col("has_helmet") & (pl.col("armor") <= 0)).sum().alias("helmet_no_armor"),
    )
    matches = sorted(per["match_key"].unique().to_list())
    mi = {m: i for i, m in enumerate(matches)}
    M = len(matches)

    PL = np.zeros((M, NR))  # players alive at freeze end
    AP = np.zeros((M, NR))  # players with armor > 0
    HL = np.zeros((M, NR))  # players with helmet
    HN = np.zeros((M, NR))  # players with helmet and no armor (consistency)
    for row in per.iter_rows(named=True):
        i = mi[row["match_key"]]
        j = row["round"] - RMIN
        PL[i, j] = row["players"]
        AP[i, j] = row["armor_pos"]
        HL[i, j] = row["helmet"]
        HN[i, j] = row["helmet_no_armor"]

    n_rounds = (PL > 0).sum(axis=0)
    tot_pl = PL.sum(axis=0)
    arm_pt = AP.sum(axis=0) / tot_pl
    hel_pt = HL.sum(axis=0) / tot_pl

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, M, size=(N_BOOT, M))
    bPL = PL[draws].sum(axis=1)
    bAP = AP[draws].sum(axis=1)
    bHL = HL[draws].sum(axis=1)
    assert (bPL > 0).all(), "a bootstrap replicate has an empty round"
    bArm = bAP / bPL
    bHel = bHL / bPL
    lo_arm, hi_arm = np.percentile(bArm, [2.5, 97.5], axis=0)
    lo_hel, hi_hel = np.percentile(bHel, [2.5, 97.5], axis=0)

    interval_ok = bool(np.all(lo_arm <= arm_pt) and np.all(arm_pt <= hi_arm)
                       and np.all(lo_hel <= hel_pt) and np.all(hel_pt <= hi_hel))
    assert np.all(lo_arm <= hi_arm) and np.all(lo_hel <= hi_hel)
    assert (n_rounds > 0).all() and (AP.sum(axis=0) >= 0).all()

    # Key numbers (pooled over rounds 1 to 24, pistols, and the halftime step).
    all_pt_arm = AP.sum() / PL.sum()
    all_pt_hel = HL.sum() / PL.sum()
    b_all_arm = bAP.sum(axis=1) / bPL.sum(axis=1)
    b_all_hel = bHL.sum(axis=1) / bPL.sum(axis=1)

    j1, j12, j13 = 0, 11, 12
    key = {
        "rounds_1_24_pooled": {
            "armor": ci_stats(b_all_arm, all_pt_arm),
            "helmet": ci_stats(b_all_hel, all_pt_hel),
        },
        "pistol_round_1": {
            "armor": ci_stats(bArm[:, j1], arm_pt[j1]),
            "helmet": ci_stats(bHel[:, j1], hel_pt[j1]),
        },
        "pistol_round_13": {
            "armor": ci_stats(bArm[:, j13], arm_pt[j13]),
            "helmet": ci_stats(bHel[:, j13], hel_pt[j13]),
        },
        "round_12": {
            "armor": ci_stats(bArm[:, j12], arm_pt[j12]),
            "helmet": ci_stats(bHel[:, j12], hel_pt[j12]),
        },
        "diff_round13_minus_round12_pts": {
            "armor": ci_stats(bArm[:, j13] - bArm[:, j12], arm_pt[j13] - arm_pt[j12]),
            "helmet": ci_stats(bHel[:, j13] - bHel[:, j12], hel_pt[j13] - hel_pt[j12]),
        },
        "helmet_without_armor_share_rounds_1_24": float(HN.sum() / PL.sum()),
        "n_rounds_1_24": int(n_rounds.sum()),
        "n_players_1_24": int(PL.sum()),
        "n_matches": int(M),
        "n_rounds_round_1": int(n_rounds[j1]),
        "n_rounds_round_12": int(n_rounds[j12]),
        "n_rounds_round_13": int(n_rounds[j13]),
        "n_rounds_round_24": int(n_rounds[NR - 1]),
        "total_rounds_in_sample_all_numbers": int(total_rounds_sample),
        "rounds_excluded_overtime_25_to_29": int(total_rounds_sample - n_rounds.sum()),
    }

    rows = []
    for j in range(NR):
        r = RMIN + j
        rows.append({
            "round": r,
            "pistol": r in PISTOL,
            "n_rounds": int(n_rounds[j]),
            "n_players": int(tot_pl[j]),
            "armor_players": int(AP.sum(axis=0)[j]),
            "armor_share": round(float(arm_pt[j]), 6),
            "armor_lo": round(float(lo_arm[j]), 6),
            "armor_hi": round(float(hi_arm[j]), 6),
            "helmet_players": int(HL.sum(axis=0)[j]),
            "helmet_share": round(float(hel_pt[j]), 6),
            "helmet_lo": round(float(lo_hel[j]), 6),
            "helmet_hi": round(float(hi_hel[j]), 6),
            "helmet_without_armor_players": int(HN.sum(axis=0)[j]),
        })

    result = {
        "id": "S1",
        "title": "Armor and helmet share at freeze end, by round number",
        "source": "data/state_sample/players_1hz.parquet; data/state_sample/rounds.parquet (cross-checks)",
        "definition": {
            "snapshot": "first snapshot (minimum tick) per match_key and round; equals rounds.live_start_tick",
            "population": "players alive at freeze end (10 per round in all checked rounds)",
            "armor": "armor > 0",
            "helmet": "has_helmet",
            "rounds": "round number 1 to 24 (regulation); overtime rounds 25 to 29 excluded",
        },
        "method": {
            "estimator": "pooled ratio of sums over matches",
            "interval": "match-cluster bootstrap, 95% percentile (2.5/97.5, numpy linear)",
            "resamples": N_BOOT,
            "seed": SEED,
            "numpy_rng": "default_rng",
        },
        "interval_contains_point_estimate_all_rounds": interval_ok,
        "key_numbers": key,
        "rounds": rows,
        "checks": checks,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    write_svg(rows, n_rounds, key, M)

    print("matches", M, "rounds 1-24", int(n_rounds.sum()), "players", int(PL.sum()))
    for rw in rows:
        print(
            rw["round"], rw["n_rounds"], rw["n_players"],
            "armor %.4f [%.4f, %.4f]" % (rw["armor_share"], rw["armor_lo"], rw["armor_hi"]),
            "helmet %.4f [%.4f, %.4f]" % (rw["helmet_share"], rw["helmet_lo"], rw["helmet_hi"]),
            "helmet_no_armor", rw["helmet_without_armor_players"],
        )
    print("interval_contains_point_estimate_all_rounds", interval_ok)
    print("checks", json.dumps(checks))


W, H = 800, 450
X0, X1 = 84.0, 772.0
Y0, Y1 = 58.0, 352.0
SLOT = (X1 - X0) / NR


def px(r):
    return X0 + (r - RMIN + 0.5) * SLOT


def py(v):
    return Y1 - v * (Y1 - Y0)


def write_svg(rows, n_rounds, key, M):
    rounds = [rw["round"] for rw in rows]
    arm_pt = [rw["armor_share"] for rw in rows]
    arm_lo = [rw["armor_lo"] for rw in rows]
    arm_hi = [rw["armor_hi"] for rw in rows]
    hel_pt = [rw["helmet_share"] for rw in rows]
    hel_lo = [rw["helmet_lo"] for rw in rows]
    hel_hi = [rw["helmet_hi"] for rw in rows]
    total = key["n_rounds_1_24"]

    def band(lo, hi):
        top = [f"{px(r):.1f},{py(hi[j]):.1f}" for j, r in enumerate(rounds)]
        bot = [f"{px(r):.1f},{py(lo[j]):.1f}" for j, r in reversed(list(enumerate(rounds)))]
        return "M " + " L ".join(top + bot) + " Z"

    out = []
    w = out.append
    w(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    w(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # pistol-round columns (behind everything)
    for p in PISTOL:
        w(f'<rect x="{px(p) - SLOT / 2:.1f}" y="{Y0:.1f}" width="{SLOT:.1f}" height="{Y1 - Y0:.1f}" fill="{C_PISTOL}"/>')

    # gridlines
    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        w(f'<line x1="{X0:.1f}" y1="{py(v):.1f}" x2="{X1:.1f}" y2="{py(v):.1f}" stroke="{C_GRID}" stroke-width="1"/>')

    # halftime reset between round 12 and round 13
    xh = X0 + HALFTIME_AFTER / NR * (X1 - X0)
    w(f'<line x1="{xh:.1f}" y1="{Y0:.1f}" x2="{xh:.1f}" y2="{Y1:.1f}" stroke="{C_ARMOR}" stroke-width="1.25" stroke-dasharray="5 4"/>')
    w(f'<text x="{xh - 6:.1f}" y="{Y1 - 8:.1f}" font-size="11" text-anchor="end" fill="{C_TEXT}">halftime reset (sides swap)</text>')

    # bands (match-cluster bootstrap 95%)
    w(f'<path d="{band(arm_lo, arm_hi)}" fill="{C_ARMOR}" fill-opacity="0.16" stroke="none"/>')
    w(f'<path d="{band(hel_lo, hel_hi)}" fill="{C_HELMET}" fill-opacity="0.16" stroke="none"/>')

    # lines and points
    for pt, col in ((arm_pt, C_ARMOR), (hel_pt, C_HELMET)):
        pts = " ".join(f"{px(r):.1f},{py(pt[j]):.1f}" for j, r in enumerate(rounds))
        w(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.25" stroke-linejoin="round"/>')
        for j, r in enumerate(rounds):
            w(f'<circle cx="{px(r):.1f}" cy="{py(pt[j]):.1f}" r="3" fill="{col}"/>')

    # pistol annotations: name at the top of each pistol column; exact values beside each point
    halo = f'paint-order="stroke" stroke="{C_PISTOL}" stroke-width="3" stroke-linejoin="round"'
    for p in PISTOL:
        j = p - RMIN
        a, h = arm_pt[j], hel_pt[j]
        w(f'<text x="{px(p):.1f}" y="{Y0 + 14:.1f}" font-size="11" text-anchor="middle" fill="{C_TEXT}" {halo}>pistol</text>')
        w(f'<text x="{px(p) + 7:.1f}" y="{py(a) - 6:.1f}" font-size="11" text-anchor="start" fill="{C_ARMOR}" {halo}>{100 * a:.1f}%</text>')
        w(f'<text x="{px(p) + 7:.1f}" y="{py(h) - 6:.1f}" font-size="11" text-anchor="start" fill="{C_HELMET}" {halo}>{100 * h:.1f}%</text>')

    # axes
    w(f'<line x1="{X0:.1f}" y1="{Y0:.1f}" x2="{X0:.1f}" y2="{Y1:.1f}" stroke="{C_ARMOR}" stroke-width="1"/>')
    w(f'<line x1="{X0:.1f}" y1="{Y1:.1f}" x2="{X1:.1f}" y2="{Y1:.1f}" stroke="{C_ARMOR}" stroke-width="1"/>')
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        w(f'<text x="{X0 - 8:.1f}" y="{py(v) + 4:.1f}" font-size="12" text-anchor="end" fill="{C_ARMOR}">{int(round(v * 100))}%</text>')
    for j, r in enumerate(rounds):
        w(f'<line x1="{px(r):.1f}" y1="{Y1:.1f}" x2="{px(r):.1f}" y2="{Y1 + 4:.1f}" stroke="{C_ARMOR}" stroke-width="1"/>')
        w(f'<text x="{px(r):.1f}" y="{Y1 + 18:.1f}" font-size="12" text-anchor="middle" fill="{C_ARMOR}">{r}</text>')
        w(f'<text x="{px(r):.1f}" y="{Y1 + 37:.1f}" font-size="11" text-anchor="middle" fill="{C_TEXT}">{int(n_rounds[j])}</text>')
    w(f'<text x="{X0 - 8:.1f}" y="{Y1 + 37:.1f}" font-size="11" text-anchor="end" fill="{C_TEXT}">n</text>')
    w(f'<text x="{(X0 + X1) / 2:.1f}" y="{Y1 + 58:.1f}" font-size="13" text-anchor="middle" fill="{C_ARMOR}">Round number (regulation rounds 1 to 24; bottom row = rounds per round number)</text>')
    w(f'<text transform="translate(20 {(Y0 + Y1) / 2:.1f}) rotate(-90)" font-size="13" text-anchor="middle" fill="{C_ARMOR}">Share of alive players at freeze end</text>')

    # legend (row 1) and sample size
    LY = 30.0
    w(f'<line x1="{X0:.1f}" y1="{LY}" x2="{X0 + 22:.1f}" y2="{LY}" stroke="{C_ARMOR}" stroke-width="2.25"/>')
    w(f'<circle cx="{X0 + 11:.1f}" cy="{LY}" r="3" fill="{C_ARMOR}"/>')
    w(f'<text x="{X0 + 28:.1f}" y="{LY + 4:.1f}" font-size="12" fill="{C_ARMOR}">Armor &gt; 0</text>')
    x2 = X0 + 120
    w(f'<line x1="{x2:.1f}" y1="{LY}" x2="{x2 + 22:.1f}" y2="{LY}" stroke="{C_HELMET}" stroke-width="2.25"/>')
    w(f'<circle cx="{x2 + 11:.1f}" cy="{LY}" r="3" fill="{C_HELMET}"/>')
    w(f'<text x="{x2 + 28:.1f}" y="{LY + 4:.1f}" font-size="12" fill="{C_ARMOR}">Helmet</text>')
    x3 = X0 + 205
    w(f'<rect x="{x3:.1f}" y="{LY - 7:.1f}" width="16" height="12" fill="{C_ARMOR}" fill-opacity="0.16"/>')
    w(f'<text x="{x3 + 22:.1f}" y="{LY + 4:.1f}" font-size="12" fill="{C_ARMOR}">95% match-cluster bootstrap interval</text>')
    w(f'<text x="{X0:.1f}" y="{H - 14:.1f}" font-size="12" text-anchor="start" fill="{C_TEXT}">n = {total:,} rounds (rounds 1 to 24; overtime rounds 25 to 29 excluded), {M} matches, freeze-end snapshot</text>')
    w('</svg>')

    with open(OUT_SVG, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
