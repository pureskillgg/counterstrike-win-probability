"""W1 supplementary: armor and health damage per enemy hit by weapon, from basic_player_hurt.

Enemy hit = attacker side and victim side both known and different (team damage and self damage
excluded). Per-weapon metrics: hits, share of hits where armor absorbed damage (armor_removed > 0),
mean armor_removed and mean health_removed per hit, and armor share of total removed points.
Uncertainty: match-cluster percentile bootstrap, 1000 resamples, seed 0.
No BLAS calls (numpy matrix products crash in this Windows build).
"""
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
N_BOOT = 1000
SEED = 0
TOP_N = 15


def files(channel):
    fs = sorted(TOME.glob(f"basic_{channel}.*/dataframe_*"))
    if not fs:
        raise SystemExit(f"no files for {channel}")
    return [str(f) for f in fs]


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


def main():
    rng = np.random.default_rng(SEED)
    hurt = pl.read_parquet(
        files("player_hurt"),
        columns=["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name",
                 "health_removed", "effective_health_removed", "armor_removed", "hit_box_code",
                 "player_team_code", "attacker_team_code"],
    )
    print("# W1 supplementary: armor and health per enemy hit, by weapon (player_hurt, 2026 tome)\n")
    print(f"- player_hurt rows: {hurt.height:,}; matches: {hurt['match_key'].n_unique()}")
    dupmask = hurt.select(["match_key", "round", "tick", "player_id", "attacker_id", "hit_box_code"]).is_duplicated()
    dup = int(dupmask.sum())
    print(f"- duplicate hit keys (match, round, tick, victim, attacker, hitbox): {dup:,} rows kept (likely multi-pellet or multi-hit events)")
    dw = (hurt.filter(dupmask).group_by("weapon_name").agg(pl.len().alias("n")).sort("n", descending=True).head(6))
    print("- duplicate-key rows by weapon (top 6): " + ", ".join(f"{r['weapon_name']}={r['n']:,}" for r in dw.iter_rows(named=True)))
    known = pl.col("player_team_code").is_in([2, 3]).fill_null(False) & pl.col("attacker_team_code").is_in([2, 3]).fill_null(False)
    enemy = hurt.filter(known & (pl.col("player_team_code") != pl.col("attacker_team_code")) &
                        (pl.col("attacker_id") != pl.col("player_id")))
    print(f"- enemy hits (both sides known, opposite sides, not self): {enemy.height:,} "
          f"({100 * enemy.height / hurt.height:.1f}% of rows)")

    matches = sorted(enemy["match_key"].unique().to_list())
    M = len(matches)
    midx = {m: i for i, m in enumerate(matches)}
    C = boot_matrix(rng, M)

    e = enemy.with_columns(
        mi=pl.col("match_key").replace_strict(midx, default=-1, return_dtype=pl.Int64),
        armor_hit=(pl.col("armor_removed") > 0).cast(pl.Int64),
        hr=pl.col("health_removed").cast(pl.Float64),
        ar=pl.col("armor_removed").cast(pl.Float64),
    )
    top = (e.group_by("weapon_name").agg(pl.len().alias("n")).sort("n", descending=True).head(TOP_N))["weapon_name"].to_list()
    e = e.filter(pl.col("weapon_name").is_in(top))
    keys = top
    kidx = {k: i for i, k in enumerate(keys)}
    g = e.group_by(["mi", "weapon_name"]).agg(
        pl.len().cast(pl.Float64).alias("n"),
        pl.col("armor_hit").sum().cast(pl.Float64).alias("n_armor_hit"),
        pl.col("hr").sum().alias("hr_sum"),
        pl.col("ar").sum().alias("ar_sum"),
    )
    NUM = np.zeros((M, len(keys), 4))
    for r in g.iter_rows(named=True):
        NUM[r["mi"], kidx[r["weapon_name"]], :] = [r["n_armor_hit"], r["hr_sum"], r["ar_sum"], r["n"]]
    # point estimates
    tot = NUM.sum(axis=0)  # (K, 4): n_armor_hit, hr_sum, ar_sum, n
    K = len(keys)
    pt_armor_share = tot[:, 0] / tot[:, 3]
    pt_hr = tot[:, 1] / tot[:, 3]
    pt_ar = tot[:, 2] / tot[:, 3]
    pt_armor_of_removed = tot[:, 2] / (tot[:, 2] + tot[:, 1])
    # bootstrap
    B = boot_sum(C, NUM.reshape(M, K * 4)).reshape(-1, K, 4)
    b_armor_share = B[:, :, 0] / B[:, :, 3]
    b_hr = B[:, :, 1] / B[:, :, 3]
    b_ar = B[:, :, 2] / B[:, :, 3]
    b_armor_of_removed = B[:, :, 2] / (B[:, :, 2] + B[:, :, 1])

    def ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        return lo, hi

    la, ha = ci(b_armor_share)
    lh, hh = ci(b_hr)
    lr, hr_ = ci(b_ar)
    lo_s, hi_s = ci(b_armor_of_removed)
    print(f"\n## Table A1. Enemy hits by weapon, top {TOP_N} by hits (all matches; 95% match-cluster CI)\n")
    print("| weapon | hits | % hits with armor absorbed [95% CI] | mean health removed per hit [95% CI] | mean armor removed per hit [95% CI] | armor share of removed points % [95% CI] |")
    print("|---|---|---|---|---|---|")
    for j in np.argsort(-tot[:, 3]):
        print(f"| {keys[j]} | {int(tot[j,3]):,} | {100*pt_armor_share[j]:.1f} [{100*la[j]:.1f}, {100*ha[j]:.1f}] | "
              f"{pt_hr[j]:.1f} [{lh[j]:.1f}, {hh[j]:.1f}] | {pt_ar[j]:.1f} [{lr[j]:.1f}, {hr_[j]:.1f}] | "
              f"{100*pt_armor_of_removed[j]:.1f} [{100*lo_s[j]:.1f}, {100*hi_s[j]:.1f}] |")
    print("\nHealth = health_removed as recorded. Values above 100 occur (headshots, overkill); the field maximum is "
          f"{int(hurt['health_removed'].max()):,} and it is not verified against the game. Means are sensitive to these outliers.")
    print("effective_health_removed (same rows) has max "
          f"{int(hurt['effective_health_removed'].max()):,}; it was not used.")
    print("Armor share of removed points = armor_removed / (armor_removed + health_removed).")
    print("Hit-box and round-level breakdowns are not included; this table is descriptive and does not adjust for distance or helmet.")


if __name__ == "__main__":
    main()
