"""W1 part 1 and 2: kills by weapon and shots by weapon, 2026 tome, all matches.

Kills come from basic_player_death; shots from basic_weapon_fire.
World deaths: weapon in {world, worldent, trigger_hurt, planted_c4} or null attacker.
Same-team deaths (including self kills): attacker side == victim side.
Enemy kills: everything else with both sides known. Attacker side is filled from the
same match, round and player id when attacker_team_code is null (lookup built from
the victim, attacker and weapon_fire team codes).
Uncertainty: percentile bootstrap over matches (resample matches with replacement),
1000 resamples, seed 0. Rounds are not resampled.
"""
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
N_BOOT = 1000
SEED = 0
WORLD_WEAPONS = ["world", "worldent", "trigger_hurt", "planted_c4"]
TEAM_NAME = {2: "T", 3: "CT"}


def files(channel):
    fs = sorted(TOME.glob(f"basic_{channel}.*/dataframe_*"))
    if not fs:
        raise SystemExit(f"no files for {channel}")
    return [str(f) for f in fs]


def norm_weapon(col):
    # silencer-off variants are the same gun as the silencer-on name
    return (
        pl.when(col == "m4a1_silencer_off").then(pl.lit("m4a1_silencer"))
        .when(col == "usp_silencer_off").then(pl.lit("usp_silencer"))
        .otherwise(col)
    )


def pct(x):
    return f"{100 * x:.2f}"


def boot_matrix(rng, n_match):
    idx = rng.integers(0, n_match, size=(N_BOOT, n_match))
    C = np.empty((N_BOOT, n_match), dtype=np.float64)
    for b in range(N_BOOT):
        C[b] = np.bincount(idx[b], minlength=n_match)
    return C


def boot_sum(C, A):
    # per-resample column sums: row b = sum_m C[b, m] * A[m, :]; no BLAS (crashes on this Windows build)
    out = np.empty((C.shape[0], A.shape[1]), dtype=np.float64)
    for b in range(C.shape[0]):
        out[b] = (C[b][:, None] * A).sum(axis=0)
    return out


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    rng = np.random.default_rng(SEED)

    death = pl.read_parquet(
        files("player_death"),
        columns=["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name",
                 "player_team_code", "attacker_team_code"],
    )
    fire = pl.read_parquet(
        files("weapon_fire"),
        columns=["match_key", "round", "tick", "player_id", "weapon_name",
                 "player_weapon_code", "player_team_code"],
    )
    rend = pl.read_parquet(files("round_end"), columns=["match_key", "round"])

    universe = sorted(
        set(death["match_key"].unique().to_list())
        | set(fire["match_key"].unique().to_list())
        | set(rend["match_key"].unique().to_list())
    )
    M = len(universe)
    midx = pl.DataFrame({"match_key": universe, "mi": list(range(M))})
    print(f"# W1 kills and shots (2026 tome, all matches)\n")
    print(f"- matches in universe (union of death, fire, round_end): {M}")
    print(f"- matches with >=1 death: {death['match_key'].n_unique()}")
    print(f"- matches with >=1 weapon_fire: {fire['match_key'].n_unique()}")
    print(f"- player_death rows: {death.height:,}; weapon_fire rows: {fire.height:,}")

    dup_d = death.select(["match_key", "round", "tick", "player_id"]).is_duplicated().sum()
    dup_f = fire.select(["match_key", "round", "tick", "player_id", "weapon_name"]).is_duplicated().sum()
    print(f"- duplicate keys: player_death (match,round,tick,player_id) = {dup_d}; "
          f"weapon_fire (match,round,tick,player_id,weapon_name) = {dup_f}")

    # ---- team lookup (match, round, player_id) -> team code, from all sources
    pairs = pl.concat([
        death.select(["match_key", "round", pl.col("player_id").alias("pid"),
                      pl.col("player_team_code").alias("team")]),
        death.select(["match_key", "round", pl.col("attacker_id").alias("pid"),
                      pl.col("attacker_team_code").alias("team")]),
        fire.select(["match_key", "round", pl.col("player_id").alias("pid"),
                     pl.col("player_team_code").alias("team")]),
    ]).drop_nulls(["pid", "team"]).filter(pl.col("team").is_in([2, 3]))
    conf = (pairs.group_by(["match_key", "round", "pid"])
            .agg(pl.col("team").n_unique().alias("nt"))
            .filter(pl.col("nt") > 1).height)
    tmap = (pairs.group_by(["match_key", "round", "pid"])
            .agg(pl.col("team").first().alias("team_map")))
    print(f"- team lookup keys: {tmap.height:,}; keys with conflicting team codes: {conf}")

    d = death.join(
        tmap.rename({"pid": "player_id"}), on=["match_key", "round", "player_id"], how="left"
    ).join(
        tmap.rename({"pid": "attacker_id", "team_map": "att_team_map"}),
        on=["match_key", "round", "attacker_id"], how="left",
    )
    att_known = pl.col("attacker_team_code").is_in([2, 3]).fill_null(False)
    vic_known = pl.col("player_team_code").is_in([2, 3]).fill_null(False)
    d = d.with_columns(
        vic_team=pl.when(vic_known).then(pl.col("player_team_code")).otherwise(pl.col("team_map")),
        att_team=pl.when(att_known).then(pl.col("attacker_team_code")).otherwise(pl.col("att_team_map")),
        att_team_filled=(~att_known) & pl.col("att_team_map").is_not_null(),
        weapon=norm_weapon(pl.col("weapon_name")),
    )
    n_att_null = d.filter(~att_known).height
    n_att_filled = d.filter(pl.col("att_team_filled")).height
    print(f"- deaths with null/0 attacker_team_code: {n_att_null:,}; filled from lookup: {n_att_filled:,}")

    d = d.with_columns(
        cat=pl.when(pl.col("weapon_name").is_in(WORLD_WEAPONS) | pl.col("attacker_id").is_null())
        .then(pl.lit("world"))
        .when(pl.col("att_team").is_null() | pl.col("vic_team").is_null())
        .then(pl.lit("unknown_team"))
        .when(pl.col("att_team") == pl.col("vic_team"))
        .then(pl.when(pl.col("attacker_id") == pl.col("player_id")).then(pl.lit("self"))
              .otherwise(pl.lit("team")))
        .otherwise(pl.lit("enemy"))
    )
    cat_counts = d.group_by("cat").agg(pl.len().alias("n")).sort("n", descending=True)
    print("\n## Death classification (all player_death rows)\n")
    tot = d.height
    rows = [(r["cat"], f"{r['n']:,}", f"{100 * r['n'] / tot:.2f}") for r in cat_counts.iter_rows(named=True)]
    print(md_table(["category", "deaths", "% of all deaths"], rows))
    nw_null = d.filter((pl.col("cat") != "world") & (~pl.col("attacker_team_code").is_in([2, 3]).fill_null(False))).height
    nw_fill = d.filter((pl.col("cat") != "world") & pl.col("att_team_filled")).height
    print(f"- non-world deaths with null/0 attacker_team_code: {nw_null:,}; of these filled from lookup: {nw_fill:,}; "
          f"left unknown (excluded as unknown_team): {nw_null - nw_fill:,}")
    print("\nworld = weapon in {world, worldent, trigger_hurt, planted_c4} or null attacker. "
          "team = same side, self = attacker is victim (subset of same side). "
          "enemy = opposite side. unknown_team = side missing after lookup.")

    # the weapon label for world deaths
    wl = (d.filter(pl.col("cat") == "world").group_by("weapon_name").agg(pl.len().alias("n"))
          .sort("n", descending=True))
    print("\nWorld death weapon labels: " + ", ".join(f"{r['weapon_name']}={r['n']}" for r in wl.iter_rows(named=True)))

    enemy = d.filter(pl.col("cat") == "enemy")
    E = enemy.height
    print(f"\nEnemy kills (denominator): {E:,}")

    # per-match matrices
    weapons = sorted(enemy["weapon"].unique().to_list())
    W = len(weapons)
    widx = {w: i for i, w in enumerate(weapons)}
    en = enemy.join(midx, on="match_key", how="left")
    K = np.zeros((M, W))
    Kct = np.zeros((M, W))
    agg = en.group_by(["mi", "weapon"]).agg(
        pl.len().alias("n"),
        (pl.col("att_team") == 3).sum().alias("n_ct"),
    )
    for r in agg.iter_rows(named=True):
        K[r["mi"], widx[r["weapon"]]] = r["n"]
        Kct[r["mi"], widx[r["weapon"]]] = r["n_ct"]

    tot_k = K.sum(axis=0)
    tot_ct = Kct.sum(axis=0)
    C = boot_matrix(rng, M)
    Kb = boot_sum(C, K)
    Kctb = boot_sum(C, Kct)
    tot_b = Kb.sum(axis=1, keepdims=True)
    share_b = Kb / tot_b
    ctshare_b = np.divide(Kctb, Kb, out=np.full_like(Kb, np.nan), where=Kb > 0)

    order = np.argsort(-tot_k)
    rows = []
    for rank, j in enumerate(order[:20], start=1):
        w = weapons[j]
        s = tot_k[j] / E
        slo, shi = np.percentile(share_b[:, j], [2.5, 97.5])
        cts = tot_ct[j] / tot_k[j] if tot_k[j] else float("nan")
        clo, chi = np.nanpercentile(ctshare_b[:, j], [2.5, 97.5])
        rows.append((rank, w, f"{int(tot_k[j]):,}", f"{100 * s:.2f} [{100 * slo:.2f}, {100 * shi:.2f}]",
                     f"{int(tot_ct[j]):,}", f"{int(tot_k[j] - tot_ct[j]):,}",
                     f"{100 * cts:.1f} [{100 * clo:.1f}, {100 * chi:.1f}]"))
    print("\n## Table 1. Top 20 weapons by enemy kills (all 1,752 matches)\n")
    print(md_table(["rank", "weapon (death label)", "kills", "share % of enemy kills [95% CI]",
                    "CT attackers", "T attackers", "CT share % [95% CI]"], rows))

    knife_mask = np.array([w.startswith("knife") or w == "bayonet" for w in weapons])
    ks = share_b[:, knife_mask].sum(axis=1)
    print(f"\nKnives (all knife_* and bayonet): kills={int(tot_k[knife_mask].sum()):,}, "
          f"share={100 * tot_k[knife_mask].sum() / E:.2f}% [{100 * np.percentile(ks, 2.5):.2f}, {100 * np.percentile(ks, 97.5):.2f}]")

    print("\n## Table 1b. All weapons with enemy kills (full list)\n")
    rows = []
    for j in order:
        rows.append((weapons[j], f"{int(tot_k[j]):,}", f"{100 * tot_k[j] / E:.3f}",
                     f"{int(tot_ct[j]):,}", f"{int(tot_k[j] - tot_ct[j]):,}"))
    print(md_table(["weapon", "kills", "share %", "CT attackers", "T attackers"], rows))

    # ---- shots
    fire = fire.join(midx, on="match_key", how="left")
    fire = fire.with_columns(weapon=norm_weapon(pl.col("weapon_name")))
    sweapons = sorted(fire["weapon"].unique().to_list())
    SW = len(sweapons)
    swidx = {w: i for i, w in enumerate(sweapons)}
    S = np.zeros((M, SW))
    Sct = np.zeros((M, SW))
    sagg = fire.group_by(["mi", "weapon"]).agg(
        pl.len().alias("n"),
        (pl.col("player_team_code") == 3).sum().alias("n_ct"),
    )
    for r in sagg.iter_rows(named=True):
        S[r["mi"], swidx[r["weapon"]]] = r["n"]
        Sct[r["mi"], swidx[r["weapon"]]] = r["n_ct"]
    tot_s = S.sum(axis=0)
    tot_sct = Sct.sum(axis=0)
    Sb = boot_sum(C, S)
    Sctb = boot_sum(C, Sct)
    S_all = Sb.sum(axis=1, keepdims=True)
    sshare_b = Sb / S_all
    S_tot = tot_s.sum()
    sorder = np.argsort(-tot_s)
    rows = []
    for rank, j in enumerate(sorder, start=1):
        w = sweapons[j]
        s = tot_s[j] / S_tot
        slo, shi = np.percentile(sshare_b[:, j], [2.5, 97.5])
        mch = int((S[:, j] > 0).sum())
        rows.append((rank, w, f"{int(tot_s[j]):,}", f"{100 * s:.3f} [{100 * slo:.3f}, {100 * shi:.3f}]",
                     f"{int(tot_sct[j]):,}", f"{int(tot_s[j] - tot_sct[j]):,}", mch))
    print(f"\n## Table 2. Shots by weapon_fire weapon name (all matches; total shots {int(S_tot):,})\n")
    print(md_table(["rank", "weapon", "shots", "share % [95% CI]", "CT-team shots", "T-team shots",
                    "matches with >=1 shot"], rows))
    print(f"\nShots by side: CT-team={int(tot_sct.sum()):,}, T-team={int(S_tot - tot_sct.sum()):,} "
          f"(T-team here = all non-CT rows, including any with null team code)")

    # ---- code to name mapping check (weapon_fire)
    null_code = fire.filter(pl.col("player_weapon_code").is_null()).height
    nonnull = fire.filter(pl.col("player_weapon_code").is_not_null())
    cn = (nonnull.group_by(["player_weapon_code", "weapon_name"]).agg(pl.len().alias("n"))
          .sort(["player_weapon_code", "n"], descending=[False, True]))
    modal = cn.group_by("player_weapon_code", maintain_order=True).first()
    tot_by_code = cn.group_by("player_weapon_code").agg(pl.col("n").sum().alias("tot"))
    mc = modal.join(tot_by_code, on="player_weapon_code")
    dis = int(mc.select((pl.col("tot") - pl.col("n")).sum()).item())
    print(f"\n## Table 3. weapon_code to weapon_name mapping (weapon_fire)\n")
    print(f"- rows with null player_weapon_code: {null_code:,}")
    print(f"- rows with non-null code: {nonnull.height:,}; rows whose weapon_name differs from the modal name "
          f"for that code: {dis:,} ({100 * dis / nonnull.height:.3f}%)")
    rows = []
    for r in mc.sort("player_weapon_code").iter_rows(named=True):
        rows.append((r["player_weapon_code"], r["weapon_name"], f"{r['n']:,}", f"{r['tot']:,}",
                     f"{100 * r['n'] / r['tot']:.2f}"))
    print(md_table(["weapon_code", "modal weapon_name", "rows with modal name", "rows with code",
                    "% modal"], rows))

    # ---- reconciliation of kill categories (for the notes)
    print("\n## Table 4. Kill-category reconciliation\n")
    rec = [(r["cat"], f"{r['n']:,}") for r in cat_counts.iter_rows(named=True)]
    print(md_table(["category", "deaths"], rec))


if __name__ == "__main__":
    main()
