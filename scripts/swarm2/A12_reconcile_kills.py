"""A12: reconcile the ak47 85,059 raw row count with the W1 enemy-kill figure (84,922).

Mirrors the W1 rules (scripts/stats/W1_weapon_kills_shots.py) on the 2026 tome:
  - team lookup (match_key, round, player_id) -> team code from victim, attacker and weapon_fire rows
  - victim and attacker sides filled from the lookup when the stored code is null or 0
  - world = weapon in {world, worldent, trigger_hurt, planted_c4} or null attacker
  - silencer-off labels merged into silencer labels (m4a1_silencer_off, usp_silencer_off)
Prints, for each merged weapon label: raw rows, enemy, team, self, unknown after fill, and the
W1 Table 1 enemy figure (copied from notes/stats/W1.md) with the difference.
Also lists every raw label that contains "_off" with its counts in deaths and in weapon_fire.
Output goes to stdout; the caller redirects to logs/swarm2/A12_reconcile_kills.log.
"""
from pathlib import Path

import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
WORLD_WEAPONS = ["world", "worldent", "trigger_hurt", "planted_c4"]

# W1 Table 1 enemy-kill counts (notes/stats/W1.md, Table 1 and 1b), for comparison only
W1_ENEMY = {
    "ak47": 84922, "m4a1_silencer": 39019, "awp": 22957, "m4a1": 16730, "usp_silencer": 12626,
    "glock": 10983, "deagle": 8723, "galilar": 8353, "mac10": 6729, "mp9": 6610, "famas": 3806,
    "mp7": 3059, "tec9": 3058, "ssg08": 2999, "fiveseven": 2502, "hegrenade": 2036, "p250": 1703,
    "xm1014": 1687, "elite": 1123, "p90": 990,
}


def folder(ch):
    hits = sorted(p for p in TOME.iterdir() if p.is_dir() and p.name.startswith(f"basic_{ch}."))
    if len(hits) != 1:
        raise SystemExit(f"{ch}: expected one folder, got {[h.name for h in hits]}")
    return hits[0]


def scan(ch):
    fs = sorted(str(p) for p in folder(ch).glob("dataframe_*"))
    if not fs:
        raise SystemExit(f"no dataframe files for {ch}")
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


def norm_weapon(col):
    return (
        pl.when(col == "m4a1_silencer_off").then(pl.lit("m4a1_silencer"))
        .when(col == "usp_silencer_off").then(pl.lit("usp_silencer"))
        .otherwise(col)
    )


def main():
    death = scan("player_death").select(
        ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name",
         "player_team_code", "attacker_team_code"]).collect()
    fire = scan("weapon_fire").select(
        ["match_key", "round", "tick", "player_id", "weapon_name", "player_team_code"]).collect()
    print(f"player_death rows: {death.height:,}; weapon_fire rows: {fire.height:,}")

    # ---- raw _off labels
    print("\n## raw labels containing _off")
    off_d = death.filter(pl.col("weapon_name").str.contains("_off")).group_by("weapon_name").agg(pl.len().alias("deaths"))
    off_f = fire.filter(pl.col("weapon_name").str.contains("_off")).group_by("weapon_name").agg(pl.len().alias("shots"))
    print(off_d)
    print(off_f)

    # ---- team lookup as in W1
    pairs = pl.concat([
        death.select(["match_key", "round", pl.col("player_id").alias("pid"),
                      pl.col("player_team_code").alias("team")]),
        death.select(["match_key", "round", pl.col("attacker_id").alias("pid"),
                      pl.col("attacker_team_code").alias("team")]),
        fire.select(["match_key", "round", pl.col("player_id").alias("pid"),
                     pl.col("player_team_code").alias("team")]),
    ]).drop_nulls(["pid", "team"]).filter(pl.col("team").is_in([2, 3]))
    conf = (pairs.group_by(["match_key", "round", "pid"]).agg(pl.col("team").n_unique().alias("nt"))
            .filter(pl.col("nt") > 1).height)
    tmap = pairs.group_by(["match_key", "round", "pid"]).agg(pl.col("team").first().alias("team_map"))
    print(f"\nteam lookup keys: {tmap.height:,}; conflicting keys: {conf}")

    d = death.join(tmap.rename({"pid": "player_id"}), on=["match_key", "round", "player_id"], how="left")
    d = d.join(tmap.rename({"pid": "attacker_id", "team_map": "att_team_map"}),
               on=["match_key", "round", "attacker_id"], how="left")
    att_known = pl.col("attacker_team_code").is_in([2, 3]).fill_null(False)
    vic_known = pl.col("player_team_code").is_in([2, 3]).fill_null(False)
    d = d.with_columns(
        vic_team=pl.when(vic_known).then(pl.col("player_team_code")).otherwise(pl.col("team_map")),
        att_team=pl.when(att_known).then(pl.col("attacker_team_code")).otherwise(pl.col("att_team_map")),
        weapon=norm_weapon(pl.col("weapon_name")),
    )
    nonworld = pl.col("weapon_name").is_in(WORLD_WEAPONS).not_() & pl.col("attacker_id").is_not_null()
    print(f"non-world deaths with null/0 attacker_team_code: {d.filter(nonworld & ~att_known).height:,}; "
          f"of these filled from lookup: {d.filter(nonworld & ~att_known & pl.col('att_team_map').is_not_null()).height:,}")
    print(f"non-world deaths with null/0 victim team code: {d.filter(nonworld & ~vic_known).height:,}; "
          f"filled from lookup: {d.filter(nonworld & ~vic_known & pl.col('team_map').is_not_null()).height:,}")

    d = d.with_columns(
        cat=pl.when(pl.col("weapon_name").is_in(WORLD_WEAPONS) | pl.col("attacker_id").is_null())
        .then(pl.lit("world"))
        .when(pl.col("att_team").is_null() | pl.col("vic_team").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("att_team") == pl.col("vic_team"))
        .then(pl.when(pl.col("attacker_id") == pl.col("player_id")).then(pl.lit("self"))
              .otherwise(pl.lit("team")))
        .otherwise(pl.lit("enemy"))
    )
    cats = d.group_by("cat").agg(pl.len().alias("n")).sort("n", descending=True)
    print("\n## death categories with lookup fill (W1 definitions)")
    for r in cats.iter_rows(named=True):
        print(f"  {r['cat']:<8} {r['n']:>9,}")

    # ---- per merged label
    per = (d.group_by(["weapon", "cat"]).agg(pl.len().alias("n"))
           .pivot(on="cat", index="weapon", values="n").fill_null(0))
    for c in ["enemy", "team", "self", "world", "unknown"]:
        if c not in per.columns:
            per = per.with_columns(pl.lit(0).cast(pl.UInt32).alias(c))
    per = per.with_columns(total=pl.col("enemy") + pl.col("team") + pl.col("self") + pl.col("world") + pl.col("unknown"))
    per = per.sort("total", descending=True)
    print("\n## per merged weapon label (W1 normalisation): raw deaths, enemy, team, self, unknown, W1 enemy, diff")
    for r in per.head(25).iter_rows(named=True):
        w = r["weapon"]
        w1 = W1_ENEMY.get(w)
        diff = "" if w1 is None else f"{r['enemy'] - w1:+d}"
        print(f"  {w:<18} total={r['total']:>7,} enemy={r['enemy']:>7,} team={r['team']:>4,} "
              f"self={r['self']:>3,} world={r['world']:>5,} unknown={r['unknown']:>4,} "
              f"W1={'' if w1 is None else f'{w1:,}':>7} diff={diff}")

    # ---- knives (W1 claims 396 enemy knife kills, all knife_* and bayonet)
    kn = d.filter(pl.col("weapon").str.starts_with("knife") | (pl.col("weapon") == "bayonet"))
    print(f"\n## knives (knife_* and bayonet): raw rows={kn.height:,}; "
          f"enemy={kn.filter(pl.col('cat') == 'enemy').height:,} (W1 says 396); "
          f"team={kn.filter(pl.col('cat') == 'team').height:,}; self={kn.filter(pl.col('cat') == 'self').height:,}; "
          f"unknown={kn.filter(pl.col('cat') == 'unknown').height:,}")

    # ---- ak47 detail
    ak = d.filter(pl.col("weapon") == "ak47")
    print(f"\n## ak47 detail: raw rows (label ak47 only) = {d.filter(pl.col('weapon_name') == 'ak47').height:,}; "
          f"merged rows = {ak.height:,}")
    print(ak.group_by("cat").agg(pl.len().alias("n")).sort("n", descending=True))
    team_ak = ak.filter(pl.col("cat") == "team")
    print("ak47 team kills: attacker side counts")
    print(team_ak.group_by("att_team").agg(pl.len().alias("n")).sort("att_team"))
    print(f"ak47 team kills with attacker_id == player_id (self): {team_ak.filter(pl.col('attacker_id') == pl.col('player_id')).height}")
    print(f"ak47 enemy kills: {ak.filter(pl.col('cat') == 'enemy').height:,}")

    # ---- shots, merged labels
    fire2 = fire.with_columns(weapon=norm_weapon(pl.col("weapon_name")))
    sh = fire2.group_by("weapon").agg(pl.len().alias("shots")).sort("shots", descending=True)
    print("\n## shots by merged label, top 12")
    for r in sh.head(12).iter_rows(named=True):
        print(f"  {r['weapon']:<18} {r['shots']:>9,}  {100 * r['shots'] / fire.height:6.2f}%")


if __name__ == "__main__":
    main()
