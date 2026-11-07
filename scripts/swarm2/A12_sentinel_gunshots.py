"""A12: two checks on the headline counts.

(1) attacker_id sentinel: player_death rows with attacker_id == 65535 (not null, not a player slot 0-19).
    Shows which weapons and which W1 categories they fall in, and whether they inflate ak47 or the enemy total.
(2) gun-only shots: weapon_fire labels classified as gun versus non-gun (knife, grenade, decoy, taser, molotov).
    Shows the gun-only total and the ak47 share of gun shots, next to the all-event share W1 reports.
Output goes to stdout; the caller redirects to logs/swarm2/A12_sentinel_gunshots.log.
"""
from pathlib import Path

import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
WORLD_WEAPONS = ["world", "worldent", "trigger_hurt", "planted_c4"]
NON_GUN_EXACT = {"flashbang", "smokegrenade", "hegrenade", "incgrenade", "molotov", "decoy", "taser",
                 "bayonet", "c4", "healthshot", "tagrenade", "firebomb", "diversion"}


def folder(ch):
    hits = sorted(p for p in TOME.iterdir() if p.is_dir() and p.name.startswith(f"basic_{ch}."))
    if len(hits) != 1:
        raise SystemExit(f"{ch}: expected one folder, got {[h.name for h in hits]}")
    return hits[0]


def scan(ch):
    fs = sorted(str(p) for p in folder(ch).glob("dataframe_*"))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


def main():
    death = scan("player_death").select(
        ["match_key", "round", "player_id", "attacker_id", "weapon_name",
         "player_team_code", "attacker_team_code"]).collect()
    print(f"player_death rows: {death.height:,}")
    print(f"attacker_id null: {death['attacker_id'].null_count():,}; attacker_id == 65535: "
          f"{death.filter(pl.col('attacker_id') == 65535).height:,}; attacker_id == 0: "
          f"{death.filter(pl.col('attacker_id') == 0).height:,}")

    # ---- (1) sentinel attacker rows
    s = death.filter(pl.col("attacker_id") == 65535)
    print("\n## (1) attacker_id == 65535 rows by weapon, top 12")
    print(s.group_by("weapon_name").agg(pl.len().alias("n")).sort("n", descending=True).head(12))
    s_att = s.group_by("attacker_team_code").agg(pl.len().alias("n")).sort("attacker_team_code")
    print("attacker_team_code on these rows:")
    print(s_att)
    s_world = s.filter(pl.col("weapon_name").is_in(WORLD_WEAPONS)).height
    print(f"sentinel rows that are world weapons: {s_world:,} of {s.height:,}")
    ak_sent = s.filter(pl.col("weapon_name") == "ak47").height
    print(f"ak47 rows with attacker_id == 65535: {ak_sent:,}")
    non_world_sent = s.filter(~pl.col("weapon_name").is_in(WORLD_WEAPONS))
    print(f"non-world sentinel rows: {non_world_sent.height:,}; with attacker_team_code in (2,3): "
          f"{non_world_sent.filter(pl.col('attacker_team_code').is_in([2, 3])).height:,}")
    ev = non_world_sent.filter(pl.col("attacker_team_code").is_in([2, 3])).with_columns(
        vic_known=pl.col("player_team_code").is_in([2, 3]))
    print("non-world sentinel rows with a known attacker side, by victim-side known flag:")
    print(ev.group_by("vic_known").agg(pl.len().alias("n")))
    cross = ev.filter(pl.col("vic_known") & (pl.col("attacker_team_code") != pl.col("player_team_code")))
    same = ev.filter(pl.col("vic_known") & (pl.col("attacker_team_code") == pl.col("player_team_code")))
    print(f"  of these, enemy-side (would count as enemy kill in W1 rules): {cross.height:,}; "
          f"same-side (would count as team kill): {same.height:,}")
    print("  weapon labels among the enemy-side sentinel rows (top 10):")
    print(cross.group_by("weapon_name").agg(pl.len().alias("n")).sort("n", descending=True).head(10))
    print("  W1 rules do not lookup player 65535 (not in the match roster), so these are counted via attacker_team_code.")

    # ---- (2) gun-only shots
    fire = scan("weapon_fire").select(["weapon_name"]).collect()
    tot = fire.height
    labs = fire.group_by("weapon_name").agg(pl.len().alias("n")).sort("n", descending=True)
    print(f"\n## (2) weapon_fire labels, all {labs.height} names (total {tot:,})")
    for r in labs.iter_rows(named=True):
        print(f"  {r['weapon_name']:<22} {r['n']:>9,}")

    def is_non_gun(name):
        n = name.lower()
        return n.startswith("knife") or n in NON_GUN_EXACT

    non_gun = sum(r["n"] for r in labs.iter_rows(named=True) if is_non_gun(r["weapon_name"]))
    gun = tot - non_gun
    ak = labs.filter(pl.col("weapon_name") == "ak47")["n"].item()
    print(f"\nnon-gun weapon_fire rows (knife*, bayonet, grenades, molotov, incgrenade, decoy, taser): {non_gun:,} "
          f"({100 * non_gun / tot:.2f}% of all weapon_fire)")
    print(f"gun-only weapon_fire rows: {gun:,}")
    print(f"ak47 share of all weapon_fire rows: {100 * ak / tot:.2f}%")
    print(f"ak47 share of gun-only weapon_fire rows: {100 * ak / gun:.2f}%")


if __name__ == "__main__":
    main()
