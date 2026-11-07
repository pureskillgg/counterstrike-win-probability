"""A12: recompute weapon kill, shot and grenade event counts from the 2026 tome.

Read-only on the tome. Projection-only lazy scans, one channel at a time.
Reports:
  (a) player_death raw rows per weapon_name (the basis of the combat.md ak47 85,059 row count)
  (b) enemy-kill, team, self, world, unknown split per weapon_name (basis of the W1 84,922 figure)
  (c) weapon_fire rows per weapon_name (shots)
  (d) grenade_state rows per event_type, grenade_id nulls, duplicate keys
  (e) weapon_fire flashbang throws vs grenade_state flashbang_detonate detonations
Output goes to stdout; the caller redirects to logs/swarm2/A12_recompute_counts.log.
"""
from pathlib import Path

import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
WORLD_WEAPONS = ["world", "worldent", "trigger_hurt", "planted_c4"]


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


def section(title):
    print(f"\n## {title}")


def main():
    # ---------------------------------------------------------------- (a)(b) kills
    death = scan("player_death").select(
        ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name",
         "player_team_code", "attacker_team_code"]
    ).collect()
    n_death = death.height
    print(f"player_death rows: {n_death:,}")
    print(f"player_death distinct match_key: {death['match_key'].n_unique():,}")

    section("(a) raw player_death rows per weapon_name, top 25 (all rows, no classification)")
    raw = death.group_by("weapon_name").agg(pl.len().alias("n")).sort("n", descending=True)
    for r in raw.head(25).iter_rows(named=True):
        print(f"  {r['weapon_name']:<22} {r['n']:>9,}  {100 * r['n'] / n_death:6.2f}%")
    ak_raw = raw.filter(pl.col("weapon_name") == "ak47")["n"].item()
    print(f"ak47 raw player_death rows: {ak_raw:,}  share of all deaths: {100 * ak_raw / n_death:.2f}%")

    section("(b) classification (W1 rules re-implemented, no lookup fill)")
    att_known = pl.col("attacker_team_code").is_in([2, 3]).fill_null(False)
    vic_known = pl.col("player_team_code").is_in([2, 3]).fill_null(False)
    d = death.with_columns(
        vic_team=pl.when(vic_known).then(pl.col("player_team_code")).otherwise(None),
        att_team=pl.when(att_known).then(pl.col("attacker_team_code")).otherwise(None),
    )
    d = d.with_columns(
        cat=pl.when(pl.col("weapon_name").is_in(WORLD_WEAPONS) | pl.col("attacker_id").is_null())
        .then(pl.lit("world"))
        .when(pl.col("att_team").is_null() | pl.col("vic_team").is_null())
        .then(pl.lit("unknown_side_no_lookup"))
        .when(pl.col("att_team") == pl.col("vic_team"))
        .then(pl.when(pl.col("attacker_id") == pl.col("player_id")).then(pl.lit("self"))
              .otherwise(pl.lit("team")))
        .otherwise(pl.lit("enemy"))
    )
    cats = d.group_by("cat").agg(pl.len().alias("n")).sort("n", descending=True)
    for r in cats.iter_rows(named=True):
        print(f"  category {r['cat']:<24} {r['n']:>9,}")
    print("  note: this run does not fill null attacker_team_code from the lookup; "
          "unknown_side_no_lookup is an upper bound on the 9 unknown-side deaths in W1.")

    section("(b) per-weapon split, top 15 and ak47")
    per = (d.group_by(["weapon_name", "cat"]).agg(pl.len().alias("n"))
           .pivot(on="cat", index="weapon_name", values="n").fill_null(0))
    cat_cols = [c for c in per.columns if c != "weapon_name"]
    per = per.with_columns(total=pl.sum_horizontal(cat_cols)).sort("total", descending=True)
    print(per.head(15))
    print(per.filter(pl.col("weapon_name") == "ak47"))
    ak_enemy = d.filter((pl.col("weapon_name") == "ak47") & (pl.col("cat") == "enemy")).height
    print(f"ak47 enemy kills (no lookup fill): {ak_enemy:,}")

    # ---------------------------------------------------------------- (c) shots
    section("(c) weapon_fire rows per weapon_name (shots), top 25")
    fire = scan("weapon_fire").select(["match_key", "round", "tick", "player_id", "weapon_name"]).collect()
    n_fire = fire.height
    print(f"weapon_fire rows: {n_fire:,}; distinct match_key: {fire['match_key'].n_unique():,}")
    sh = fire.group_by("weapon_name").agg(pl.len().alias("n")).sort("n", descending=True)
    for r in sh.head(25).iter_rows(named=True):
        print(f"  {r['weapon_name']:<22} {r['n']:>9,}  {100 * r['n'] / n_fire:6.2f}%")
    ak_shots = sh.filter(pl.col("weapon_name") == "ak47")["n"].item()
    print(f"ak47 shots (weapon_fire rows): {ak_shots:,}  share: {100 * ak_shots / n_fire:.2f}%")
    fl_throws = sh.filter(pl.col("weapon_name") == "flashbang")["n"].sum()
    print(f"flashbang weapon_fire rows (throws): {fl_throws:,}")
    dup_f = fire.select(["match_key", "round", "tick", "player_id", "weapon_name"]).is_duplicated().sum()
    print(f"weapon_fire duplicate (match,round,tick,player_id,weapon_name) rows: {dup_f:,}")
    del fire

    # ---------------------------------------------------------------- (d) grenade_state
    section("(d) grenade_state event_type counts")
    gs = scan("grenade_state").select(
        ["match_key", "round", "tick", "event_type", "entity_id", "grenade_id", "tick_throw", "player_id"]
    ).collect()
    print(f"grenade_state rows: {gs.height:,}")
    ec = (gs.group_by("event_type").agg(
        pl.len().alias("rows"),
        pl.col("grenade_id").null_count().alias("grenade_id_null"),
        pl.col("grenade_id").n_unique().alias("grenade_id_distinct"),
    ).sort("rows", descending=True))
    for r in ec.iter_rows(named=True):
        print(f"  {r['event_type']:<26} rows={r['rows']:>9,} grenade_id_null={r['grenade_id_null']:>7,} "
              f"grenade_id_distinct={r['grenade_id_distinct']:>7,}")

    det = gs.filter(pl.col("event_type") == "flashbang_detonate")
    print(f"flashbang_detonate rows: {det.height:,}")
    nn = det.filter(pl.col("grenade_id").is_not_null()).select(["match_key", "round", "grenade_id"]).n_unique()
    print(f"flashbang_detonate distinct (match_key, round, grenade_id), non-null ids: {nn:,}")
    print(f"flashbang_detonate distinct (match_key, round, entity_id): "
          f"{det.select(['match_key', 'round', 'entity_id']).n_unique():,}")
    print(f"flashbang_detonate duplicate (match,round,tick,entity_id,grenade_id) rows: "
          f"{det.select(['match_key', 'round', 'tick', 'entity_id', 'grenade_id']).is_duplicated().sum():,}")
    print(f"flashbang_detonate with null grenade_id: {det['grenade_id'].null_count():,}")

    section("(e) flashbang throws (weapon_fire) vs flashbang detonations (grenade_state)")
    print(f"  throws={fl_throws:,} detonations={det.height:,} "
          f"difference(throws - detonations)={fl_throws - det.height:,}")


if __name__ == "__main__":
    main()
