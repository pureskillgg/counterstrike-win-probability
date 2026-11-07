"""Feasibility probe 2: null shares on all columns, weapon/team/sentinel values, sample key overlap (read-only).

Run: pixi run python -I -X utf8 scripts\\eda\\feas_grenade_probe2.py
"""
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"


def read(ch):
    folder = TOME / f"basic_{ch}{SUFFIX}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def null_report(name, df, cols=None):
    print(f"\n--- null counts: {name} (rows={df.height})")
    for c in (cols or df.columns):
        n = df.get_column(c).null_count()
        if n:
            print(f"  {c}: {n} null ({n / df.height:.4f})")


def main():
    rounds = pl.read_parquet(ROOT / "data" / "state_sample" / "rounds.parquet")
    print("sample rounds columns:", rounds.columns)
    print("sample rounds rows:", rounds.height, "matches:", rounds["match_key"].n_unique())
    print("sample match_key examples:", rounds["match_key"].unique().head(3).to_list())
    print("split values:", rounds["split"].value_counts().to_dicts() if "split" in rounds.columns else None)
    sample_keys = set(rounds["match_key"].unique().to_list())

    for ch in ["grenade_state", "molotov_state", "molotov_fire", "player_hurt", "player_blind",
               "bomb_state", "bomb_action", "bomb_defuse"]:
        df = read(ch)
        mk = set(df["match_key"].unique().to_list())
        print(f"\n### {ch}: rows={df.height} matches_in_tome={len(mk)} "
              f"sample_matches_present={len(mk & sample_keys)} of {len(sample_keys)}")
        null_report(ch, df)
        if "event_type" in df.columns:
            ev = df.group_by("event_type").agg(
                pl.len().alias("n"),
                pl.col("player_id_fixed").null_count().alias("null_player_id_fixed")
                if "player_id_fixed" in df.columns else pl.lit(None).alias("null_player_id_fixed"),
            ).sort("n", descending=True)
            print("event_type x null player_id_fixed:")
            print(ev)
        if ch == "player_hurt":
            print("weapon_name values (top 30):")
            print(df.group_by("weapon_name").len().sort("len", descending=True).head(30))
            print("player_team_code values:", df.group_by("player_team_code").len().sort("player_team_code").rows())
            print("attacker_team_code values:", df.group_by("attacker_team_code").len().sort("attacker_team_code").rows())
            print("hit_box_code distinct count:", df["hit_box_code"].n_unique())
        if ch == "grenade_state":
            for et in ["hegrenade_detonate", "smokegrenade_detonate", "smokegrenade_expired", "flashbang_detonate"]:
                sub = df.filter(pl.col("event_type") == et)
                print(f"{et}: rows={sub.height} null grenade_id={sub['grenade_id'].null_count()} "
                      f"null entity_id_fixed={sub['entity_id_fixed'].null_count()} "
                      f"null player_id_fixed={sub['player_id_fixed'].null_count()} "
                      f"null tick_throw={sub['tick_throw'].null_count()} "
                      f"null x_pos={sub['x_pos'].null_count()}")
            print("grenade_state player_team_code:", df.group_by("player_team_code").len().sort("player_team_code").rows())
            dup = df.filter(pl.col("event_type").is_in(["smokegrenade_detonate", "hegrenade_detonate"])) \
                    .group_by(["match_key", "round", "entity_id_fixed"]).len().filter(pl.col("len") > 1)
            print("detonate (match, round, entity_id_fixed) groups with >1 row:", dup.height)
            ex = df.filter(pl.col("event_type") == "smokegrenade_expired")
            print("expired (match, round, entity_id_fixed) groups with >1 row:",
                  ex.group_by(["match_key", "round", "entity_id_fixed"]).len().filter(pl.col("len") > 1).height)
        if ch == "molotov_state":
            print("molotov event_type x null entity_id_fixed:", df.group_by("event_type").agg(
                pl.len(), pl.col("entity_id_fixed").null_count()).rows())
            print("molotov_state player_team_code:", df.group_by("player_team_code").len().sort("player_team_code").rows())
        if ch == "molotov_fire":
            print("molotov_fire distinct (match, round, entity_id_fixed):",
                  df.select(["match_key", "round", "entity_id_fixed"]).unique().height)
        if ch in ("bomb_action", "bomb_defuse", "bomb_state"):
            for c in ("site_code",):
                if c in df.columns:
                    print(f"{c} value counts:", df.group_by(c).len().sort("len", descending=True).head(8).rows())
            print("player_id raw == 0 count:", df.filter(pl.col("player_id") == 0).height if "player_id" in df.columns else "n/a")
        if ch == "player_blind":
            print("player_blind distinct weapon? n/a")


if __name__ == "__main__":
    main()
