# -*- coding: utf-8 -*-
"""Follow-up checks on combat anomalies. Read-only. Run: pixi run python -I scripts\eda\combat_investigate.py"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CH = {c: os.path.join(TOME, f"basic_{c}.2026-09-18,2026-10-08", "dataframe_00000")
      for c in ["player_death", "player_hurt", "other_death", "bullet_damage", "player_blind", "player_footstep"]}
CH["player_hurt_p1"] = os.path.join(TOME, "basic_player_hurt.2026-09-18,2026-10-08", "dataframe_00001")
h = pq.read_table(os.path.join(TOME, "header", "dataframe_00000"),
                  columns=["key", "map_name", "match_type", "match_date", "max_rounds", "tick_rate", "match_id"]).to_pandas()
h["day"] = h["match_date"].str[:10]
h = h.set_index("key")

def load(path, cols):
    return pq.read_table(path, columns=cols).to_pandas()

def zero_cov(ch):
    p = CH[ch]
    mk = load(p, ["match_key"])["match_key"].unique()
    have = set(mk)
    if ch == "player_hurt":
        have |= set(load(CH["player_hurt_p1"], ["match_key"])["match_key"].unique())
    z = h.assign(has=h.index.isin(have))
    print(f"\n== zero-coverage {ch}: matches with rows {z.has.sum()} / {len(z)}")
    for col in ["map_name", "match_type", "day"]:
        g = z.groupby(col)["has"].agg(["sum", "count"])
        g["missing"] = g["count"] - g["sum"]
        g = g.sort_values("missing", ascending=False)
        print(f"  by {col} (top 8 by missing):")
        print(g.head(8).to_string())
    return z

z_bd = zero_cov("bullet_damage")
zero_cov("other_death")
zero_cov("player_blind")
zero_cov("player_footstep")
zero_cov("player_hurt")
print("\nbullet_damage zero matches: max_rounds dist", z_bd.loc[~z_bd.has, "max_rounds"].value_counts().head(5).to_dict())
print("bullet_damage nonzero matches: max_rounds dist", z_bd.loc[z_bd.has, "max_rounds"].value_counts().head(5).to_dict())

# player_id mapping consistency (player_death)
pdth = load(CH["player_death"], ["match_key", "round", "tick", "player_id", "player_id_fixed", "attacker_id", "attacker_id_fixed", "assister_id", "assister_id_fixed"])
print("\n== player_death id structure")
per_match_ids = pdth.groupby("match_key")["player_id"].nunique()
print("distinct raw player_id per match: ", per_match_ids.describe().round(2).to_dict())
per_match_fixed = pdth.groupby("match_key")["player_id_fixed"].nunique()
print("distinct fixed player_id per match:", per_match_fixed.describe().round(2).to_dict())
m = pdth.dropna(subset=["player_id_fixed"]).groupby(["match_key", "player_id"])["player_id_fixed"].nunique()
print("raw->fixed consistent within match-player (share of pairs with 1 fixed value):", round(float((m == 1).mean()), 4))
ct = pdth.dropna(subset=["player_id_fixed"]).drop_duplicates(["match_key", "player_id"])
ct = ct.groupby(["player_id", "player_id_fixed"]).size()
print("raw player_id=0 maps to fixed:", ct.loc[0].sort_values(ascending=False).head(4).to_dict() if 0 in ct.index.get_level_values(0) else None)
print("raw player_id=1 maps to fixed:", ct.loc[1].sort_values(ascending=False).head(4).to_dict())
print("sample rows (match, raw, fixed):", pdth[["player_id", "player_id_fixed"]].head(8).values.tolist())
print("per-match raw player_id sample sets:", pdth.groupby("match_key")["player_id"].agg(lambda s: sorted(s.unique())).head(3).to_dict())
print("per-match fixed sample sets:", pdth.dropna(subset=["player_id_fixed"]).groupby("match_key")["player_id_fixed"].agg(lambda s: sorted(s.unique())).head(3).to_dict())

asst = pdth["assister_id"]
print("\n== assister encoding: null", int(asst.isna().sum()), " 65535", int((asst == 65535).sum()), " 0", int((asst == 0).sum()), " -1", int((asst == -1).sum()))
print("   assister_id_fixed null", int(pdth.assister_id_fixed.isna().sum()))
print("   attacker_id 65535 rows", int((pdth.attacker_id == 65535).sum()), "; attacker_id null", int(pdth.attacker_id.isna().sum()))

# player_hurt: health_removed and effective
hurt = load(CH["player_hurt"], ["match_key", "round", "tick", "health", "armor", "health_removed", "armor_removed", "effective_health_removed", "weapon_name", "player_id"])
print("\n== player_hurt damage")
print("health_removed > 100:", int((hurt.health_removed > 100).sum()), " > 200:", int((hurt.health_removed > 200).sum()), " max", int(hurt.health_removed.max()))
big = hurt[hurt.health_removed > 100].sort_values("health_removed", ascending=False)
print(big[["match_key", "round", "tick", "weapon_name", "health", "health_removed", "effective_health_removed"]].head(6).assign(match_key=lambda d: d.match_key.str[-36:-0].str[:8]).to_string())
print("weapon_name for health_removed>100:", big.weapon_name.value_counts().head(5).to_dict())
eff = hurt[hurt.effective_health_removed > hurt.health_removed]
print("effective > health_removed:", len(eff), " diff dist:", (eff.effective_health_removed - eff.health_removed).describe().round(2).to_dict())
print(eff[["weapon_name", "health", "health_removed", "effective_health_removed"]].head(5).to_string())
print("health_removed==0 rows:", int((hurt.health_removed == 0).sum()), "effective==0:", int((hurt.effective_health_removed == 0).sum()))
print("hurt health==0 rows and health_removed>0 (should be kills):", int(((hurt.health == 0) & (hurt.health_removed > 0)).sum()))
print("hurt weapon_name empty examples:", int((hurt.weapon_name == "").sum()))

# z and velocity outliers by map (player_death)
pos = load(CH["player_death"], ["match_key", "round", "tick", "player_z_pos", "player_x_vel", "player_y_vel", "player_z_vel"])
pos["map"] = pos.match_key.map(h["map_name"])
hi = pos[pos.player_z_pos.abs() > 10000]
print("\n== z>10000 rows in player_death:", len(hi), " maps:", hi["map"].value_counts().head(6).to_dict(), " matches:", hi.match_key.nunique())
fast = pos[(pos.player_x_vel.abs() > 3000) | (pos.player_y_vel.abs() > 3000) | (pos.player_z_vel.abs() > 3000)]
print("velocity>3000 rows:", len(fast), fast.match_key.nunique(), "matches")
print(fast[["tick", "player_x_vel", "player_y_vel", "player_z_vel"]].head(4).to_string())

# sentinel-ish tick / second coincidence check
for ch in ["player_death", "player_footstep"]:
    d = load(CH[ch], ["match_key", "round", "tick", "second"])
    s = d[d.second == 999]
    print(f"\n{ch} second==999 rows: {len(s)}; tick/64 for these: {(s.tick/64).round(3).unique()[:3]}")
    s2 = d[d.tick.isin([65535, 32767, 9999])]
    print(f"{ch} tick in (65535,32767,9999): {len(s2)} rows; second/tick ratio ok: {bool(np.allclose(s2.second, s2.tick/64, atol=0.02))}")

# entity_id 999 in player_blind
bl = load(CH["player_blind"], ["match_key", "round", "tick", "entity_id", "player_id", "attacker_id"])
print("\n== player_blind entity_id 999:", int((bl.entity_id == 999).sum()), " entity_id dist top:", bl.entity_id.value_counts().head(3).to_dict())
print("   self-blind rows (player==attacker):", int((bl.player_id == bl.attacker_id).sum()), " of", len(bl))
ex = bl[bl.entity_id == 999].head(2)
print(ex.to_string())
