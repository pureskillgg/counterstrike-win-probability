# -*- coding: utf-8 -*-
"""Second-pass checks: health_removed meaning, assister encoding, raw player_id per match, entity 999."""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def pth(ch, page="dataframe_00000"):
    return os.path.join(TOME, f"basic_{ch}.2026-09-18,2026-10-08", page)
def load(p, cols=None):
    return pq.read_table(p, columns=cols).to_pandas()

hurt = pd.concat([load(pth("player_hurt"), None), load(pth("player_hurt", "dataframe_00001"), None)], ignore_index=True)
print("hurt rows", len(hurt))
hr = hurt["health_removed"]
print("health_removed quantiles:", hr.quantile([.5, .9, .99, .999]).to_dict(), " max", int(hr.max()))
print("share >100:", round(float((hr > 100).mean()), 4), " share >100 with health==0:", round(float(((hr > 100) & (hurt.health == 0)).sum() / (hr > 100).sum()), 4))
print("health_removed==4999 rows:", int((hr == 4999).sum()), "; values >=1000:", int((hr >= 1000).sum()))
print("top health_removed values:", hr.value_counts().head(8).to_dict())
print("by weapon, share >100:", hurt.assign(b=hr > 100).groupby("weapon_name")["b"].mean().sort_values(ascending=False).head(8).round(3).to_dict())

# prev-health test: sort by match, round, player, tick
h = hurt.sort_values(["match_key", "round", "player_id", "tick"], kind="stable").copy()
h["prev_health"] = h.groupby(["match_key", "round", "player_id"])["health"].shift(1)
h["prev_tick"] = h.groupby(["match_key", "round", "player_id"])["tick"].shift(1)
ok = h.dropna(subset=["prev_health"])
delta = ok["prev_health"] - ok["health"]
print("rows with prev health:", len(ok), " health_removed == prev-health:", round(float((ok.health_removed == delta).mean()), 4),
      " health_removed == prev-health (+-1):", round(float(((ok.health_removed - delta).abs() <= 1).mean()), 4))
big = ok[ok.health_removed > 100].head(4)
print(big[["weapon_name", "prev_health", "health", "health_removed", "effective_health_removed", "prev_tick", "tick"]].to_string())
print("share of >100 rows where prev-health delta agrees within 1:", round(float(((big.health_removed - (big.prev_health - big.health)).abs() <= 1).mean()), 3) if len(big) else None)
over = ok[ok.health_removed > 100]
print("of rows with health_removed>100 and a prev row, share where prev-health delta also >100:", round(float(((over.prev_health - over.health) > 100).mean()), 4), " n", len(over))
print("effective_health_removed max:", int(hurt.effective_health_removed.max()), " >100:", int((hurt.effective_health_removed > 100).sum()))

# assister encoding in player_death
pdth = load(pth("player_death"))
valid = pdth.assister_id.notna() & (pdth.assister_id != 65535)
print("\nassister valid (not null, not 65535):", int(valid.sum()), " of", len(pdth))
print("  assister_x_pos null among valid:", int((valid & pdth.assister_x_pos.isna()).sum()))
print("  assister_x_pos null among 65535/null:", int(((~valid) & pdth.assister_x_pos.isna()).sum()), " non-null:", int(((~valid) & pdth.assister_x_pos.notna()).sum()))
print("  assister_id==0 rows:", int((pdth.assister_id == 0).sum()), " with pos null:", int(((pdth.assister_id == 0) & pdth.assister_x_pos.isna()).sum()))
print("  assister_id == player_id rows:", int((pdth.assister_id == pdth.player_id).sum()))

# raw player_id per match > 10
mc = pdth.groupby("match_key")["player_id"].nunique()
odd = mc[mc > 10]
print("\nmatches with >10 raw player_id values:", len(odd), " examples:", odd.head(3).to_dict())
if len(odd):
    k = odd.index[0]
    print("   id counts for", k[-37:-6], pdth[pdth.match_key == k]["player_id"].value_counts().sort_index().to_dict())
    print("   fixed counts:", pdth[pdth.match_key == k]["player_id_fixed"].value_counts().sort_index().to_dict())

# entity_id near 999 in player_blind
bl = load(pth("player_blind"), ["entity_id"])["entity_id"]
print("\nentity_id counts 997..1001:", {int(v): int((bl == v).sum()) for v in [997, 998, 999, 1000, 1001]})
