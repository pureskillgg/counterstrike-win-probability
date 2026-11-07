import glob
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import polars as pl

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/serving")
import win_probability as WP  # noqa: E402

SIDE_NAMES = ["alive", "hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
              "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip", "money"]
NAMES = ["CT_" + x for x in SIDE_NAMES] + ["T_" + x for x in SIDE_NAMES]
ST19 = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle", "sniper_other", "smg",
        "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip"]
DIFF6 = ["hp", "rifle", "equip", "armor_n", "awp", "smg"]
exp = json.load(open(SCR + "/export_sample.json"))
KEYS = list(exp.keys())
TOME = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
PM = "E:/pureskill_tomes/pertick_status_1001_1008/per_match"

files = [f for d in sorted(glob.glob(os.path.join(TOME, "basic_player_info*"))) if os.path.isdir(d)
         for f in sorted(glob.glob(os.path.join(d, "dataframe_*")))]
pinfo = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").filter(pl.col("match_key").is_in(KEYS))
pinfo = pinfo.select("match_key", "round", "player_id", "team_code").filter(pl.col("team_code").is_not_null())
print("player_info rows (sample matches)", pinfo.height, flush=True)

health_nonpositive = 0
status_rows = 0
rounds = 0
fields_bad = Counter()
fields_n = Counter()
examples = []
missing_roster = 0
for key in KEYS:
    st = pl.read_parquet(os.path.join(PM, hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".parquet"))
    status_rows += st.height
    health_nonpositive += st.filter(pl.col("health") <= 0).height
    ros = pinfo.filter(pl.col("match_key") == key)
    rmap = {(int(r[0]), r[1]): int(r[2]) for r in ros.select("round", "player_id", "team_code").iter_rows()}
    freeze_list = [rd["freeze"] for rd in exp[key]["rounds"]]
    fst = st.filter(pl.col("tick").is_in(freeze_list))
    by_round = {}
    for r in fst.iter_rows(named=True):
        by_round.setdefault(int(r["round"]), []).append(r)
    for rd in exp[key]["rounds"]:
        rounds += 1
        rows = [r for r in by_round.get(rd["rn"], []) if r["tick"] == rd["freeze"]]
        players = []
        for r in rows:
            side = rmap.get((int(r["round"]), r["player_id"]))
            if side is None:
                missing_roster += 1
                continue
            players.append({
                "side": side, "health": r["health"], "armor": r["armor"], "has_helmet": r["has_helmet"],
                "has_defuser": r["has_defuser"], "is_defusing": r["is_defusing"], "primary": r["inv_primary"],
                "flashbang": r["inv_flashbang"], "smokegrenade": r["inv_smokegrenade"], "hegrenade": r["inv_hegrenade"],
                "molotov": r["inv_molotov"], "incgrenade": r["inv_incgrenade"], "decoy": r["inv_decoy"],
                "equipment_value": r["equipment_value_calc"],
            })
        out = WP._loadout_from_players(players)
        lo = np.asarray(rd["loadout"], dtype=np.float64)
        for fld in ST19:
            for side, pref in (("ct", "CT"), ("t", "T")):
                fields_n[fld] += 1
                a = out[f"st_{fld}_{side}"]
                b = lo[NAMES.index(f"{pref}_{fld}")]
                if abs(a - b) > 1e-3:
                    fields_bad[fld] += 1
                    if len(examples) < 5:
                        examples.append((key[-14:], rd["rn"], fld, side, a, b))
        for fld in DIFF6:
            fields_n["diff_" + fld] += 1
            a = out[f"st_{fld}_diff"]
            b = lo[NAMES.index("CT_" + fld)] - lo[NAMES.index("T_" + fld)]
            if abs(a - b) > 1e-3:
                fields_bad["diff_" + fld] += 1
                if len(examples) < 5:
                    examples.append((key[-14:], rd["rn"], "diff_" + fld, "-", a, b))

print("matches", len(KEYS), "| freeze rounds checked", rounds, "| status rows (all ticks)", status_rows)
print("status rows with health <= 0 (all ticks, sample matches):", health_nonpositive)
print("freeze-tick status rows with no roster side:", missing_roster)
print("_loadout_from_players vs export loadout (k=0), fields compared:", sum(fields_n.values()), "| mismatches:", sum(fields_bad.values()))
print("mismatches by field:", dict(fields_bad))
for e in examples:
    print("  example", e)
