"""Pure-Python loadout features for serving: the same numbers features_status.py builds in polars for training.

    players = [{"side": "CT", "health": 100, "armor": 100, "has_helmet": True, "has_defuser": True, "is_defusing": False,
                "primary": 16, "flashbang": 1, "smokegrenade": 0, "hegrenade": 1, "molotov": 0, "incgrenade": 0, "decoy": 0,
                "equipment_value": 5200}, ...]     # ALIVE players only, one dict each, as in the live game state
    loadout_from_players(players) -> {"st_hp_ct": ..., ..., "st_smg_diff": ...}      (46 numbers)

Missing player fields default to 0 / False (a player without a recorded primary is "no primary"); an unrecognised side raises
ValueError, which the predictor turns into an invalid-input fallback. No numpy or polars needed.
"""
from wp_items import AK, AWP, DIFF_FIELDS, HEAVY, M4, RIFLE, SIDE_CT, SIDE_FIELDS, SIDE_T, SMG, SNIPER_OTHER

_SIDE = {"ct": SIDE_CT, "t": SIDE_T, "3": SIDE_CT, "2": SIDE_T, 3: SIDE_CT, 2: SIDE_T}


def _side(v):
    key = v.strip().lower() if isinstance(v, str) else v
    if key not in _SIDE:
        raise ValueError(f"unknown side {v!r}")
    return _SIDE[key]


def _n(p, k):
    v = p.get(k, 0)
    if isinstance(v, bool):
        return int(v)
    if v is None:
        return 0
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite {k}")
    return f


def loadout_from_players(players):
    agg = {SIDE_CT: dict.fromkeys(SIDE_FIELDS, 0.0), SIDE_T: dict.fromkeys(SIDE_FIELDS, 0.0)}
    for p in players:
        a = agg[_side(p.get("side"))]
        prim = int(_n(p, "primary"))
        a["hp"] += _n(p, "health")
        a["armor_n"] += 1 if _n(p, "armor") > 0 else 0
        a["armor_sum"] += _n(p, "armor")
        a["helmet_n"] += 1 if _n(p, "has_helmet") else 0
        a["kit_n"] += 1 if _n(p, "has_defuser") else 0
        a["defusing"] += 1 if _n(p, "is_defusing") else 0
        a["ak"] += prim in AK
        a["m4"] += prim in M4
        a["awp"] += prim in AWP
        a["rifle"] += prim in RIFLE
        a["sniper_other"] += prim in SNIPER_OTHER
        a["smg"] += prim in SMG
        a["heavy"] += prim in HEAVY
        a["no_primary"] += prim == 0
        a["flash"] += _n(p, "flashbang")
        a["smoke"] += _n(p, "smokegrenade")
        a["he"] += _n(p, "hegrenade")
        a["fire"] += _n(p, "molotov") + _n(p, "incgrenade")
        a["decoy"] += _n(p, "decoy")
        a["equip"] += _n(p, "equipment_value")
    out = {}
    for name in SIDE_FIELDS:
        out[f"st_{name}_ct"] = float(agg[SIDE_CT][name])
        out[f"st_{name}_t"] = float(agg[SIDE_T][name])
    for name in DIFF_FIELDS:
        out[f"st_{name}_diff"] = out[f"st_{name}_ct"] - out[f"st_{name}_t"]
    return out
