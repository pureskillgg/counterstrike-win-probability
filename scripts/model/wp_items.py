"""CS2 item definition indexes used by the loadout features. One place, imported by training (features_status.py) and serving
(snapshot_features.py), so the two cannot drift. Verified against weapon_fire.weapon_name on the 2026 data (7 ak47, 9 awp,
16 m4a1, 60 m4a1_silencer, 61 usp_silencer, 4 glock, ...); inv_primary 0 means no primary weapon."""
RIFLE = (7, 8, 10, 13, 16, 39, 60)       # ak47, aug, famas, galilar, m4a1, sg556, m4a1_silencer
M4 = (16, 60)
AK = (7,)
AWP = (9,)
SNIPER_OTHER = (11, 38, 40)              # g3sg1, scar20, ssg08
SMG = (17, 19, 23, 24, 26, 33, 34)       # mac10, p90, mp5sd, ump45, bizon, mp7, mp9
HEAVY = (14, 25, 27, 28, 29, 35)         # m249, xm1014, mag7, negev, sawedoff, nova
SIDE_T, SIDE_CT = 2, 3

# per-side aggregate names, in the order the model's feature list uses them (suffix _ct / _t is added)
SIDE_FIELDS = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle", "sniper_other", "smg",
               "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip"]
DIFF_FIELDS = ["hp", "rifle", "equip", "armor_n", "awp", "smg"]
