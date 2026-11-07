"""Append the verifier verdict block to logs\\stats\\verify_w1.log (text only; numbers come from the run sections above)."""
LOG = r"G:\pureskill\csds-cs2\logs\stats\verify_w1.log"
lines = [
    "",
    "== VERDICT (independent verifier, W1 headline numbers)",
    "H1 enemy kills 247,629 / ak47 84,922 / ak47 share 34.29 [33.95, 34.66]: CONFIRMED (exact). "
    "Needs team codes filled from player_info (null team codes in player_death) and silencer-off names merged "
    "(m4a1_silencer_off, usp_silencer_off into the silencer names). Raw codes alone give 246,762 enemy kills and ak47 84,678 (34.32).",
    "H2 shots 5,483,781 / ak47 shots 1,678,356 / share 30.61 [30.31, 30.91]: CONFIRMED (exact).",
    "H3 rifle-side round win 81.6 (n=266; CT 134 at 79.9, T 132 at 83.3): CONFIRMED under the definition that the other side has "
    "no rifle, no sniper (AWP/SSG 08) and no other heavy (G3SG1/SCAR-20 counted as heavy). CI [75.9, 86.6] vs report [76.2, 86.5]. "
    "The report does not define the subset; the looser 'other side has no rifle' definition gives n=433 and 76.7 and does not match.",
    "Pistol rounds: round 1 CT 50.0 (144 matches), round 13 CT 42.1 (126): CONFIRMED. CI endpoints differ by up to 0.6 pp.",
    "Holdings (CT/T): rifle 88.9/91.0, SMG 37.5/37.4, sniper 51.7/38.7, pistol-only 43.2/49.4, all pistol-only 2.4/2.8: CONFIRMED. "
    "Other heavy 18.6/8.4 matches only when G3SG1/SCAR-20 count as heavy.",
    "Armor/helmet: pistol-round armor>0 and helmet, and non-pistol class helmet/armor (Table 6, Table 7): CONFIRMED (exact).",
    "Equipment split: rifle side 21.2k vs SMG/pistol side 5.9k over all non-pistol side-rounds; >$5k gap 248 rounds at 83.5, "
    "within/behind 18: CONFIRMED.",
    "Death classes: self 73, team 669, enemy 247,629 confirmed; world 5,101 (report 5,100) and unknown 8 (report 9) off by one row each.",
    "NOT CHECKED: equipment_value_calc definition; health_removed values; W1 notes tables beyond the headline set; Meridian steps.",
]
with open(LOG, "a", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("appended verdict to", LOG)
