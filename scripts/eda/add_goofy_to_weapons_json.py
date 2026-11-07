"""Add the ranked goofy list and the cleared items to notes/eda/weapons.json (data only)."""
import json

PATH = r"G:\pureskill\csds-cs2\notes\eda\weapons.json"

GOOFY_TOP5 = [
    {
        "rank": 1,
        "issue": "Raw ID columns disagree with their *_fixed twins in most rows",
        "channels": ["weapon_fire", "grenade_bounce", "molotov_state", "grenade_state", "molotov_fire"],
        "evidence": "player_id != player_id_fixed: 4,996,539 of 5,483,781 weapon_fire rows (91.1%), "
                    "1,128,757 of 1,236,044 grenade_bounce rows (91.3%), 82,246 of 208,908 molotov_state rows (39.4%). "
                    "entity_id != entity_id_fixed: 531,031/531,031 grenade_state, 208,864/208,908 molotov_state, "
                    "3,277,716/3,278,394 molotov_fire rows.",
        "why_it_misleads": "Raw IDs look like clean small integers, so a model or join on them silently attributes "
                           "events to the wrong player or entity. schema_dictionary.md already says to use _fixed.",
    },
    {
        "rank": 2,
        "issue": "molotov_state encodes 'no player' two ways: -1 and null",
        "channels": ["molotov_state"],
        "evidence": "15,414 rows (7.4%: 14,143 inferno_expire, 1,271 inferno_startburn) have -1 in player_id, "
                    "player_id_fixed, tick_throw and extinguisher_id; burn_duration is -1 on the 14,143 expire rows. "
                    "118,682 rows (56.8%) are null in player_x/y/z_pos, velocities, angles, team code and weapon code.",
        "why_it_misleads": "A tree or linear model reads -1 as a real player ID or tick. Nulls and -1 need different handling.",
    },
    {
        "rank": 3,
        "issue": "weapon_action reload/zoom rows arrive in bursts on consecutive ticks, not one row per event",
        "channels": ["weapon_action"],
        "evidence": "Gap between consecutive reload rows of the same player and round: median 1 tick; 82.3% of "
                    "174,840 gaps are 2 ticks or less. Median reload rows per match = 74. Three matches have "
                    "113 to 132 rows per round against a median of 11.9 (m_889fed2c, m_3abe5e3f and one more); the "
                    "largest single player-round has 846 rows.",
        "why_it_misleads": "schema_dictionary.md says one row per reload or zoom event. Row counts are not event "
                           "counts, so per-round reload or zoom features are inflated. The size of the inflation is not measured here.",
    },
    {
        "rank": 4,
        "issue": "One match has zero rows in all six weapons channels",
        "channels": ["weapon_fire", "weapon_action", "grenade_state", "grenade_bounce", "molotov_state", "molotov_fire"],
        "evidence": "csds/2026/10/07/m_0abf1183afb5ddc1c0197beb/csds has no rows in any of the six channels. "
                    "It is in the header (1,752 keys), so it is not an orphan key.",
        "why_it_misleads": "Either an empty or dropped demo, or a dropped join. A model would silently treat it as a no-weapons round set. "
                           "Not checked against round_end here; the other EDA groups should check it.",
    },
    {
        "rank": 5,
        "issue": "Velocity extremes and exact -1.0 values in weapon_fire, weapon_action and grenade_state",
        "channels": ["weapon_fire", "weapon_action", "grenade_state"],
        "evidence": "|x_vel| or |y_vel| above 2000 in 71 weapon_fire rows (largest |y_vel| 162,514 at round 15, tick 98209); "
                    "|z_vel| above 2000 in 56 rows. Exact -1.0 in 347 to 575 weapon_fire rows per axis. "
                    "Exact 0.0 in 25% of rows for x_vel and 62% for z_vel (plausibly standing players; not treated as a flaw).",
        "why_it_misleads": "Small counts, but huge magnitudes give scale-sensitive models (logistic regression) strong leverage. "
                           "-1.0 is a sentinel candidate for velocity; not confirmed.",
    },
]

CLEARED = [
    "tick 65535, 32767 and 9999 and second 999.0 (tick 63936): flagged as sentinels, but neighbouring tick values have "
    "similar counts (weapon_fire tick 65535: 31 rows, neighbours 40 and 35). Ordinary timestamps.",
    "entity_id 999: a real raw index in 46 to 184 matches. Its entity_id_fixed values are valid "
    "(grenade 10001 to 10616, molotov 1003 to 1193).",
    "player_phi_ang runs 1 to 179 (median 92.5) and player_theta_ang runs -180 to 180 (median 0.4). My +-90 range "
    "for theta was an assumption and is wrong; the angle convention is not stated in schema_dictionary.md.",
    "weapon_name vs player_weapon_code: 2,939 rows (0.054%) disagree with their code's main name. Already documented as noise in schema_dictionary.md.",
    "second equals tick/64 in every row (max difference 0). It is a derived duplicate of tick, not an error.",
    "No exact duplicate rows, no duplicate keys (match, round, tick and the channel key), no identical or constant columns, "
    "no schema drift across pages, no match spanning pages, no match keys outside the header.",
    "No column named health, armor, money, ammo or ping exists in these six channels, so the name-based impossible-value rule had nothing to test.",
    "Row-count outliers: the same match keys top several channels (m_6ba4651d, m_046a770d, m_57108aff). Probably long matches, but round counts were not checked.",
]

CHECKS = [
    "Per channel: rows, pages, page row counts, per-column dtype, null rate, distinct count, min, max, median, top values.",
    "Sentinel counts (4294967295, 65535, 32767, 2147483647, -1, 999, 9999), spikes above 20%, name-based impossible-value rules, range rules.",
    "Identical and near-identical columns (same-signature pairs plus every *_fixed pair), with differing-row counts.",
    "Exact duplicate rows and duplicate keys via 64-bit row hashes (pyarrow streaming, column by column).",
    "Per-match rows against the 1,752 header keys (header column 'key'): zero-row matches, far-from-median top 3, multi-page matches.",
    "Schema drift across pages, second versus tick/64, weapon_name versus player_weapon_code mapping.",
    "Targeted follow-ups: tick neighbours, molotov_state encodings, reload gaps, entity_id 999, weapon_action outliers (logs/eda_weapons_targeted*.txt).",
]


def main():
    d = json.load(open(PATH, encoding="utf-8"))
    d["goofy_top5"] = GOOFY_TOP5
    d["cleared_after_check"] = CLEARED
    d["checks_run"] = CHECKS
    d["not_written"] = "weapons.md was not written: the agent rule forbids report .md files. This JSON and the reply carry the findings."
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    print("ok", len(d["goofy_top5"]), "goofy items;", len(d["cleared_after_check"]), "cleared")


if __name__ == "__main__":
    main()
