"""W1: assemble notes/stats/W1.md from the three W1 logs (tables are copied verbatim from the logs)."""
from pathlib import Path

ROOT = Path("G:/pureskill/csds-cs2")
LOGS = ROOT / "logs" / "stats"
OUT = ROOT / "notes" / "stats" / "W1.md"

HEADER = """# W1: weapon statistics, freeze-end holdings and round outcomes (2026 tome)

Agent: W1. Repo root: G:\\pureskill\\csds-cs2. Interpreter: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe (-I -X utf8).

## Scope and sets used

- Kills and shots (parts 1 and 2): all 1,752 matches in the 2026 tome (`tomes/2026-10-01_to_08/tome/csds`). The bake-off kept set (1,521) is not applied. Shots are weapon_fire events, not hits.
- Freeze-end holdings and round outcomes (parts 3 and 4): the 1 Hz sample, 150 matches (120 train, 30 test; pooled, no model selection), 2,673 match-rounds.
- Pistol rounds: round 1 and round 13 (checked against round_buy_types.is_pistol, zero disagreements). Non-pistol rounds: 2,403 (50 of them are overtime rounds above 24, none inside the rifle-vs-SMG comparison).
- Armor (supplementary): player_hurt (all 1,751 matches with hits) for per-hit armor by weapon; freeze-end armor and helmet from the 1 Hz sample.

## Definitions

- World death: weapon_name in {world, worldent, trigger_hurt, planted_c4} or null attacker. Attacker id 0 is a valid player slot (its kills have matching attacker weapon codes), so it is not treated as world.
- Team death: attacker and victim on the same side (team_code 2 = T, 3 = CT), attacker not the victim. Self kills (attacker = victim, non-world) are reported separately.
- Attacker side: attacker_team_code; when null, filled from the same match, round and player id using victim, attacker and weapon_fire team codes (no conflicting codes). Of 507 non-world deaths with a null attacker side, 500 were filled and 7 stayed unknown. Nine deaths with a side still unknown after the lookup are excluded as unknown_team.
- Enemy kill: opposite sides, both known, not world. Denominator for kill shares is enemy kills (247,629).
- Silencer-off labels (m4a1_silencer_off, usp_silencer_off) are merged into m4a1_silencer and usp_silencer for kills.
- Freeze-end snapshot: every player row at the minimum tick of each (match_key, round); seconds_since_live = 0 in all of them.
- Holding classes (from inv_primary at freeze-end): RIFLE = AK-47, M4A4, M4A1-S, Galil AR, FAMAS, AUG, SG 553; SMG = MP9, MAC-10, MP7, P90, MP5-SD, UMP-45, PP-Bizon; SNIPER = AWP, SSG 08; HEAVY = G3SG1, SCAR-20, M249, Negev, XM1014, MAG-7, Nova, Sawed-Off; NONE = no primary (pistol-only). "At least one player" means the flag is 1 if any player on that side holds that class.
- Side class: RIFLE if the side has a rifle; SNIPER_HEAVY if no rifle but a sniper or heavy; SMG_PISTOL otherwise (every player is SMG or pistol-only).
- Win rate comparison: non-pistol rounds where exactly one side is RIFLE and the other is SMG_PISTOL. Rifle side = the side with at least one rifle.
- Intervals: 95% percentile intervals from a match-cluster bootstrap (matches resampled with replacement, rounds not resampled), 1000 resamples, numpy default_rng(0). One resample matrix per script, reused across tables. Logistic coefficient intervals are bootstrap percentiles of refitted models; SEs and p-values are from the point-estimate Hessian (Wald).
- Logistic regression: unpenalized IRLS written by hand (statsmodels is not in the project env, and numpy matrix products crash in this Windows build, so the bootstrap uses element-wise sums). eq_k = team equipment_value_calc difference in $1,000; nr_diff = rifle-holder count difference; rifle_ind_diff = rifle-holding indicator difference. Model B is the primary model for "equipment and rifle count controlled" (all non-pistol rounds, CT perspective).

## Mapping used

- weapon_code to weapon_name (weapon_fire): codes are CS item definition indices. The modal name per code is used as the reference (Table 3 below). Rows with a null code: 14,853. Rows where weapon_name differs from the modal name for a non-null code: 2,939 (0.054%), matching the known caveat. Code 49 has only 68 rows and its modal name (ak47, 19 rows) is not a reliable mapping; weapon_name is used for all weapon labels, not weapon_code.
- inv_primary (1 Hz) to class: Table 0 in the holdings appendix. All 25 codes present at freeze-end are mapped; all 24 non-zero codes match their weapon_fire modal names. Code 0 = no primary.
- weapon_fire also has knife, grenade, decoy and taser names (for example knife_t 120,321 events, flashbang 138,784). "Shots" in Table 2 therefore include these; they are not gun shots, and gun-only counts would be a subset that this run did not separate.

## Headline numbers (all cited values are in the appendices below)

- 253,480 death events: 247,629 enemy kills, 5,100 world, 669 team, 73 self, 9 unknown side.
- Top enemy-kill weapons: ak47 34.3% [33.95, 34.66] (84,922 kills; CT side 24.6%), m4a1_silencer 15.8% [15.47, 16.04] (CT side 92.4%), awp 9.3% [9.00, 9.54] (CT side 62.9%), m4a1 6.8%, usp_silencer 5.1%, glock 4.4% (CT side 1.4%). Knives: 396 kills (0.16%).
- Shots: 5,483,781 weapon_fire events; ak47 30.6%, m4a1_silencer 13.0%, m4a1 6.5%, glock 4.2%.
- Freeze-end holdings, non-pistol side-rounds (2,403 per side): at least one rifle CT 88.9% [87.7, 90.1], T 91.0% [89.8, 92.2]; at least one sniper CT 51.7% [48.4, 55.1], T 38.7% [35.5, 41.9]; at least one pistol-only player CT 43.2% [40.5, 45.9], T 49.4% [46.9, 51.9]; at least one SMG 37.5% (CT) and 37.4% (T).
- Win rate, rifle side vs SMG/pistol-only side: 81.6% [76.2, 86.5] over 266 rounds (125 matches). Across all non-pistol side-rounds, RIFLE-class sides average $21.2k team equipment vs $5.9k for SMG_PISTOL-class sides (Table 4b), and in 248 of the 266 comparison rounds the rifle side is more than $5k ahead (Table 4c), so this raw rate is strongly equipment-confounded.
- Equipment-controlled regression (all 2,403 non-pistol rounds, CT perspective): equipment difference beta 0.0622 per $1k (OR 1.064 [1.051, 1.079]); rifle-holder count difference beta 0.0465 per rifle (OR 1.048 [0.993, 1.110], p = 0.13). With the rifle indicator instead of the count: equipment beta 0.0597 per $1k; rifle indicator beta 0.4435 (OR 1.56 [1.18, 2.11]), against OR 3.40 without the equipment control. So equipment is controlled for, and about 64% of the unadjusted rifle-indicator effect on the log-odds scale (1 - 0.4435/1.2245) is explained by equipment.
- Pistol rounds: CT won 50.0% [41.6, 57.9] of round 1 (144 matches) and 42.1% [33.6, 50.7] of round 13 (126 matches). No helmets at freeze-end in either pistol round; armor > 0 in 43 to 49% of players.
- Armor per enemy hit: 80% of ak47 hits and 83% of m4a1 hits are absorbed in part by armor; 0% for inferno (fire). Player-level freeze-end (non-pistol): helmet 76.1%, armor > 0 86.0%; no-primary players helmet 30.0%.

## Not checked or caveats

- equipment_value_calc is used as given; its definition is not in the repo (not verified against the game's economy).
- health_removed (player_hurt) is used as recorded. Values above 100 occur (headshots, overkill), the field maximum is 4,999 and was not verified; means are sensitive to outliers. effective_health_removed (max 446) was not used.
- Duplicate player_hurt keys (11,981 rows; mostly xm1014, mag7, nova) were kept as multi-pellet hits; not verified against the demo.
- One team-round at freeze-end has 4 players in the sample (equipment sum is lower for that round).
- One non-pistol round (match csds/2026/10/07/m_partial..., round 16) has no round_buy_types row; it is classed by round number.
- Six sample matches start after round 1 and 18 end before round 13 (1 Hz coverage; pistol counts are 144 and 126 matches, not 150).
- The 1 Hz sample is not the 1,521-match kept set. Holdings and win rates are descriptive and conditional on the freeze-end snapshot, not causal.
- The "meridian checkpoint / provenance update" in the relayed request was not performed: the meridian-outputs MCP server needs OAuth (authorize via /mcp), and it was outside this task's write scope. No provenance was recorded.

## Appendix A. Kills, shots and code mapping (log: logs/stats/W1_weapon_kills_shots.log)

"""

FOOTER_B = "\n\n## Appendix B. Holdings, win rates, regression and pistol/armor tables (log: logs/stats/W1_freeze_holdings.log)\n\n"
FOOTER_C = "\n\n## Appendix C. Armor and damage per hit by weapon (log: logs/stats/W1_hurt_armor.log)\n\n"


def shift(text):
    lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            lines.append("#" + line)
        elif line.startswith("# "):
            lines.append("##" + line[1:])
        else:
            lines.append(line)
    return "\n".join(lines)


def main():
    a = (LOGS / "W1_weapon_kills_shots.log").read_text(encoding="utf8")
    b = (LOGS / "W1_freeze_holdings.log").read_text(encoding="utf8")
    c = (LOGS / "W1_hurt_armor.log").read_text(encoding="utf8")
    doc = HEADER + shift(a) + FOOTER_B + shift(b) + FOOTER_C + shift(c) + "\n"
    OUT.write_text(doc, encoding="utf8")
    print(f"wrote {OUT} ({len(doc)} chars)")


if __name__ == "__main__":
    main()
