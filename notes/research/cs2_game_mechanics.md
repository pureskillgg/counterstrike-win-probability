# CS2 game mechanics: research notes (Task A)

Date: 2026-10-13. Scope: game knowledge only. No data was modified, nothing was written under E:, no packages were installed. Scripts: none needed; data-side checks were grep reads of notes/spec/csds_dictionary.csv.

Evidence labels: VERIFIED = I opened the source (Read or WebFetch) or read the CSV, and the source is named. UNVERIFIED = search snippet, model-written page summary, or memory. "VERIFIED (CS:GO)" means the source is about CS:GO, so it may not hold for CS2.

## 0. Method and access limits

Opened and read:
- arXiv 2109.12990, Xenopoulos, Coelho, Silva, "Optimal Team Economic Decisions in Counter-Strike" (IJCAI AISA workshop 2021). PDF pages 1 to 4 read via the Read tool (saved copy in the session tool-results folder). Data: CS:GO, 6,538 HLTV demofiles, April 2020 to April 2021.
- arXiv 2011.01324, Xenopoulos, Doraiswamy, Silva (2020). Abstract page only, via WebFetch.
- scope.gg CS:GO economy guide (scope.gg/guides/economy_csgo_en/), via WebFetch.
- scope.gg CS2 first-kill article (Premier and FACEIT, since Jan 2026), via WebFetch.
- dust2.us article on the 2026-07-08 CS2 C4 rework, via WebFetch.
- Steam CS2 news feed (store.steampowered.com/feeds/news/app/730), via WebFetch. Model-written summary only; the full item text was not read.

Blocked or failed (so the matching values stay UNVERIFIED): Liquipedia Economy, Defuse_kit and Kevlar_Vest (HTTP 403); Valve news index counter-strike.net/news/updates (placeholder page); counterstrike.fandom.com (HTTP 402); csdb.gg/economy-guide (HTTP 429); sc.bayesesports.com (DNS failure).

## 1. Corrections to the brief

- "24-round halves" is wrong. MR12 means 12 rounds per half, 24 regulation rounds, 13 wins to take the map. VERIFIED (CS:GO) that CS:GO MR15 was 15 rounds per half, 30 max, 16 wins (2109.12990, section 3.1). The CS2 MR12 figures are UNVERIFIED from a primary source. They agree with the local digest (notes/spec/csgo_and_docs_digest.md, section 2.6, DOCUMENTED from docs.pureskill.gg) and with third-party snippets.
- Tomes folder exists at G:\pureskill\csds-cs2\tomes (VERIFIED by directory listing). Raw per-tick files are on E:, not opened here.

## 2. Verified facts

### 2.1 Economy (CS:GO, MR15)
- Round win: $3,250 per player. Bomb detonated or defused: $3,500. VERIFIED (CS:GO; 2109.12990 Figure 1 and section 3.1).
- Loss bonus: $1,400 after one lost round, rising by $500 per consecutive loss to a cap of $3,400. The bonus resets after a win. VERIFIED (CS:GO; 2109.12990 Figure 1).
- Pistol round: $800 per player at the start of each half. VERIFIED on the scope.gg CS:GO page only (CS:GO-oriented page, CS2 not stated).
- Bomb plant: $300 to the planter, and $800 extra to the attacking team if it still loses after the plant. VERIFIED on the scope.gg CS:GO page. Other sources disagree (UNVERIFIED).
- Buy types (CS:GO, 2109.12990 Table 1; round win rate by buy type): Eco 3%, Low buy 27%, Half buy 34%, Hero low buy 25%, Hero half buy 48%, Full buy 59%. Thresholds: Eco = equipment 0 to 3k, spend 0 to 2k; Full buy = equipment plus spend above 20k. VERIFIED (CS:GO, image of Table 1). These are averages that mix team strength with buy choice, so they are not causal effects.
- Winning the first two rounds raised a team's game win probability by roughly 15% on CS:GO Inferno (2109.12990 Figure 2 caption, logistic baseline). VERIFIED (CS:GO).

### 2.2 Model benchmarks (CS:GO)
- Log-loss, weighted average over 7 maps (2109.12990 Table 2): logistic regression 0.793, XGBoost 0.739, neural net 0.736. Using a one-hot map feature gives XGBoost 0.730. VERIFIED (CS:GO).
- 2109.12990 section 2 says the 2020 model of Xenopoulos et al. used XGBoost with inputs such as remaining players and total equipment. VERIFIED for that sentence. Its claim that equipment is the top feature came from a snippet (UNVERIFIED).

### 2.3 CS2 observational: opening kill (scope.gg, Premier and FACEIT since Jan 2026)
- The team with the opening kill wins 64.8% to 72.6% of rounds, depending on map and side. VERIFIED (opened page). The page gives no sample size and no confidence intervals. The effect is observational and confounded.
- Train: CT 71.5% (Premier) and 72.6% (FACEIT); T 65.2% and 64.8%. Mirage: CT 70.4 and 70.8; T 67.2 and 68.5. Dust 2: CT 68.1 and 68.1; T 69.4 and 70.9. Anubis T 68.8 and 70.9. VERIFIED (opened page).

### 2.4 CS2 patch changes that break data regimes
- 2026-07-08 update: C4 explosion range and damage redesigned on all official defusal maps. Numbers not given. VERIFIED (dust2.us page, WebFetch). The article does not mention timers, defuse or money.
- 2026-09-23 update: freeze time, warmup and technical timeout times reduced. No amounts given. UNVERIFIED (Steam feed summary only).

## 3. UNVERIFIED third-party values (CS2 unless noted)

- Armor: kevlar vest $650; helmet add-on $350 (only with the vest); assault suit $1,000. Sources: Liquipedia and bo3.gg search snippets. The three figures are internally consistent. UNVERIFIED.
- Defuse kit: $400, CT only. Defuse 10 s without kit, 5 s with kit. Sources: csdb.gg, thespike.gg, Liquipedia snippets. UNVERIFIED.
- Bomb timer: 40 s in CS2 (mp_c4timer default, third-party). 35 s in CS:GO is VERIFIED (2109.12990 section 3.1).
- Freeze time: 15 s (command reference snippet) against 20 s (another snippet). The local digest, DOCUMENTED, says 20 s Premier and FACEIT, 15 s Competitive, round time 115 s. Conflict remains.
- Grenade prices: smoke $300, flash $200, HE $300, molotov $400 (T only), incendiary $500 (CT; CS:GO was $600), decoy $50. UNVERIFIED.
- Armor math: a 4x headshot base multiplier (English csdb calculator). Helmet cuts AK-47 head damage from 144 to 111 and AWP from 460 to 448 (snippets). The Portuguese and Chinese csdb pages say 2.3x, which conflicts with the English page. UNVERIFIED.
- Other third-party claims: refunds during buy time are new in CS2; force buys are common; M4A4 price cut from $3,100 to $3,000 in May 2024; CS2 smokes are volumetric; sub-tick input timestamps; HLTV Rating 3.0 (Aug 2025) with a Round Swing metric and a 5-second trade window; scope.gg FACEIT side data (Nuke, Mirage and Train CT-leaning; Anubis, Dust 2, Ancient and Inferno T-leaning); csdb pooled pro side data (Mirage CT 55.4% vs T 50.1%; Dust 2 and Inferno about even); clutch rates via a secondary report (1v2 about 19%, 1v3 about 4%, 1v4 about 0.6%, 1v5 under 0.1%); Bayes Esports blog correlation 0.71 between previous-round win and equipment edge (CS:GO, page not opened); pistol-round winners more likely to win matches (Yalcinar and Kilci 2023, abstract snippet). All UNVERIFIED.
- Loss bonus ladder: third-party sources say CS2 kept the CS:GO ladder. UNVERIFIED for CS2.

## 4. Ranked list: factors most likely to move round win probability, strongest first

The ranking is my judgment from the sources above. Effect sizes are UNVERIFIED for CS2 unless noted. Every feature must use only state at or before the prediction tick. round_end, win_reason and later player_death rows must not enter features.

1. Alive players per team (man advantage, 5v4 to 1v3). Belief: large. Source: 2109.12990 section 2 (remaining players as a feature, VERIFIED); the ordering is UNVERIFIED. Measure: count living players per team per tick from player_vector or player_status (rows only for living players), with transitions from player_death (tick, attacker_id, player_id, team via player_info.team_code).
2. Bomb state: planted, defused, time left on the bomb timer, defuse kit present. Belief: large after plant. Source: game rules (UNVERIFIED values). Measure: bomb_state (planted, site_code), bomb_defuse (has_kit, tick), round_end.win_reason_code (1 exploded, 7 defused, per the local digest).
3. Round time remaining (timeout CT wins). Belief: moderate to large. Source: rules; win_reason_code 12 and 13 (digest, DOCUMENTED). Measure: round_state and round_end.second; header freeze_time and round_time_defuse (see section 5).
4. Team equipment value and buy type. Belief: large. Source: 2109.12990 Table 1 (CS:GO, VERIFIED). Measure: player_status.money and freezetime_end_equipment_cost (telemetry; raw E: only); item_equip.item and weapon_type_code. Buy thresholds from CS:GO are a proxy for CS2 (UNVERIFIED).
5. Round history and loss streak. Belief: moderate. Source: 2109.12990 Figure 2 (about 15% after two wins, VERIFIED CS:GO); loss ladder (VERIFIED CS:GO). Measure: round_end.winner_team_code sequence per match, consecutive losses per side.
6. Opening kill. Belief: moderate. Source: scope.gg CS2 2026 (VERIFIED, observational). Measure: first player_death per round by tick, with attacker team.
7. Side (CT or T) and map. Belief: small to moderate per map. Source: scope.gg and csdb (UNVERIFIED). Measure: player_info.team_code at freeze end and round_end.winner_team_code grouped by header.map_name and side.
8. Armor and helmet. Belief: small per round. Source: csdb and steamanalyst snippets (UNVERIFIED). Measure: player_status.armor and has_helmet; player_hurt.armor, armor_removed, health_removed.
9. Weapon tiers (AWP and rifles). Belief: moderate, mostly via alive state. Source: none verified. Measure: item_equip.item and weapon_type_code; player_death.weapon_name.
10. Trades. Belief: small, mostly captured by alive count. Source: HLTV Rating 3.0 snippet (UNVERIFIED). Measure: derive from player_death sequences within a window of a teammate's death; the dictionary has no trade column.
11. Grenades and utility (smoke, flash). Belief: small per round. Source: volumetric smokes (UNVERIFIED). Measure: grenade_state, grenade_vector.grenade_type, player_death.is_flash_assist, is_through_smoke, is_attacker_blind. I did not check player_blind columns.
12. Individual skill and rank. Belief: small to moderate as a prior. Source: digest (DOCUMENTED). Measure: player_info.rank, rank_type, wins (capped at 2501).

## 5. Data-side findings (VERIFIED by reading notes/spec/csds_dictionary.csv)

- player_status (telemetry) has money, armor, health, has_helmet, has_defuser and freezetime_end_equipment_cost. These are in the raw E: files, not in the tomes. Pull only the needed columns when reading E:.
- player_death has attacker_id, assister_id, weapon_name, is_headshot, penetration_amount, is_flash_assist, is_through_smoke, is_attacker_blind, is_noscope and second.
- bomb_defuse has has_kit. bomb_state has site_code. bomb_action has site_code.
- item_equip has item, weapon_type_code. grenade_vector has grenade_type.
- player_hurt has health, armor, health_removed, armor_removed.
- header: max_rounds, freeze_time, round_time_defuse and bomb_time are NOT in csds_dictionary.csv (grep showed no hits). notes/spec/schema_2024_vs_2026.md says the 2026 parquet has them in 40 of 40 sampled folders. UNVERIFIED by me, and the local digest (section 1, item 5) also notes the mismatch.

## 6. What could not be verified

- Primary values for CS2 economy, timers, equipment prices and armor (Liquipedia, Valve and csdb blocked or failed). Everything in section 3 is UNVERIFIED.
- Whether CS2 kept CS:GO's loss-bonus ladder and plant rewards.
- CS2 bomb timer (40 s) and defuse times (10 s and 5 s).
- Freeze time: 15 s or 20 s for CS2 modes, and the amount of the 2026-09-23 reduction.
- Any CS2 win-probability model with calibration results. None was found.
- Effect sizes for trades, clutches, armor, utility, weapon tiers and map side in CS2.
- Sample size and method behind the scope.gg opening-kill numbers.
