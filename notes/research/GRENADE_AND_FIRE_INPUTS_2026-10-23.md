# Do the papers' models see grenades? (checked 2026-10-23 from full text)

## Szmida & Toka (THGNN, paper 28), text in scratchpad szmida_full.txt

- Raw tracking: "The resulting tracking data contains all state-describing features of the players, including their positions, velocities, weapons, inventories and health points. It also includes grenade trajectories, among other key attributes." (p.1-2)
- Map nodes: "Each stored state-describing properties in its feature vector, such as X, Y and Z coordinates and boolean flags indicating whether it is on fire or smoked." (sec 2.1)
- Map-node node size: each node is sized "approximately the size of a smoke grenade's coverage" (sec 2.1).
- Event datasets: map-specific conditions "like whether an area is burning or smoked" are features (sec 4.2).
- Explanations: incendiary and smoke grenade throws are discussed as actions with SHAP credit (Figures 5.3, 5.4).
- So grenades enter the model as map-state flags (on fire / smoked per node), not as grenade positions or entities.

## Xenopoulos & Silva (2207.14124), text in scratchpad xeno2207.txt

- Player node features (esports): "X/Y/Z coordinates, health and armor remaining, total equipment value and grenades remaining, distance to both bombsites, indicators for player side, if the player is alive, if the player has a helmet and if the player has a defuse kit, and a one hot encoding of the part of the map the player is located." Grenades are a count of what the player holds, not positions and not active smokes or fires.
- Related work: Makarov et al. use "total number of grenades remaining for each team".
- Esports models: one model per map ("we create a different model for each of the six maps"), 70% of game states for training.
- Esports log loss (Table I): State 0.4351, GAT 0.4465, GCN+State 0.4357, GAT+State 0.4276; AUC 0.8715 to 0.8753. Their split is by game state, not by match, so it is not comparable with our frozen match split.

## Our data and our models

- Our tomes hold the full grenade event stream with coordinates: basic_grenade_state (event types smokegrenade_detonate, smokegrenade_expired, flashbang_detonate, hegrenade_detonate, decoy_started, decoy_detonate; x_pos, y_pos, z_pos; thrower). basic_molotov_state (inferno_startburn, inferno_expire, x/y/z). basic_grenade_bounce, basic_molotov_fire.
- So active smoke volumes and fire zones per second can be rebuilt from detonate-to-expire and startburn-to-expire intervals.
- What the models use today: the 59 baseline features hold grenade counts carried in inventory (st_smoke_*, st_fire_*, st_flash_*, st_he_*, st_decoy_*). The round-state arrays (per-slot positions and yaw) hold no grenade data at all. The loadout channels hold has_smoke, has_flash, has_he_or_fire, which are also inventory flags. The temporal models therefore see no active grenades or fires.

## Implication

Our models see neither active grenades nor fires, although we have both in the tomes. Szmida & Toka encode fire and smoke as map-node flags, and Xenopoulos & Silva use only held counts. A candidate per-second feature set: smoke-active and fire-active flags per map zone (or per map-graph node), with age since detonation, and the thrower's side. Build it from basic_grenade_state and basic_molotov_state.
