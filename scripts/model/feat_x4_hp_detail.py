"""x4_hp_detail: per-player hit points of the living, split by side (beyond the team total of hp_features).

  compute(G, store) -> polars DataFrame, one row per grid row, same order, only these columns (suffix _ct / _t):
    hpp_le40_<s>      living players of the side with health <= 40
    hpp_eq100_<s>     living players with health == 100 (no health lost; armor-only hits keep health at 100)
    hpp_min_<s>       lowest health among the living (NaN when nobody of the side is alive)
    hpp_max_<s>       highest health among the living (NaN when nobody is alive)
    hpp_sd_<s>        population standard deviation of health among the living (0 for one living player, NaN for none)
    hpp_arm_n_<s>     living players whose armor is known (hit at least once in this round)
    hpp_arm_mean_<s>  mean armor of those players (NaN when none)
    hpp_arm_min_<s>   lowest armor of those players (NaN when none)

Causality: a player's health at tick t is the health reported by his last player_hurt row with tick <= t of the same round
(100 before any hit). A player is dead from the first player_death or health-0 player_hurt row of the round on. Armor is the
armor reported by that same last hit; purchases after it are not seen, so it can be stale. The roster of a round is its
player_info snapshot, plus players who appear only in player_hurt rows (as victim or attacker): they count as living from their
first team-coded hurt row on (side = that row's team code). Per side, the rows where the reconstructed living count differs from
the grid's alive count of that side (roster gaps, missed death events) are set to NaN for the per-player columns: the identity
of those players is unknown there.
"""
import numpy as np
import polars as pl

from features_events import BIG, CT, T, _round_ids, scan_events

NEVER = np.int64(1) << 60  # tick sentinel: a player never seen dying / joining in the round
SIDES = ((CT, "ct"), (T, "t"))
ALIVE_COL = {"ct": "alive_ct", "t": "alive_t"}


def _roster(store, keys):
    """One row per (round, player) with side (2 = T, 3 = CT), enter (tick from which the player counts as in the round:
    -1 for the player_info snapshot, else the tick of his first team-coded player_hurt row as victim or attacker) and slot =
    position of the player among the roster of the round (by player_id), so 2024 raw ids (65280 and up) and 2026 ids (0..9)
    are handled alike."""
    pi = (scan_events(store, "player_info", ["match_key", "round", "player_id", "team_code"])
          .join(keys, on=["match_key", "round"], how="inner")
          .filter(pl.col("team_code").is_in([T, CT]))
          .select("rid", "player_id", pl.col("team_code").alias("side"), pl.lit(-1).cast(pl.Int64).alias("enter")))
    pi_keys = pi.select("rid", "player_id")
    victims = (scan_events(store, "player_hurt", ["match_key", "round", "tick", "player_id", "player_team_code"])
               .join(keys, on=["match_key", "round"], how="inner")
               .filter(pl.col("player_team_code").is_in([T, CT]))
               .select("rid", "player_id", pl.col("player_team_code").alias("side"), "tick"))
    attackers = (scan_events(store, "player_hurt", ["match_key", "round", "tick", "attacker_id", "attacker_team_code"])
                 .join(keys, on=["match_key", "round"], how="inner")
                 .filter(pl.col("attacker_team_code").is_in([T, CT]))
                 .select("rid", pl.col("attacker_id").alias("player_id"), pl.col("attacker_team_code").alias("side"), "tick"))
    seen = (pl.concat([victims, attackers])
            .group_by(["rid", "player_id"])
            .agg(pl.col("tick").min().alias("enter"), pl.col("side").sort_by("tick").first().alias("side"))
            .join(pi_keys, on=["rid", "player_id"], how="anti")
            .select("rid", "player_id", "side", "enter"))
    return (pl.concat([pi, seen])
            .sort(["rid", "player_id"])
            .with_columns((pl.col("player_id").rank("ordinal").over("rid").cast(pl.Int64) - 1).alias("slot")))


def _hurt(store, keys, roster):
    """player_hurt of rostered players, one row per (slot, round, tick): health and armor after the tick's last hit."""
    return (scan_events(store, "player_hurt", ["match_key", "round", "tick", "player_id", "health", "armor"])
            .join(keys, on=["match_key", "round"], how="inner")
            .join(roster.select("rid", "player_id", "slot"), on=["rid", "player_id"], how="inner")
            .group_by(["slot", "rid", "tick"])
            .agg(pl.col("health").min(), pl.col("armor").min())  # hit points only fall within a round: min = value after the last hit
            .sort(["slot", "rid", "tick"]))


def _death_ticks(store, keys, roster, hurt):
    """First death tick per (round, slot): player_death rows, or a health-0 hurt row when the death event is missing."""
    d = (scan_events(store, "player_death", ["match_key", "round", "tick", "player_id"])
         .join(keys, on=["match_key", "round"], how="inner")
         .join(roster.select("rid", "player_id", "slot"), on=["rid", "player_id"], how="inner")
         .select("rid", "slot", "tick"))
    z = hurt.filter(pl.col("health") == 0).select("rid", "slot", "tick")
    return pl.concat([d, z]).group_by(["rid", "slot"]).agg(pl.col("tick").min().alias("dead_tick"))


def _per_side(G, store):
    """Returns (feature dict of numpy arrays, reconstructed living count per side name, reconciled flag per side name)."""
    keys, rid_g = _round_ids(G)
    n_rounds = keys.height
    n = G.height
    roster = _roster(store, keys)
    n_slots = int(roster["slot"].max()) + 1

    rid_r = roster["rid"].to_numpy().astype(np.int64)
    slot_r = roster["slot"].to_numpy().astype(np.int64)
    side_tab = np.zeros((n_rounds, n_slots), dtype=np.int8)
    side_tab[rid_r, slot_r] = roster["side"].to_numpy().astype(np.int8)
    enter_tab = np.full((n_rounds, n_slots), NEVER, dtype=np.int64)
    enter_tab[rid_r, slot_r] = roster["enter"].to_numpy().astype(np.int64)

    hurt = _hurt(store, keys, roster)
    dt = _death_ticks(store, keys, roster, hurt)
    dead = np.full((n_rounds, n_slots), NEVER, dtype=np.int64)
    dead[dt["rid"].to_numpy().astype(np.int64), dt["slot"].to_numpy().astype(np.int64)] = dt["dead_tick"].to_numpy().astype(np.int64)

    hslot = hurt["slot"].to_numpy().astype(np.int64)
    hk = hurt["rid"].to_numpy().astype(np.int64) * BIG + hurt["tick"].to_numpy().astype(np.int64)
    hh = hurt["health"].to_numpy().astype(np.float32)
    ha = hurt["armor"].to_numpy().astype(np.float32)

    rid64 = rid_g.astype(np.int64)
    base = rid64 * BIG
    tick_g = G["tick"].to_numpy().astype(np.int64)
    q = base + tick_g

    nan32 = np.float32(np.nan)
    zero32 = np.float32(0.0)
    acc = {}
    for _, name in SIDES:  # health and armor are integers: float32 sums are exact
        acc[name] = dict(cnt=np.zeros(n, np.int16), le40=np.zeros(n, np.int16), eq100=np.zeros(n, np.int16),
                         hmin=np.full(n, np.inf, np.float32), hmax=np.full(n, -np.inf, np.float32),
                         s1=np.zeros(n, np.float32), s2=np.zeros(n, np.float32),
                         arm_n=np.zeros(n, np.int16), arm_sum=np.zeros(n, np.float32),
                         arm_min=np.full(n, np.inf, np.float32))

    for s in range(n_slots):
        lo = np.searchsorted(hslot, s, side="left")
        hi = np.searchsorted(hslot, s, side="right")
        if hi > lo:
            ks, hs, as_ = hk[lo:hi], hh[lo:hi], ha[lo:hi]
            idx = np.searchsorted(ks, q, side="right") - 1  # last hurt row of this slot with key <= (round, tick) of the row
            idxc = np.maximum(idx, 0)
            ok = (idx >= 0) & (ks[idxc] >= base)  # same round only
            h = np.where(ok, hs[idxc], np.float32(100.0)).astype(np.float32)
            a = np.where(ok, as_[idxc], nan32).astype(np.float32)
        else:
            h = np.full(n, 100.0, np.float32)
            a = np.full(n, nan32, np.float32)
        side_g = side_tab[rid64, s]
        alive = (tick_g >= enter_tab[rid64, s]) & (tick_g < dead[rid64, s])  # in the round from enter on, dead from a death tick on
        for side, name in SIDES:
            m = alive & (side_g == side)
            A = acc[name]
            A["cnt"] += m
            A["le40"] += m & (h <= 40)
            A["eq100"] += m & (h >= 100)
            A["hmin"] = np.minimum(A["hmin"], np.where(m, h, np.float32(np.inf)))
            A["hmax"] = np.maximum(A["hmax"], np.where(m, h, np.float32(-np.inf)))
            A["s1"] += np.where(m, h, zero32)
            A["s2"] += np.where(m, h * h, zero32)
            known = m & ~np.isnan(a)
            A["arm_n"] += known
            A["arm_sum"] += np.where(known, a, zero32)
            A["arm_min"] = np.minimum(A["arm_min"], np.where(known, a, np.float32(np.inf)))

    feats, counts, reconciled = {}, {}, {}
    for _, name in SIDES:
        A = acc[name]
        cnt = A["cnt"].astype(np.int64)
        reconciled[name] = cnt == G[ALIVE_COL[name]].to_numpy().astype(np.int64)
        keep = reconciled[name]  # rows where the identity of the living is unknown become NaN
        has = (cnt > 0) & keep
        arm_n = A["arm_n"].astype(np.int64)
        arm_has = (arm_n > 0) & keep
        c = np.maximum(cnt, 1).astype(np.float64)
        mean = A["s1"].astype(np.float64) / c
        sd = np.sqrt(np.maximum(A["s2"].astype(np.float64) / c - mean ** 2, 0.0))
        ca = np.maximum(arm_n, 1).astype(np.float64)
        feats[f"hpp_le40_{name}"] = np.where(keep, A["le40"].astype(np.float64), np.nan).astype(np.float32)
        feats[f"hpp_eq100_{name}"] = np.where(keep, A["eq100"].astype(np.float64), np.nan).astype(np.float32)
        feats[f"hpp_min_{name}"] = np.where(has, A["hmin"], nan32).astype(np.float32)
        feats[f"hpp_max_{name}"] = np.where(has, A["hmax"], nan32).astype(np.float32)
        feats[f"hpp_sd_{name}"] = np.where(has, sd, np.nan).astype(np.float32)
        feats[f"hpp_arm_n_{name}"] = np.where(keep, arm_n.astype(np.float64), np.nan).astype(np.float32)
        feats[f"hpp_arm_mean_{name}"] = np.where(arm_has, A["arm_sum"].astype(np.float64) / ca, np.nan).astype(np.float32)
        feats[f"hpp_arm_min_{name}"] = np.where(arm_has, A["arm_min"], nan32).astype(np.float32)
        counts[name] = cnt
    return feats, counts, reconciled


def compute(G, store):
    feats, _, _ = _per_side(G, store)
    return pl.DataFrame(feats)
