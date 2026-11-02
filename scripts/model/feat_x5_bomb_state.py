"""x5_bomb_state: bomb plant and defuse state, causal (a row uses only events of its own round with tick <= the row tick).

  compute(G, store) -> polars DataFrame, one row per grid row, same order, only these float32 columns (NaN = unknown):
    defuse_active_ct     1 while a defuse attempt is in progress, else 0
    defuse_elapsed_s_ct  seconds since the attempt in progress began (NaN when no attempt is in progress)
    defuse_attempts_ct   defuse attempts begun so far in the round (0 before the first)
    defuse_kit_ct        1 if the attempt in progress uses a defuser kit, 0 if not (NaN when no attempt is in progress)
    plant_elapsed_s_t    seconds since the bomb was planted (NaN before the plant)
    plant_site_t         bombsite of the plant, 0 or 1 within the map (NaN before the plant or when the site is unknown)

Sources: bomb_defuse (bomb_begin_defuse, and bomb_abort_defuse where a store has such rows) and bomb_state (bomb_planted,
bomb_defused, bomb_exploded).

Defuse in progress. The combined tomes record no bomb_abort_defuse rows, so an attempt cannot be seen to stop when a player
lets go. An attempt therefore counts as in progress from its begin event until the earliest of: an end event (bomb_defused,
bomb_exploded, or an abort row if present) at or before the row tick, or the defuse time (5 s with a kit, 10 s without),
which stands in for the unrecorded abort. Begin rows repeat while a defuse is held, so a begin opens a new attempt only when it
comes more than the defuse time after the previous begin of the round; identical begin rows (same tick and player) count once.

Bombsite. The site_code value is an entity index that changes from match to match (a map shows 80 to 150 distinct values
over the store), so a per-map ranking of site codes is not defined. Instead the planter's position at the plant event
(same tick as the event, so causal) is assigned to the nearer of two bombsite centres of its map. The centres (SITE_CENTRES)
are static map geometry: 2D k-means over the planted positions of the store, with the label 0 for the centre with the smaller
x. Labels are consistent within a map; which label is site A is not known.
"""
import numpy as np
import polars as pl

from features_events import BIG, _round_ids, scan_events

TPS = 64  # ticks per second
KIT_TICKS = 5 * TPS  # defuse time with a kit
NOKIT_TICKS = 10 * TPS  # defuse time without a kit
NEVER = np.int64(1) << 60  # tick sentinel: no plant in the round

SITE_CENTRES = {  # map -> ((x, y) of label 0, (x, y) of label 1)
    "de_ancient": ((-1355.0, 777.0), (840.0, 18.0)),
    "de_anubis": ((-1099.0, 652.0), (1227.0, 1919.0)),
    "de_cache": ((-112.0, -1254.0), (-92.0, 1841.0)),
    "de_dust2": ((-1565.0, 2567.0), (1117.0, 2520.0)),
    "de_inferno": ((426.0, 2815.0), (2016.0, 295.0)),
    "de_mirage": ((-1992.0, 328.0), (-409.0, -2110.0)),
    "de_nuke": ((640.0, -727.0), (646.0, -1222.0)),
}


def _last_at_or_before(sorted_keys, q, base):
    """Index of the last key <= q, and whether that key belongs to the same round (key >= base)."""
    if sorted_keys.size == 0:
        return np.zeros(q.size, dtype=np.int64), np.zeros(q.size, dtype=bool)
    i = np.searchsorted(sorted_keys, q, side="right") - 1
    ic = np.maximum(i, 0)
    return ic, (i >= 0) & (sorted_keys[ic] >= base)


def _site_labels(maps, x, y):
    out = np.full(len(maps), np.nan)
    for i, m in enumerate(maps):
        c = SITE_CENTRES.get(m)
        if c is None:
            continue
        d0 = (x[i] - c[0][0]) ** 2 + (y[i] - c[0][1]) ** 2
        d1 = (x[i] - c[1][0]) ** 2 + (y[i] - c[1][1]) ** 2
        out[i] = 1.0 if d1 < d0 else 0.0
    return out


def compute(G, store):
    keys, rid_g = _round_ids(G)
    n = G.height
    R = keys.height
    rid = rid_g.astype(np.int64)
    tick_g = G["tick"].to_numpy().astype(np.int64)
    q = rid * BIG + tick_g
    base = rid * BIG

    # ---- defuse attempts (CT side)
    D = (scan_events(store, "bomb_defuse", ["match_key", "round", "tick", "event_type", "player_id", "has_kit"])
         .join(keys, on=["match_key", "round"], how="inner")
         .filter(pl.col("event_type").is_in(["bomb_begin_defuse", "bomb_abort_defuse"])))
    begins = (D.filter(pl.col("event_type") == "bomb_begin_defuse")
              .with_columns(pl.col("has_kit").fill_null(False).cast(pl.Int64).alias("kit"))
              .unique(["rid", "tick", "player_id"], keep="first")
              .sort(["rid", "tick"]))
    bk = begins["rid"].to_numpy().astype(np.int64) * BIG + begins["tick"].to_numpy().astype(np.int64)
    bt = begins["tick"].to_numpy().astype(np.int64)
    bkit = begins["kit"].to_numpy().astype(np.int64)
    m = bk.size
    rid_b = bk // BIG
    same_prev = np.zeros(m, dtype=bool)
    same_prev[1:] = rid_b[1:] == rid_b[:-1]
    prev_dur = np.zeros(m, dtype=np.int64)
    prev_dur[1:] = np.where(bkit[:-1] == 1, KIT_TICKS, NOKIT_TICKS)
    gap = np.zeros(m, dtype=np.int64)
    gap[1:] = bt[1:] - bt[:-1]
    # begin rows keep coming while a defuse is held: a begin opens a new attempt only when it comes more than the defuse
    # time after the previous begin of the round (the first begin of a round always opens one)
    is_start = ~same_prev | (gap > prev_dur)
    cs = np.concatenate([[0], np.cumsum(is_start)])
    first_idx = np.searchsorted(bk, base, side="left")
    end_idx = np.searchsorted(bk, q, side="right")
    attempts = cs[end_idx] - cs[first_idx]  # attempts opened at or before the row tick, same round
    bi, has_b = _last_at_or_before(bk, q, base)  # last begin at or before the row tick
    if m:
        start_acc = np.maximum.accumulate(np.where(is_start, np.arange(m), 0))
        si = start_acc[bi]  # begin index that opened the attempt the last begin belongs to
        s_tick = bt[si]
        kit_s = bkit[si]
        b_tick = bt[bi]
        kit_last = bkit[bi]
        bkey_last = bk[bi]
    else:
        s_tick = np.zeros(n, np.int64)
        kit_s = np.zeros(n, np.int64)
        b_tick = np.zeros(n, np.int64)
        kit_last = np.zeros(n, np.int64)
        bkey_last = np.zeros(n, np.int64)

    S = (scan_events(store, "bomb_state", ["match_key", "round", "tick", "event_type", "player_x_pos", "player_y_pos"])
         .join(keys, on=["match_key", "round"], how="inner"))
    ends = pl.concat([
        D.filter(pl.col("event_type") == "bomb_abort_defuse").select("rid", "tick"),
        S.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).select("rid", "tick"),
    ]).sort(["rid", "tick"])
    ek = ends["rid"].to_numpy().astype(np.int64) * BIG + ends["tick"].to_numpy().astype(np.int64)
    ei, e_ok = _last_at_or_before(ek, q, base)
    ekey_last = ek[ei] if ek.size else np.zeros(n, np.int64)
    ended = e_ok & has_b & (ekey_last >= bkey_last)  # an end event at or after the last begin
    dur = np.where(kit_last == 1, KIT_TICKS, NOKIT_TICKS)
    active = has_b & ~ended & (tick_g - b_tick <= dur)  # the latest begin is recent and no end event follows it
    defuse_elapsed = np.where(active, (tick_g - s_tick) / TPS, np.nan)  # time since the attempt opened
    defuse_kit = np.where(active, kit_s, np.nan)

    # ---- plant and bombsite (T side)
    P = S.filter(pl.col("event_type") == "bomb_planted")
    plant_r = np.full(R, NEVER, np.int64)
    fp = P.group_by("rid").agg(pl.col("tick").min().alias("plant_tick"))
    plant_r[fp["rid"].to_numpy().astype(np.int64)] = fp["plant_tick"].to_numpy().astype(np.int64)
    site_tick_r = np.full(R, NEVER, np.int64)
    site_lab_r = np.full(R, np.nan)
    site_src = (P.drop_nulls(["player_x_pos", "player_y_pos"]).sort(["rid", "tick"])
                .group_by("rid", maintain_order=True)
                .agg(pl.col("tick").first().alias("site_tick"), pl.col("player_x_pos").first().alias("sx"),
                     pl.col("player_y_pos").first().alias("sy")))
    if site_src.height:
        maps = (G.select("match_key", "round", "map_name").unique(["match_key", "round"])
                .join(keys, on=["match_key", "round"], how="inner").select("rid", "map_name"))
        site_src = site_src.join(maps, on="rid", how="left")
        sr = site_src["rid"].to_numpy().astype(np.int64)
        site_tick_r[sr] = site_src["site_tick"].to_numpy().astype(np.int64)
        site_lab_r[sr] = _site_labels(site_src["map_name"].to_list(), site_src["sx"].to_numpy().astype(np.float64),
                                      site_src["sy"].to_numpy().astype(np.float64))
    plant_t = plant_r[rid]
    plant_elapsed = np.where(plant_t <= tick_g, (tick_g - plant_t) / TPS, np.nan)
    site = np.where(site_tick_r[rid] <= tick_g, site_lab_r[rid], np.nan)

    return pl.DataFrame({
        "defuse_active_ct": active.astype(np.float32),
        "defuse_elapsed_s_ct": defuse_elapsed.astype(np.float32),
        "defuse_attempts_ct": attempts.astype(np.float32),
        "defuse_kit_ct": defuse_kit.astype(np.float32),
        "plant_elapsed_s_t": plant_elapsed.astype(np.float32),
        "plant_site_t": site.astype(np.float32),
    })
