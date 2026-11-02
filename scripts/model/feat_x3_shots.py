"""x3_shots: shooting so far this round, per side (CT and T).

Every column ends with the side that fired or dealt the damage (_ct or _t). A value at row tick t uses only events of the
same round with tick <= t (searchsorted on rid*BIG+tick keys, the same rule as cum_by_round).

  shots_fired_<s>      firearm shots fired so far this round (weapon_fire; knives, grenades, zeus and C4 are left out)
  shots_per_alive_<s>  shots_fired_<s> divided by the players of that side still alive at the row (NaN if none alive)
  hits_<s>             enemy hits landed with a firearm so far (player_hurt; team hits, world damage and grenade or
                       fire damage are left out)
  hit_ratio_<s>        hits_<s> / shots_fired_<s> (NaN while the side has fired no shot); values above 1 occur (up to 2.5 in
                       the 400-row check)
  head_hits_<s>        firearm hits to the head so far (hit_box_code 1 = head: highest mean damage for rifles, checked)
  dmg_<s>              effective health removed by those firearm hits so far

Side of a shooter or a victim: the roster entry of that round in player_info (exact on this store: it agrees with the event
team codes on every player_hurt row where both exist; weapon_fire lacks a team code on about 3 % of rows).
Events whose round, player or side cannot be resolved are left out of the counts.

Grid rounds are matched to events by a 64-bit hash of the match key (no join of the whole grid).
"""
import numpy as np
import polars as pl

from features_events import BIG, CT, T, scan_events

PBIG = 1 << 20  # player_id slot in the roster key rid * PBIG + player_id (ids on this store reach 65285, not 0..9)
RSLOT = 64      # round slots in the lookup table (rounds run 1..24)
CHUNK = 2_000_000
HEAD = 1        # hit_box_code of a head hit
# firearms only: knives, grenades, zeus and C4 are not shots and their damage is not a bullet hit
GUNS = ["deagle", "elite", "fiveseven", "glock", "ak47", "aug", "awp", "famas", "g3sg1", "galilar", "m249", "m4a1",
        "mac10", "p90", "mp5sd", "ump45", "xm1014", "bizon", "mag7", "negev", "sawedoff", "tec9", "hkp2000", "mp7",
        "mp9", "nova", "p250", "scar20", "sg556", "ssg08", "cz75a", "revolver", "usp_silencer", "m4a1_silencer"]


def _grid_index(G):
    """keys: one row per grid (match, round) with rid. rid_g: rid of every grid row. h_sorted (sorted match hashes) and
    lut[code * RSLOT + round] = rid map an event to its grid round (see _rid_of)."""
    keys = G.select("match_key", "round").unique(maintain_order=True).with_row_index("rid")
    if keys["round"].max() >= RSLOT or keys["round"].min() < 0:
        raise ValueError("round outside the lookup table")
    h_key = keys["match_key"].hash().to_numpy()
    h_sorted = np.unique(h_key)
    if len(h_sorted) != keys["match_key"].n_unique():
        raise ValueError("two grid matches share a hash")
    lut = np.full(len(h_sorted) * RSLOT, -1, dtype=np.int64)
    lut[np.searchsorted(h_sorted, h_key) * RSLOT + keys["round"].to_numpy()] = keys["rid"].to_numpy().astype(np.int64)
    rid_g = _rid_of(h_sorted, lut, G["match_key"].hash().to_numpy(), G["round"].to_numpy())
    if (rid_g < 0).any():
        raise ValueError("grid row without a round id")
    return keys, rid_g, h_sorted, lut


def _rid_of(h_sorted, lut, h, rnd):
    """Grid rid of events with match hash h and round rnd; -1 when the round is not in the grid."""
    code = np.minimum(np.searchsorted(h_sorted, h), len(h_sorted) - 1)
    rid = lut[code * RSLOT + np.clip(rnd, 0, RSLOT - 1)]
    return np.where((h_sorted[code] == h) & (rnd >= 0) & (rnd < RSLOT), rid, -1)


def _roster(store, keys):
    """Sorted roster keys rid * PBIG + player_id and the side (2 = T, 3 = CT) of each entry, for the grid rounds."""
    PI = (scan_events(store, "player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True)
          .join(keys, on=["match_key", "round"], how="inner")
          .select(pl.col("rid").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("team_code").cast(pl.Int64))
          .filter(pl.col("team_code").is_in([T, CT]) & pl.col("player_id").is_between(0, PBIG - 1)))
    rk = PI["rid"].to_numpy() * PBIG + PI["player_id"].to_numpy()
    order = np.argsort(rk, kind="stable")
    return rk[order], PI["team_code"].to_numpy()[order].astype(np.int8)


def _side_of(roster, rid, pid):
    """Side of player pid in grid round rid from the roster; 0 where the player is not in the roster."""
    rk, rt = roster
    q = rid * PBIG + pid
    idx = np.minimum(np.searchsorted(rk, q), len(rk) - 1)
    hit = (rk[idx] == q) & (pid >= 0) & (pid < PBIG)
    return np.where(hit, rt[idx], 0).astype(np.int8)


def _shot_events(store, h_sorted, lut, roster):
    """Keys rid*BIG+tick and side of every firearm shot in a grid round whose side is known."""
    WF = scan_events(store, "weapon_fire", ["match_key", "round", "tick", "player_id", "weapon_name"])
    for c in ("round", "tick", "player_id"):
        if WF[c].null_count():
            raise ValueError(f"weapon_fire.{c} has nulls")
    n = WF.height
    mk, rnd, tick = WF["match_key"], WF["round"].to_numpy(), WF["tick"].to_numpy()
    pid, gun = WF["player_id"].to_numpy(), WF["weapon_name"].is_in(GUNS).to_numpy()
    ks, sds = [], []
    for s in range(0, n, CHUNK):  # chunks keep the hash and lookup temporaries small
        e = min(n, s + CHUNK)
        rid = _rid_of(h_sorted, lut, mk.slice(s, e - s).hash().to_numpy(), rnd[s:e])
        ok = gun[s:e] & (rid >= 0)
        rid = rid[ok]
        sd = _side_of(roster, rid, pid[s:e][ok])
        good = (sd == T) | (sd == CT)
        ks.append(rid[good] * BIG + tick[s:e][ok][good])
        sds.append(sd[good])
    del WF, mk
    k = np.concatenate(ks) if ks else np.zeros(0, dtype=np.int64)
    sd = np.concatenate(sds) if sds else np.zeros(0, dtype=np.int8)
    return k, sd


def _hit_events(store, h_sorted, lut, roster):
    """Enemy firearm hits: keys rid*BIG+tick, attacker side, head flag and effective damage."""
    H = scan_events(store, "player_hurt", ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name",
                                           "hit_box_code", "effective_health_removed"])
    H = H.filter(pl.col("weapon_name").is_in(GUNS) & pl.col("attacker_id").is_not_null()
                 & pl.col("player_id").is_not_null())
    rid = _rid_of(h_sorted, lut, H["match_key"].hash().to_numpy(), H["round"].to_numpy())
    keep = rid >= 0
    rid = rid[keep]
    tick = H["tick"].to_numpy()[keep]
    vic = H["player_id"].to_numpy()[keep]
    att = H["attacker_id"].to_numpy()[keep]
    head = (H["hit_box_code"].fill_null(-1).to_numpy()[keep] == HEAD)
    dmg = H["effective_health_removed"].fill_null(0).clip(lower_bound=0).to_numpy()[keep].astype(np.float64)
    del H
    vs = _side_of(roster, rid, vic)
    as_ = _side_of(roster, rid, att)
    ok = ((vs == T) | (vs == CT)) & ((as_ == T) | (as_ == CT)) & (vs != as_)
    return rid[ok] * BIG + tick[ok], as_[ok], head[ok], dmg[ok]


def _count_upto(ks_sorted, base, tick_g):
    """Number of sorted event keys of the row's round with tick <= the row tick."""
    hi = np.searchsorted(ks_sorted, base + tick_g, side="right")
    lo = np.searchsorted(ks_sorted, base, side="left")
    return (hi - lo).astype(np.float64)


def _sum_upto(k_ev, v_ev, base, tick_g):
    """Sum of v over the events of the row's round with tick <= the row tick (same rule as cum_by_round)."""
    order = np.argsort(k_ev, kind="stable")
    ks = k_ev[order]
    cs = np.concatenate([[0.0], np.cumsum(v_ev[order].astype(np.float64))])
    hi = np.searchsorted(ks, base + tick_g, side="right")
    lo = np.searchsorted(ks, base, side="left")
    return cs[hi] - cs[lo]


def compute(G, store):
    keys, rid_g, h_sorted, lut = _grid_index(G)
    base = rid_g * BIG
    tick_g = np.asarray(G["tick"].to_numpy(), dtype=np.int64)
    roster = _roster(store, keys)
    k_s, s_s = _shot_events(store, h_sorted, lut, roster)
    k_h, s_h, head_h, dmg_h = _hit_events(store, h_sorted, lut, roster)
    cols = {}
    for side, tag, alive_col in ((CT, "ct", "alive_ct"), (T, "t", "alive_t")):
        shots = _count_upto(np.sort(k_s[s_s == side]), base, tick_g)
        m = s_h == side
        hits = _sum_upto(k_h[m], np.ones(int(m.sum())), base, tick_g)
        heads = _sum_upto(k_h[m], head_h[m], base, tick_g)
        dmg = _sum_upto(k_h[m], dmg_h[m], base, tick_g)
        alive = np.asarray(G[alive_col].to_numpy(), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_alive = np.where(alive > 0, shots / alive, np.nan)
            ratio = np.where(shots > 0, hits / shots, np.nan)
        cols[f"shots_fired_{tag}"] = shots.astype(np.float32)
        cols[f"shots_per_alive_{tag}"] = per_alive.astype(np.float32)
        cols[f"hits_{tag}"] = hits.astype(np.float32)
        cols[f"hit_ratio_{tag}"] = ratio.astype(np.float32)
        cols[f"head_hits_{tag}"] = heads.astype(np.float32)
        cols[f"dmg_{tag}"] = dmg.astype(np.float32)
        del shots, hits, heads, dmg, per_alive, ratio
    return pl.DataFrame(cols)


feat = compute
