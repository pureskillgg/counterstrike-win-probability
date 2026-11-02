"""x1_kill_timing: kill timing and momentum inside the round.

Every value at a row uses only player_death events of the same (match_key, round) with tick <= the row tick, read through
features_events.scan_events. player_info is the static per-round roster (no tick) and is used only to fill missing team codes.

  x1_secs_since_death_ct    seconds since the most recent CT death (NaN before the first CT death)
  x1_secs_since_death_t     seconds since the most recent T death (NaN before the first T death)
  x1_first_kill_ct          +1 if CT got the round's first kill, -1 if T did, 0 if no kill yet
  x1_secs_since_first_kill  seconds since the round's first kill (NaN before it)
  x1_kills_ct_10s           kills made by CT in the last 10 s (ticks in (t-640, t])
  x1_kills_t_10s            kills made by T in the last 10 s
  x1_last_kill_trade        1 if the latest kill was a trade, 0 if not, NaN before any kill. A trade: the latest kill's victim
                            is the killer of the kill just before it, that earlier kill came at most 5 s before, and the
                            latest killer is on the side of that earlier victim.
  x1_last_kill_ct           +1 if the latest kill was made by CT, -1 by T, 0 if no kill yet
  x1_secs_since_last_kill   seconds since the latest kill of either side (NaN before any kill)

Conventions
  - Side codes: T = 2, CT = 3 (FE.T, FE.CT). The victim's side is the event's own player_team_code, else the roster team, else the
    opposite of the attacker's side. The attacker's side is the event's attacker_team_code, else the roster team, else the opposite
    of the victim's side (an enemy kill is the usual case).
  - A kill is a death by a real player (not the world id 65535, not a suicide) where killer and victim are on different sides.
    Teamkills count as deaths, not as kills.
  - Only the first death record of a player in a round is used; a second record of the same player in the same round is a data
    artifact (about 0.1% of rows).
  - Events with the same tick are ordered by player_id, so the result does not depend on the order of the rows read.
"""
import numpy as np
import polars as pl

import features_events as FE

BIG = FE.BIG
T, CT = FE.T, FE.CT
WORLD = 65535
TRADE_TICKS = 5 * 64
WINDOW_TICKS = 10 * 64
NEW_COLUMNS = ["x1_secs_since_death_ct", "x1_secs_since_death_t", "x1_first_kill_ct", "x1_secs_since_first_kill",
               "x1_kills_ct_10s", "x1_kills_t_10s", "x1_last_kill_trade", "x1_last_kill_ct", "x1_secs_since_last_kill"]


def _side(expr):
    return pl.when(expr.is_in([T, CT])).then(expr).otherwise(None)


def _last_before(ev_key, q, base):
    """Index of the last event with key <= q, and whether it belongs to the same round (key >= base)."""
    n = len(q)
    if ev_key.size == 0:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    idx = np.searchsorted(ev_key, q, side="right") - 1
    ok = idx >= 0
    idx = np.where(ok, idx, 0)
    ok &= ev_key[idx] >= base
    return idx, ok


def _secs_since(ev_key, ev_tick, q, base, tick_g):
    idx, ok = _last_before(ev_key, q, base)
    out = np.full(len(q), np.nan, dtype=np.float32)
    if ev_key.size:
        out[ok] = ((tick_g - ev_tick[idx]) / 64.0)[ok]
    return out


def _count_window(ev_key, q, base, tick_g, window):
    if ev_key.size == 0:
        return np.zeros(len(q), dtype=np.float32)
    hi = np.searchsorted(ev_key, q, side="right")
    lo = np.searchsorted(ev_key, base + tick_g - window, side="right")
    return (hi - lo).astype(np.float32)


def compute(G, store="combined"):
    keys, rid_g = FE._round_ids(G)
    n_rounds = keys.height
    tick_g = G["tick"].to_numpy().astype(np.int64)
    base = rid_g * BIG
    q = base + tick_g

    # ---- deaths: one record per player per round (the earliest), sides, kill flag
    D = FE.scan_events(store, "player_death", ["match_key", "round", "tick", "player_id", "attacker_id",
                                                "player_team_code", "attacker_team_code"])
    D = D.join(keys, on=["match_key", "round"], how="inner").with_columns(
        pl.col("rid").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
        pl.col("attacker_id").cast(pl.Int64), pl.col("player_team_code").cast(pl.Int64),
        pl.col("attacker_team_code").cast(pl.Int64))
    D = D.sort(["rid", "player_id", "tick"]).unique(subset=["rid", "player_id"], keep="first", maintain_order=True)

    PI = (FE.scan_events(store, "player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True)
          .select("match_key", "round", "player_id", pl.col("team_code").cast(pl.Int64).alias("roster_team")))
    D = D.join(PI, on=["match_key", "round", "player_id"], how="left")
    D = D.join(PI.rename({"player_id": "attacker_id", "roster_team": "attacker_roster_team"}),
               on=["match_key", "round", "attacker_id"], how="left")

    D = D.with_columns(
        pl.col("attacker_id").is_not_null().fill_null(False).alias("_has_att"),
    ).with_columns(
        (pl.col("_has_att") & (pl.col("attacker_id") != WORLD) & (pl.col("attacker_id") != pl.col("player_id"))).alias("real"),
        pl.coalesce([_side(pl.col("player_team_code")), _side(pl.col("roster_team"))]).alias("sv0"),
        pl.coalesce([_side(pl.col("attacker_team_code")), _side(pl.col("attacker_roster_team"))]).alias("sa0"),
    ).with_columns(
        pl.coalesce([pl.col("sv0"), pl.when(pl.col("real")).then(5 - pl.col("sa0")).otherwise(None)]).alias("sv"),
        pl.when(pl.col("real")).then(pl.coalesce([pl.col("sa0"), 5 - pl.col("sv0")])).otherwise(None).alias("sa"),
    ).with_columns(
        (pl.col("real") & pl.col("sa").is_in([T, CT]) & pl.col("sv").is_in([T, CT])
         & (pl.col("sa") != pl.col("sv"))).fill_null(False).alias("is_kill"),
    )

    # ---- death recency per side
    Dv = D.filter(pl.col("sv").is_in([T, CT])).sort(["rid", "tick", "player_id"])
    dkey = Dv["rid"].to_numpy() * BIG + Dv["tick"].to_numpy()
    dtick = Dv["tick"].to_numpy()
    dside = Dv["sv"].to_numpy()
    out = {}
    out["x1_secs_since_death_ct"] = _secs_since(dkey[dside == CT], dtick[dside == CT], q, base, tick_g)
    out["x1_secs_since_death_t"] = _secs_since(dkey[dside == T], dtick[dside == T], q, base, tick_g)

    # ---- kills, sorted by round, tick, victim id (deterministic order for ties)
    Kv = D.filter(pl.col("is_kill")).sort(["rid", "tick", "player_id"]).with_columns(
        (pl.col("player_id") == pl.col("attacker_id").shift(1).over("rid")).fill_null(False).alias("_v_is_prev_k"),
        (pl.col("tick") - pl.col("tick").shift(1).over("rid") <= TRADE_TICKS).fill_null(False).alias("_close"),
        (pl.col("sa") == pl.col("sv").shift(1).over("rid")).fill_null(False).alias("_same_side"),
    ).with_columns(
        (pl.col("_v_is_prev_k") & pl.col("_close") & pl.col("_same_side")).alias("trade"),
    )
    kkey = Kv["rid"].to_numpy() * BIG + Kv["tick"].to_numpy()
    ktick = Kv["tick"].to_numpy()
    kct = Kv["sa"].to_numpy() == CT
    ksign = np.where(kct, 1.0, -1.0).astype(np.float32)
    ktrade = Kv["trade"].to_numpy().astype(np.float32)

    # first kill of the round: its tick and sign, looked up per round then mapped onto grid rows
    fk_tick = np.full(n_rounds, -1, dtype=np.int64)
    fk_sign = np.zeros(n_rounds, dtype=np.float32)
    if Kv.height:
        first = Kv.unique(subset=["rid"], keep="first", maintain_order=True)
        fr = first["rid"].to_numpy()
        fk_tick[fr] = first["tick"].to_numpy()
        fk_sign[fr] = np.where(first["sa"].to_numpy() == CT, 1.0, -1.0).astype(np.float32)
    ft = fk_tick[rid_g]
    known = (ft >= 0) & (ft <= tick_g)
    out["x1_first_kill_ct"] = np.where(known, fk_sign[rid_g], 0.0).astype(np.float32)
    out["x1_secs_since_first_kill"] = np.where(known, (tick_g - ft) / 64.0, np.nan).astype(np.float32)

    # kills by side in the last 10 s
    out["x1_kills_ct_10s"] = _count_window(kkey[kct], q, base, tick_g, WINDOW_TICKS)
    out["x1_kills_t_10s"] = _count_window(kkey[~kct], q, base, tick_g, WINDOW_TICKS)

    # latest kill: trade flag, sign, and time since
    idx, ok = _last_before(kkey, q, base)
    out["x1_last_kill_trade"] = np.where(ok, ktrade[idx] if kkey.size else 0.0, np.nan).astype(np.float32)
    out["x1_last_kill_ct"] = np.where(ok, ksign[idx] if kkey.size else 0.0, 0.0).astype(np.float32)
    out["x1_secs_since_last_kill"] = _secs_since(kkey, ktick, q, base, tick_g)

    return pl.DataFrame({c: out[c] for c in NEW_COLUMNS})
