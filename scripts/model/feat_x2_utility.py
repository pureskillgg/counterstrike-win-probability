"""x2_utility: grenades and molotovs used so far this round, and enemy flash blinds so far, per side.

Every column ends with the side that threw the grenade or flashed the enemy (_ct or _t). A value at row tick t uses only
events of the same round with tick <= t.

  det_flash_<s>, det_smoke_<s>, det_he_<s>, det_molo_<s>, det_decoy_<s>
        grenades of that type that have landed so far this round (flashbang, smoke, HE and decoy detonation or start;
        molotov / incendiary fire start)
  det_total_<s>           the five counts above added up
  smoke_active_<s>        smokes from this side whose detonation is less than 18 s old
  molo_active_<s>         molotov / incendiary fires from this side less than 7 s old
  blind_enemy_<s>         enemy players blinded by flashes from this side so far (self-flashes and team flashes excluded)
  blind_enemy_sec_<s>     enemy blind seconds caused by this side so far
  blind_enemy_active_<s>  enemy blinds from this side still in effect at the row tick

Side of a thrower or flashed player: the event's own team code (2 = T, 3 = CT) first, else the player's roster entry in
player_info for that round. Events whose side cannot be resolved are left out of the counts.

Throws are counted from grenade_state detonations, not from weapon_fire: weapon_fire has about 18 million rows in this
store and would add about 1.3 GB of working memory. On this store the detonation counts match the weapon_fire grenade
throw counts to within 0.2 %; the only difference is the flight time between throw and landing.
"""
import numpy as np
import polars as pl

from features_events import CT, T, _round_ids, cum_by_round, scan_events

SMOKE_TICKS = 18 * 64
MOLO_TICKS = 7 * 64
DET_KINDS = {"flashbang_detonate": "flash", "smokegrenade_detonate": "smoke", "hegrenade_detonate": "he",
             "decoy_started": "decoy"}
KEY = ["match_key", "round"]


def _roster(store):
    """Round roster from player_info. It has no tick column: it is a round-level snapshot."""
    return (scan_events(store, "player_info", ["match_key", "round", "player_id", "team_code"])
            .unique(["match_key", "round", "player_id"], keep="last", maintain_order=True)
            .select("match_key", "round", pl.col("player_id").alias("_pid"), pl.col("team_code").alias("_roster")))


def _side(df, roster, pid, team):
    """Side of the player in column pid: the event's own team code first, else the roster entry of that round."""
    df = df.join(roster, left_on=KEY + [pid], right_on=KEY + ["_pid"], how="left")
    own = pl.when(pl.col(team).is_in([T, CT])).then(pl.col(team))
    ros = pl.when(pl.col("_roster").is_in([T, CT])).then(pl.col("_roster"))
    df = df.with_columns(pl.coalesce([own, ros]).cast(pl.Int64).alias("side"))
    return df.drop([c for c in ("_roster", "_pid") if c in df.columns])


def _ids(df, keys):
    """Keep only rows of rounds in the grid and attach the grid round id rid."""
    return df.join(keys, on=KEY, how="inner").with_columns(pl.col("rid").cast(pl.Int64), pl.col("tick").cast(pl.Int64))


def _starts_ends(f, dur):
    """+1 at the start tick and -1 at start + dur, so the cumulative sum counts the items still active at each tick.
    dur is an int or a column expression, in ticks."""
    start = f.select("rid", "tick").with_columns(pl.lit(1.0).alias("v"))
    end = f.select("rid", (pl.col("tick") + dur).alias("tick")).with_columns(pl.lit(-1.0).alias("v"))
    return pl.concat([start, end])


def _cum(G, rid_g, frame, value):
    return cum_by_round(G, frame.select("rid", "tick", value), value, rid_g)


def compute(G, store):
    keys, rid_g = _round_ids(G)
    roster = _roster(store)
    cols = {}

    # grenades landed so far, by type; the thrower side from the event's team code or the roster
    GS = (scan_events(store, "grenade_state", ["match_key", "round", "tick", "event_type", "entity_id", "player_id",
                                               "player_team_code"])
          .filter(pl.col("event_type").is_in(list(DET_KINDS)))
          .unique(["match_key", "round", "tick", "entity_id", "event_type"], keep="first", maintain_order=True))
    GS = _side(_ids(GS, keys), roster, "player_id", "player_team_code")
    kinds = {kind: GS.filter(pl.col("event_type") == et) for et, kind in DET_KINDS.items()}

    # molotov / incendiary fires started (inferno_startburn); the thrower is player_id of the fire
    MS = (scan_events(store, "molotov_state", ["match_key", "round", "tick", "event_type", "entity_id", "player_id",
                                               "player_team_code"])
          .filter(pl.col("event_type") == "inferno_startburn")
          .unique(["match_key", "round", "tick", "entity_id", "event_type"], keep="first", maintain_order=True))
    kinds["molo"] = _side(_ids(MS, keys), roster, "player_id", "player_team_code")

    # flashes: the flasher is attacker_id, the victim is player_id; keep enemy blinds only
    PB = (scan_events(store, "player_blind", ["match_key", "round", "tick", "player_id", "attacker_id", "blind_duration",
                                              "player_team_code", "attacker_team_code"])
          .unique(["match_key", "round", "tick", "player_id", "attacker_id"], keep="first", maintain_order=True)
          .filter(pl.col("player_id") != pl.col("attacker_id")))
    PB = _side(_ids(PB, keys), roster, "attacker_id", "attacker_team_code").rename({"side": "att"})
    PB = _side(PB, roster, "player_id", "player_team_code").rename({"side": "vic"})
    PB = (PB.filter(pl.col("att").is_not_null() & pl.col("vic").is_not_null() & (pl.col("att") != pl.col("vic")))
          .with_columns((pl.col("blind_duration") * 64).floor().cast(pl.Int64).alias("dticks")))

    for s, tag in ((CT, "ct"), (T, "t")):
        tot = np.zeros(G.height)
        for kind in ("flash", "smoke", "he", "molo", "decoy"):
            arr = _cum(G, rid_g, kinds[kind].filter(pl.col("side") == s).with_columns(pl.lit(1.0).alias("v")), "v")
            cols[f"det_{kind}_{tag}"] = arr
            tot += arr
        cols[f"det_total_{tag}"] = tot
        cols[f"smoke_active_{tag}"] = _cum(G, rid_g, _starts_ends(kinds["smoke"].filter(pl.col("side") == s), SMOKE_TICKS), "v")
        cols[f"molo_active_{tag}"] = _cum(G, rid_g, _starts_ends(kinds["molo"].filter(pl.col("side") == s), MOLO_TICKS), "v")
        bl = PB.filter(pl.col("att") == s)
        cols[f"blind_enemy_{tag}"] = _cum(G, rid_g, bl.with_columns(pl.lit(1.0).alias("v")), "v")
        cols[f"blind_enemy_sec_{tag}"] = _cum(G, rid_g, bl.with_columns(pl.col("blind_duration").alias("v")), "v")
        cols[f"blind_enemy_active_{tag}"] = _cum(G, rid_g, _starts_ends(bl, pl.col("dticks")), "v")

    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()})


feat = compute
