"""Living-player loadout features from the 1 Hz player_status extraction (data/status_1hz/*.parquet, 2026 only).

The extraction keeps one row per ALIVE player at exactly the grid ticks, so a feature for grid row (match, round, tick) is a
function of that tick's snapshot alone: causal by construction (no event stream, no later information).

Per side (_ct / _t), over the players alive at that second:
  hp          sum of health                    armor_n     players with armor > 0       armor_sum   sum of armor points
  helmet_n    players with a helmet            kit_n       CT only: defuse kits         defusing    CT only: someone defusing
  ak, m4, awp, rifle (all rifles), sniper_other (scout, autos), smg, heavy (shotgun / machine gun), no_primary
  flash, smoke, he, fire (molotov + incendiary), decoy    grenades held (counts)
  equip       sum of equipment_value_calc (value of what the living still carry)
Item codes are CS2 item definition indexes (verified against weapon_fire.weapon_name: 7 ak47, 9 awp, 16 m4a1, 60 m4a1_silencer ...).

  loadout_features(G, status_dir) -> polars DataFrame, same rows and order as G (NaN where a match has no status file)
"""
import glob
import os

import numpy as np
import polars as pl

import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(__file__))

ROOT = r"G:\pureskill\csds-cs2"
T, CT = 2, 3
from wp_items import HEAVY, M4, RIFLE, SMG, SNIPER_OTHER  # noqa: E402  (shared with serving: snapshot_features.py)

RIFLE, SNIPER_OTHER, SMG, HEAVY, M4 = list(RIFLE), list(SNIPER_OTHER), list(SMG), list(HEAVY), list(M4)


def load_status(status_dir=None, matches=None):
    status_dir = status_dir or os.path.join(ROOT, "data", "status_1hz")
    fs = sorted(glob.glob(os.path.join(status_dir, "*.parquet")))
    S = pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed")
    if matches is not None:
        S = S.filter(pl.col("match_key").is_in(matches))
    return S


def side_aggregates(S):
    """One row per (match_key, round, tick, side) with the loadout aggregates."""
    p = pl.col("inv_primary")
    return S.group_by(["match_key", "round", "tick", "side"]).agg(
        pl.col("health").sum().alias("hp"),
        (pl.col("armor") > 0).sum().alias("armor_n"), pl.col("armor").sum().alias("armor_sum"),
        pl.col("has_helmet").sum().alias("helmet_n"), pl.col("has_defuser").sum().alias("kit_n"),
        pl.col("is_defusing").sum().alias("defusing"),
        (p == 7).sum().alias("ak"), p.is_in(M4).sum().alias("m4"), (p == 9).sum().alias("awp"),
        p.is_in(RIFLE).sum().alias("rifle"), p.is_in(SNIPER_OTHER).sum().alias("sniper_other"),
        p.is_in(SMG).sum().alias("smg"), p.is_in(HEAVY).sum().alias("heavy"), (p == 0).sum().alias("no_primary"),
        pl.col("inv_flashbang").sum().alias("flash"), pl.col("inv_smokegrenade").sum().alias("smoke"),
        pl.col("inv_hegrenade").sum().alias("he"), (pl.col("inv_molotov") + pl.col("inv_incgrenade")).sum().alias("fire"),
        pl.col("inv_decoy").sum().alias("decoy"), pl.col("equipment_value_calc").sum().alias("equip"),
    )


STATUS_COLS = ["match_key", "round", "tick", "side", "health", "armor", "has_helmet", "has_defuser", "is_defusing", "inv_primary",
               "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy", "equipment_value_calc"]


def loadout_aggregates(status_dir, matches, files=None):
    """Per-day-file aggregation (the raw 1 Hz table is ~31 M rows; only the small per-side aggregates are kept in memory).
    files: optional list of day files to read (default: every file in status_dir), so a caller can work one day at a time.
    Returns (A, have): A one row per (match_key, round, tick, side); have = the match_keys that have any status rows."""
    status_dir = status_dir or os.path.join(ROOT, "data", "status_1hz")
    parts, haves = [], []
    for f in (files if files is not None else sorted(glob.glob(os.path.join(status_dir, "*.parquet")))):
        S = pl.read_parquet(f, columns=STATUS_COLS).filter(pl.col("match_key").is_in(matches))
        if S.height:
            parts.append(side_aggregates(S))
            haves.append(S.select("match_key").unique())
        del S
    A = pl.concat(parts, how="vertical_relaxed").unique(subset=["match_key", "round", "tick", "side"], keep="first")
    return A, pl.concat(haves).unique()


def inconsistent_rounds(G, status_dir=None):
    """(match_key, round) pairs where, at any grid tick of the round, the number of living players in the raw player table differs from
    the event-derived alive counts of the grid (a missing event or a missing table row; CSDS "known flaws": events can be missing within a
    match). 0.01% of rounds in 2026 (audit Y2). The snapshot model should neither train nor be scored on them: the table and the counts
    would then describe two different games. Matches without any status file are not touched here."""
    status_dir = status_dir or os.path.join(ROOT, "data", "status_1hz")
    keys = G.select("match_key", "round", "tick", "alive_ct", "alive_t")
    mk = G["match_key"].unique().to_list()
    parts = []
    for f in sorted(glob.glob(os.path.join(status_dir, "*.parquet"))):
        S = pl.read_parquet(f, columns=["match_key", "round", "tick", "side"]).filter(pl.col("match_key").is_in(mk))
        if S.height:
            parts.append(S.group_by(["match_key", "round", "tick"]).agg((pl.col("side") == CT).sum().alias("n_ct"), (pl.col("side") == T).sum().alias("n_t")))
    if not parts:
        return pl.DataFrame({"match_key": [], "round": []}, schema={"match_key": pl.String, "round": G.schema["round"]})
    N = pl.concat(parts)
    have = N.select("match_key").unique()
    J = keys.join(have, on="match_key", how="semi").join(N, on=["match_key", "round", "tick"], how="left").with_columns(pl.col("n_ct").fill_null(0), pl.col("n_t").fill_null(0))
    return J.filter((pl.col("n_ct") != pl.col("alive_ct")) | (pl.col("n_t") != pl.col("alive_t"))).select("match_key", "round").unique()


def drop_inconsistent_rounds(G, status_dir=None):
    return G.join(inconsistent_rounds(G, status_dir), on=["match_key", "round"], how="anti")


def loadout_features(G, status_dir=None, files=None):
    A, have = loadout_aggregates(status_dir, G["match_key"].unique().to_list(), files=files)
    cols = [c for c in A.columns if c not in ("match_key", "round", "tick", "side")]
    ct = A.filter(pl.col("side") == CT).drop("side").rename({c: c + "_ct" for c in cols})
    t = A.filter(pl.col("side") == T).drop("side").rename({c: c + "_t" for c in cols})
    key = ["match_key", "round", "tick"]
    base = G.select(key)
    # a side with nobody alive has no rows: its counts are 0 (valid), but a match without any status file stays NaN
    have = have.with_columns(pl.lit(True).alias("_has"))
    J = base.join(ct, on=key, how="left").join(t, on=key, how="left").join(have, on="match_key", how="left")
    out = {}
    for c in J.columns:
        if c in key or c == "_has":
            continue
        v = J[c].to_numpy().astype(np.float64)
        v = np.where(np.isnan(v) & J["_has"].fill_null(False).to_numpy(), 0.0, v)
        out["st_" + c] = v.astype(np.float32)
    df = pl.DataFrame(out)
    # differences the trees would otherwise have to discover
    df = df.with_columns([(pl.col(f"st_{b}_ct") - pl.col(f"st_{b}_t")).alias(f"st_{b}_diff") for b in ("hp", "rifle", "equip", "armor_n", "awp", "smg")])
    return df


compute = None  # needs a status_dir; wrapped by the experiment scripts (loadout_features)
