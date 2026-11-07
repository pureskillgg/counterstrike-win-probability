"""Round win probability from DSDK channel data: P(team wins) at every live second of every round, in percent.

    from win_probability import win_probability
    probs = win_probability(data, team="T")                    # {tick: percent}
    probs = win_probability(data, team="T", unit="fraction")   # {tick: 0 to 1}
    data = load_tome_match(tome_dir, match_key)                 # the channels of one match, read from a DSDK tome folder

  command line (same reading and output):
    python win_probability.py --tome <csds folder> --match <match key> --team T [--unit fraction] [--out probs.json]

data      dict of DataFrames keyed by channel name, read as-is from the DSDK tome (keys with or without the "basic_" prefix):
          header, round_state, round_end, player_info, player_death, bomb_state (round_start is used if round_state has no freeze end).
          pandas or polars DataFrames both work.
output    dict  tick -> percent (0 to 100) for the team asked ("CT" or "T"); unit="fraction" gives 0 to 1 instead. One entry per live second of each round: ticks freeze_end + 64 k,
          from the start of live play up to the round end. Round ticks never overlap, so a match gives one dictionary.
model     winprob_v6_drop_heavy (the default; the submission, decided 20 October): 57 inputs, one XGBoost model (seed 0), no score lead. The inputs: the alive counts,
          clock, bomb and map (from the event channels), and 44 loadout inputs from the per-tick player_status channel: 19 per side
          (health, armour, helmets, kits, defusing, weapon classes, grenades, equipment value) for each of the two sides, plus 6
          CT-minus-T differences. A tome without player_status cannot be scored: the function returns {} and says so in last_warnings.
          Trained on the current split manifest. Other models, passed by name: winprob_v8_ensemble (58 inputs: the same with the score lead,
          the freeze-end CT score minus the T score from round_state; the mean of five seeds; model_name="winprob_v8_ensemble") and
          winprob_v7_basic_scorelead (14 inputs, event channels only) with
          model_name="winprob_v7_basic_scorelead".
never raises for a bad match: a map we do not model gives all-zero map inputs, a wingman or other non-standard match gives a warning and a
dictionary from the rounds it can read, and an unreadable round is skipped. Every such case is listed in win_probability.last_warnings.

Dependencies: xgboost (to load the model), polars (to read the channels), numpy. Nothing else from this project.
"""
import json
import os
import warnings

import numpy as np
import polars as pl
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "models"))
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
TICK = 64
ROUND_SECONDS = 115
UNPLANTED_CAP_SECONDS = 116
CT, T = 3, 2
START_PLAYERS = 5
_cache = {}
last_warnings = []
# CS2 item definition indexes for the loadout inputs (same values as scripts/model/wp_items.py, which the training uses)
RIFLE = (7, 8, 10, 13, 16, 39, 60)
M4 = (16, 60)
AK = (7,)
AWP = (9,)
SNIPER_OTHER = (11, 38, 40)
SMG = (17, 19, 23, 24, 26, 33, 34)
HEAVY = (14, 25, 27, 28, 29, 35)
SIDE_FIELDS = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle", "sniper_other", "smg",
               "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip"]
DIFF_FIELDS = ["hp", "rifle", "equip", "armor_n", "awp", "smg"]


def _loadout_from_players(players):
    """The all-zero loadout ZERO_LOADOUT is the only use of this function (a tick with no player_status rows gets it).
    It is not the per-second loadout: win_probability() builds those in its polars group_by over player_status.
    Input: the players of one second. It returns 20 fields per side (heavy is computed but is not a v6 or v8 input)
    and the 6 CT-minus-T differences, so 46 keys; the model reads only its own 44 names.
    players: dicts with side (CT or T as 3 or 2), health, armor, has_helmet, has_defuser, is_defusing, primary, flashbang,
    smokegrenade, hegrenade, molotov, incgrenade, decoy, equipment_value. Definitions follow scripts/model/features_status.py
    side_aggregates, except that a null primary counts as no primary (fill_null(0)), as in the group_by."""
    agg = {CT: dict.fromkeys(SIDE_FIELDS, 0.0), T: dict.fromkeys(SIDE_FIELDS, 0.0)}
    for p in players:
        a = agg[p["side"]]
        prim = int(p.get("primary") or 0)
        a["hp"] += float(p.get("health") or 0)
        a["armor_n"] += 1 if float(p.get("armor") or 0) > 0 else 0
        a["armor_sum"] += float(p.get("armor") or 0)
        a["helmet_n"] += 1 if p.get("has_helmet") else 0
        a["kit_n"] += 1 if p.get("has_defuser") else 0
        a["defusing"] += 1 if p.get("is_defusing") else 0
        a["ak"] += prim in AK
        a["m4"] += prim in M4
        a["awp"] += prim in AWP
        a["rifle"] += prim in RIFLE
        a["sniper_other"] += prim in SNIPER_OTHER
        a["smg"] += prim in SMG
        a["heavy"] += prim in HEAVY
        a["no_primary"] += prim == 0
        a["flash"] += float(p.get("flashbang") or 0)
        a["smoke"] += float(p.get("smokegrenade") or 0)
        a["he"] += float(p.get("hegrenade") or 0)
        a["fire"] += float(p.get("molotov") or 0) + float(p.get("incgrenade") or 0)
        a["decoy"] += float(p.get("decoy") or 0)
        a["equip"] += float(p.get("equipment_value") or 0)
    out = {}
    for name in SIDE_FIELDS:
        out[f"st_{name}_ct"] = float(agg[CT][name])
        out[f"st_{name}_t"] = float(agg[T][name])
    for name in DIFF_FIELDS:
        out[f"st_{name}_diff"] = out[f"st_{name}_ct"] - out[f"st_{name}_t"]
    return out


ZERO_LOADOUT = _loadout_from_players([])


def _frame(x):
    if x is None:
        return None
    if isinstance(x, pl.DataFrame):
        return x
    return pl.from_pandas(x)


def _channel(data, name):
    for key in (name, f"basic_{name}"):
        if key in data and data[key] is not None:
            return _frame(data[key])
    return None


def _model(model_name):
    if model_name not in _cache:
        with open(os.path.join(MODEL_DIR, f"{model_name}.meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if "members" in meta:  # an ensemble of seed models: the served probability is the mean of their probabilities
            prefix = meta.get("member_prefix", model_name.replace("_ensemble", ""))
            members = []
            for m in meta["members"]:
                bst = xgb.Booster()
                bst.load_model(os.path.join(MODEL_DIR, f"{prefix}_seed{m['seed']}.json"))
                members.append((bst, int(m["best_iteration"])))
            _cache[model_name] = (members, meta["features"], None)
        else:
            bst = xgb.Booster()
            bst.load_model(os.path.join(MODEL_DIR, f"{model_name}.json"))
            _cache[model_name] = (bst, meta["features"], int(meta.get("best_iteration", bst.num_boosted_rounds() - 1)))
    return _cache[model_name]


def _features_for_round(frz, end, plant, roster_ct, roster_t, deaths, map_name, feature_names, lead=0.0, load=None):
    """Yields (tick, feature vector) for every live second of one round. lead: freeze-end CT score minus T score (score lead).
    load: {tick: loadout inputs} from the per-tick player status (a tick without rows gets the all-zero loadout: nobody alive on it)."""
    deaths = sorted(deaths)
    planted_at = plant
    dead_ct, dead_t, seen = 0, 0, set()
    dead_index = 0
    last = end if planted_at is not None else min(end, frz + UNPLANTED_CAP_SECONDS * TICK)
    k = 0
    rows = []
    while True:
        tick = frz + TICK * k
        if tick >= last:
            break
        while dead_index < len(deaths) and deaths[dead_index][0] <= tick:
            _, pid, side = deaths[dead_index]
            dead_index += 1
            if pid in seen:
                continue
            seen.add(pid)
            if side == CT:
                dead_ct += 1
            elif side == T:
                dead_t += 1
        alive_ct = max(0, roster_ct - dead_ct)
        alive_t = max(0, roster_t - dead_t)
        planted = int(planted_at is not None and planted_at <= tick)
        vals = {
            "alive_ct": alive_ct, "alive_t": alive_t, "ct_alive_minus_t": alive_ct - alive_t, "bomb_planted": planted,
            "seconds_since_round_live": (tick - frz) / TICK,
            "seconds_since_plant": (tick - planted_at) / TICK if planted else -1.0,
        }
        for m in MAPS:
            vals[f"map_is_{m}"] = int(map_name == m)
        vals["score_lead_diag"] = float(lead)
        if load is not None:
            vals.update(load.get(tick, ZERO_LOADOUT))
        rows.append((tick, [float(vals[f]) for f in feature_names]))
        k += 1
    return rows


def win_probability(data, team="CT", model_name="winprob_v6_drop_heavy", unit="percent"):
    global last_warnings
    last_warnings = []
    team = str(team).upper()
    if team not in ("CT", "T"):
        raise ValueError('team must be "CT" or "T"')
    if unit not in ("percent", "fraction"):
        raise ValueError('unit must be "percent" or "fraction"')
    bst, feature_names, best = _model(model_name)

    def warn(msg):
        last_warnings.append(msg)
        warnings.warn(msg)

    header = _channel(data, "header")
    if header is None or header.height == 0:
        warn("no header channel: map and mode unknown; scoring with all map inputs 0")
        map_name, is_wingman, max_rounds = None, False, 24
    else:
        h = header.row(0, named=True)
        if header.height > 1:
            warn(f"{header.height} header rows: only the first match is scored")
        map_name = str(h.get("map_name") or "").lower().replace("de_", "") or None
        is_wingman = bool(h.get("is_wingman") or False)
        max_rounds = int(h.get("max_rounds") or 24)
    if map_name not in MAPS:
        warn(f"map {map_name!r} is not one of the modelled maps: all map inputs set to 0")
        map_name = None
    if is_wingman:
        warn("wingman match: the model is trained on competitive 5v5 only; probabilities are not valid for this match")
    if max_rounds != 24:
        warn(f"max_rounds {max_rounds}: the model is trained on 24-round regulation; overtime rounds are scored as normal rounds")

    state = _channel(data, "round_state")
    freeze = {}
    if state is not None and "event_type" in state.columns:
        for r in state.filter(pl.col("event_type") == "round_freeze_end").select("round", "tick").iter_rows():
            freeze.setdefault(int(r[0]), int(r[1]))
    rstart = _channel(data, "round_start")
    if rstart is not None:
        for r in rstart.select("round", "tick").iter_rows():
            freeze.setdefault(int(r[0]), int(r[1]))
    # freeze-end scores (score lead): the round_state row at the freeze end, CT and T score
    freeze_scores = {}
    if state is not None and "event_type" in state.columns and "ct_score" in state.columns and "t_score" in state.columns:
        for r in state.filter(pl.col("event_type") == "round_freeze_end").select("round", "ct_score", "t_score").iter_rows():
            freeze_scores.setdefault(int(r[0]), (r[1], r[2]))
    uses_lead = "score_lead_diag" in feature_names
    uses_loadout = any(f.startswith("st_") for f in feature_names)

    ends = _channel(data, "round_end")
    if ends is None:
        warn("no round_end channel: no rounds to score")
        return {}

    roster = {}
    pinfo = _channel(data, "player_info")
    if pinfo is not None:
        for r in pinfo.select("round", "player_id", "team_code").iter_rows():
            if r[0] is None or r[2] is None:
                continue
            roster.setdefault(int(r[0]), {}).setdefault(int(r[2]), set()).add(r[1])

    # loadout inputs from the per-tick player status, for the live seconds only: the side of each player comes from the roster of its round
    loadout_by_round = {}
    if uses_loadout:
        ps = _channel(data, "player_status")
        if ps is None:
            warn("no player_status channel: the loadout inputs need per-tick player status, so no probabilities are returned")
            return {}
        roster_rows = [(rnd, pid, side_code) for rnd, teams in roster.items() for side_code, pids in teams.items() for pid in pids]
        fdf = pl.DataFrame({"round": list(freeze.keys()), "frz": list(freeze.values())}, schema={"round": pl.Int64, "frz": pl.Int64})
        live = (ps.join(fdf, on="round", how="inner")
                .filter(((pl.col("tick") - pl.col("frz")) % TICK == 0) & (pl.col("tick") >= pl.col("frz"))))
        side_schema = {"round": pl.Int64, "player_id": live.schema["player_id"], "side": pl.Int64}
        side_df = pl.DataFrame(roster_rows, schema=side_schema, orient="row") if roster_rows else pl.DataFrame(schema=side_schema)
        joined = live.join(side_df, on=["round", "player_id"], how="left")
        skipped = joined.filter(pl.col("side").is_null()).height
        joined = joined.filter(pl.col("side").is_not_null())
        if skipped:
            warn(f"{skipped} player-status rows had no side in the roster and were left out of the loadout inputs")
        inv = pl.col("inv_primary").fill_null(0).cast(pl.Int64)
        agg = joined.group_by(["round", "tick", "side"]).agg([
            pl.col("health").fill_null(0).cast(pl.Float64).sum().alias("hp"),
            (pl.col("armor").fill_null(0) > 0).cast(pl.Int64).sum().alias("armor_n"),
            pl.col("armor").fill_null(0).cast(pl.Float64).sum().alias("armor_sum"),
            pl.col("has_helmet").fill_null(False).cast(pl.Int64).sum().alias("helmet_n"),
            pl.col("has_defuser").fill_null(False).cast(pl.Int64).sum().alias("kit_n"),
            pl.col("is_defusing").fill_null(False).cast(pl.Int64).sum().alias("defusing"),
            (inv == AK[0]).cast(pl.Int64).sum().alias("ak"),
            inv.is_in(list(M4)).cast(pl.Int64).sum().alias("m4"),
            (inv == AWP[0]).cast(pl.Int64).sum().alias("awp"),
            inv.is_in(list(RIFLE)).cast(pl.Int64).sum().alias("rifle"),
            inv.is_in(list(SNIPER_OTHER)).cast(pl.Int64).sum().alias("sniper_other"),
            inv.is_in(list(SMG)).cast(pl.Int64).sum().alias("smg"),
            inv.is_in(list(HEAVY)).cast(pl.Int64).sum().alias("heavy"),
            (inv == 0).cast(pl.Int64).sum().alias("no_primary"),
            pl.col("inv_flashbang").fill_null(0).cast(pl.Float64).sum().alias("flash"),
            pl.col("inv_smokegrenade").fill_null(0).cast(pl.Float64).sum().alias("smoke"),
            pl.col("inv_hegrenade").fill_null(0).cast(pl.Float64).sum().alias("he"),
            (pl.col("inv_molotov").fill_null(0) + pl.col("inv_incgrenade").fill_null(0)).cast(pl.Float64).sum().alias("fire"),
            pl.col("inv_decoy").fill_null(0).cast(pl.Float64).sum().alias("decoy"),
            pl.col("equipment_value_calc").fill_null(0).cast(pl.Float64).sum().alias("equip"),
        ])
        by_rt = {}
        for r in agg.iter_rows(named=True):
            side_key = "ct" if int(r["side"]) == CT else "t"
            d = by_rt.setdefault((int(r["round"]), int(r["tick"])), {})
            for f in SIDE_FIELDS:
                d[f"st_{f}_{side_key}"] = float(r[f])
        for (rnd, tick), d in by_rt.items():
            for side_key in ("ct", "t"):
                for f in SIDE_FIELDS:
                    d.setdefault(f"st_{f}_{side_key}", 0.0)
            for f in DIFF_FIELDS:
                d[f"st_{f}_diff"] = d[f"st_{f}_ct"] - d[f"st_{f}_t"]
            loadout_by_round.setdefault(rnd, {})[tick] = d

    deaths = {}
    pdeath = _channel(data, "player_death")
    if pdeath is not None:
        for r in pdeath.select("round", "tick", "player_id", "player_team_code").iter_rows():
            if r[3] is None:
                continue
            deaths.setdefault(int(r[0]), []).append((int(r[1]), r[2], int(r[3])))

    plants = {}
    bomb = _channel(data, "bomb_state")
    if bomb is not None and "event_type" in bomb.columns:
        for r in bomb.filter(pl.col("event_type") == "bomb_planted").select("round", "tick").iter_rows():
            plants.setdefault(int(r[0]), []).append(int(r[1]))

    out = {}
    for r in ends.select("round", "tick").iter_rows():
        rnd, end = int(r[0]), int(r[1])
        frz = freeze.get(rnd)
        if frz is None:
            warn(f"round {rnd}: no freeze-end tick: skipped")
            continue
        if end <= frz:
            continue
        ct_n = len(roster.get(rnd, {}).get(CT, set()))
        t_n = len(roster.get(rnd, {}).get(T, set()))
        if ct_n == 0 and t_n == 0:
            ct_n = t_n = START_PLAYERS
            warn(f"round {rnd}: no roster: assumed 5 vs 5")
        elif ct_n != START_PLAYERS or t_n != START_PLAYERS:
            warn(f"round {rnd}: {ct_n} CT and {t_n} T at the start (not 5 vs 5); scored with the counts found")
        in_win = sorted(t for t in plants.get(rnd, []) if frz <= t <= end)
        plant = in_win[0] if in_win else None
        rd = [(t, pid, side) for (t, pid, side) in deaths.get(rnd, []) if frz <= t <= end]
        lead = 0.0
        if uses_lead:
            sc = freeze_scores.get(rnd)
            if sc is None or sc[0] is None or sc[1] is None:
                warn(f"round {rnd}: no freeze-end scores: score lead set to 0")
            else:
                lead = float(sc[0]) - float(sc[1])
        for tick, vec in _features_for_round(frz, end, plant, ct_n, t_n, rd, map_name, feature_names, lead,
                                             loadout_by_round.get(rnd, {}) if uses_loadout else None):
            out[tick] = vec
    if not out:
        return {}
    ticks = sorted(out)
    X = np.array([out[t] for t in ticks], dtype=np.float32)
    if isinstance(bst, list):  # ensemble: mean of the seed models' probabilities
        p_ct = np.mean([m.predict(xgb.DMatrix(X), iteration_range=(0, it + 1)) for m, it in bst], axis=0)
    else:
        p_ct = bst.predict(xgb.DMatrix(X), iteration_range=(0, best + 1))
    # CT is rounded first and T is taken from the rounded CT, so CT + T is exactly 100 (fraction: 1) on every tick
    scale, digits = (1.0, 6) if unit == "fraction" else (100.0, 4)
    ct = {int(t): round(float(scale * v), digits) for t, v in zip(ticks, p_ct)}
    if team == "CT":
        return ct
    return {t: round(scale - v, digits) for t, v in ct.items()}


TOME_CHANNELS = ["round_state", "round_start", "round_end", "player_info", "player_death", "bomb_state", "player_status"]


def load_tome_match(tome_dir, match_key, suffix=None):
    """dict of polars DataFrames for one match, read from a DSDK tome's csds folder: header/ and basic_<channel><suffix>/ pages.
    The reading is the one the parity test uses (test_win_probability.py). Every date-range folder of a channel is read, so a match is
    found whichever range it is in. suffix, e.g. ".2026-09-18,2026-10-08", restricts the read to one folder."""
    import glob
    out = {}
    pages = sorted(glob.glob(os.path.join(tome_dir, "header", "dataframe_*")))
    if pages:
        hdr = pl.concat([pl.read_parquet(f) for f in pages], how="vertical_relaxed")
        out["header"] = hdr.filter(pl.col("key") == match_key)
    for name in TOME_CHANNELS:
        if suffix is not None:
            folders = [os.path.join(tome_dir, f"basic_{name}{suffix}")]
        else:
            folders = sorted(glob.glob(os.path.join(tome_dir, f"basic_{name}*")))
        files = [f for folder in folders if os.path.isdir(folder) for f in sorted(glob.glob(os.path.join(folder, "dataframe_*")))]
        if not files:
            continue
        df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
        out[name] = df.filter(pl.col("match_key") == match_key) if "match_key" in df.columns else df
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="P(team wins) at every live second of one match, from a DSDK tome folder.")
    ap.add_argument("--tome", required=True, help="the tome's csds folder (holds header/ and basic_*/)")
    ap.add_argument("--match", required=True, help="match key as it appears in the header")
    ap.add_argument("--team", default="CT", choices=["CT", "T"])
    ap.add_argument("--unit", default="percent", choices=["percent", "fraction"])
    ap.add_argument("--suffix", default=None, help='folder suffix, e.g. ".2026-09-18,2026-10-08" (default: every basic_* folder)')
    ap.add_argument("--out", default=None, help="write {tick: value} as JSON here; without it, print a summary")
    a = ap.parse_args()
    data = load_tome_match(a.tome, a.match, a.suffix)
    probs = win_probability(data, team=a.team, unit=a.unit)
    for w in last_warnings:
        print(f"warning: {w}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in probs.items()}, f)
        print(f"{len(probs)} ticks written to {a.out}")
    elif probs:
        ticks = sorted(probs)
        print(f"{len(probs)} ticks; first {ticks[0]}: {probs[ticks[0]]}; last {ticks[-1]}: {probs[ticks[-1]]}")
    else:
        print("no ticks: nothing could be scored for this match")


if __name__ == "__main__":
    main()
