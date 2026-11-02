"""CPU inference for the CS2 round win-probability model. Meant to be imported by the live service.

    from winprob import WinProb
    wp = WinProb.load("models/winprob_v1.json", fallback_path="models/winprob_v1_basic.json")
    out = wp.predict({"seconds_since_round_live": 71.4, "bomb_planted": 1, "seconds_since_plant": 12.0,
                      "players": [{"side": "CT", "health": 80, "armor": 100, "has_helmet": True, "has_defuser": True,
                                   "primary": 16, "flashbang": 1, "smokegrenade": 1, "equipment_value": 5600}, ...],  # ALIVE players
                      "map_name": "de_mirage", "mode": "competitive"})
    out["p_ct_win"]        # probability that CT win the round: always a finite float in [0, 1]
    wp.explain(state)      # the same plus the contribution of every input to the log-odds (exact tree SHAP)

Two ways to call it
  * snapshot model  : give the table of living players (health, armor, helmet, kit, primary weapon, grenades, equipment value),
                      the bomb state and the clock. Alive counts come from the table. This is the best model.
  * alive-count mode: no player table, only alive_ct / alive_t / bomb / clock (and any optional extra inputs the basic model
                      lists). Used automatically when the table is missing or unusable and a fallback model is loaded.

Contract (checked by scripts/model/stress_test_winprob.py)
  * predict() / predict_many() / explain() NEVER raise. Whatever arrives (missing keys, strings, NaN, negative or huge numbers, a
    corrupt player table, an unknown map, wingman or hostage mode, 2v2 alive counts) the result is a dict with a finite p_ct_win in
    [0, 1], a `status`, a `reliable` flag and a list of `warnings`. A model output that is not finite is never served as a number:
    the prior comes back as invalid_input.
  * "ok"            valid input in scope: the model's number.
  * "fallback"      the player table was missing or invalid (or the snapshot model lacks an input the state cannot supply), so the
                    alive-count model answered (less detail, still calibrated).
  * "terminal"      the round is already decided, a state the model was never trained on (training stops before the decisive
                    event): bomb_defused true -> 1.0 (the caller owns the event; the flag is accepted, 0/1/true/false text too);
                    CT dead -> 0.0; T dead with no plant -> 1.0; bomb planted 41 s ago or more (it has exploded) -> 0.0;
                    no plant and 115 s or more on the round clock (time ran out) -> 1.0. The 2026 grid has no unplanted row past
                    second 114 and no seconds_since_plant above 40.98, so these are rules, not model answers. T dead with the
                    bomb down and still inside the timer goes to the model (CT must still defuse). `reliable` is False when the
                    map or mode is out of scope. This class answers states, it does not track a round.
  * Units            seconds_since_round_live and seconds_since_plant are SECONDS (use seconds_from_ticks() for ticks). A clock
                    beyond +-MAX_SECONDS (200 s) is not a round clock in seconds (ticks, milliseconds): invalid_input, never a
                    confident terminal. A planted bomb with seconds_since_plant above the round clock (plant before the round
                    started) or planted later than 125 s into the round (the 2026 grid's latest plant is at 115.0 s, the 2024
                    grid's at 120.8 s) is an impossible clock: invalid_input. Text numbers are
                    accepted with a warning. Health and armor must be 0-100 points: a table whose living players all have values
                    in 0..1 with a fraction (0.8) is a unit error and makes the table unusable; so does a bool where a number
                    belongs. A value outside the physical range is clamped, warned, and the answer is marked reliable False.
  * Input rules      the player table lists LIVING players: a row with health <= 0 is dropped (before its other fields are looked at,
                    so a dead player with null loadout is fine) and warned; spectator / unassigned rows (side 0, 1, 'spec', ...) are
                    ignored and warned and never count as a team; any other unknown side makes the table unusable. A living row
                    missing side, health, armor, primary or equipment_value makes the whole table unusable (fallback model, never
                    zeros). The other keys (has_helmet, has_defuser, is_defusing, flashbang, smokegrenade, hegrenade, molotov,
                    incgrenade, decoy) are optional and count as 0/False when absent; the warning lists which keys were absent,
                    and the answer is marked reliable False when a key is absent from EVERY row and the model uses its feature.
                    bomb_planted is required (absent or not a clear yes/no -> invalid_input); with a plant, seconds_since_plant is
                    required and must be >= 0. Counts must be >= 0. A model input the state cannot supply is never sent as NaN:
                    the alive-count model answers instead (fallback) or, with no fallback, invalid_input.
  * Scope            the in-scope maps and modes come from the loaded model's meta.json (scope.maps / scope.modes, else its map_is_*
                    features); IN_SCOPE_MAPS / IN_SCOPE_MODES are only the defaults when the meta has none ('premier' counts as
                    'competitive'). A map or mode that is blank / absent is NOT a silent 'ok': it is out_of_scope with reliable False
                    and a 'map not given' / 'mode not given' warning. A map listed in scope without a map_is_ input in the answering
                    model is out_of_scope too (it would be answered as the all-zero 'other' row).
  * "out_of_scope"  a map or mode outside the trained scope (or not given), or more than 5 players on a side: the model still
                    returns its number, flagged `reliable: False`.
  * "invalid_input" required inputs missing or not numbers and no fallback possible, an impossible clock, a unit error, or a
                    non-finite model output: the prior (base rate) is returned and `reliable` is False.
"""
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # single-row serving: a one-thread pool is much faster than spinning up all cores
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import threading

import numpy as np

from snapshot_features import _side, loadout_from_players
from wp_items import AK, AWP, HEAVY, M4, RIFLE, SIDE_CT, SIDE_T, SMG, SNIPER_OTHER

IN_SCOPE_MAPS = ("de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke")
IN_SCOPE_MODES = ("competitive", "premier", "")
MODE_ALIASES = {"premier": "competitive"}  # same 5v5 rules; counted in scope whenever the alias target is
TICK_RATE = 64
MAX_SECONDS = 200.0   # a round is 115 s plus the 40 s post-plant timer (155 s); a clock beyond +-200 s is not seconds (ticks, ms): invalid_input
# The training rows stop before the decisive event, so the model has never seen a state past the bomb timer or past the round clock
# (2026 grid: seconds_since_plant <= 40.98; unplanted rows end at second 114; the 2024 era has 3,277 late-logged unplanted rows at 115-123 s).
# Such states are answered by rule, not by the model:
BOMB_TIMER_MAX = 41.0     # planted and this many seconds ago or more: the bomb has exploded -> T win (terminal, p_ct_win 0)
ROUND_CLOCK_MAX = 115.0   # not planted and this many seconds live or more: the clock ran out -> CT win (terminal, p_ct_win 1)
CLOCK_SLACK = 1.0         # seconds_since_plant may exceed seconds_since_round_live by this much before the clock is impossible (plant before the round)
PLANT_LATEST_MAX = 125.0  # latest plausible plant time (seconds_since_round_live - seconds_since_plant): 115 s clock, 2026 max 115.0, 2024 max 120.8 (late-logged)
PLAYER_REQUIRED = ("side", "health", "armor", "primary", "equipment_value")  # a living row missing one of these makes the table unusable
# optional per-player keys (absent = 0 / False) and the snapshot aggregate each one feeds, so the predictor can tell whether the model uses it
PLAYER_OPTIONAL = {"has_helmet": ("helmet_n",), "has_defuser": ("kit_n",), "is_defusing": ("defusing",), "flashbang": ("flash",),
                   "smokegrenade": ("smoke",), "hegrenade": ("he",), "molotov": ("fire",), "incgrenade": ("fire",), "decoy": ("decoy",)}
NON_PLAYER_SIDES = ("0", "1", "spec", "spectator", "unassigned", "coach", "none")  # rows that are not on a team: ignored, never CT or T
KNOWN_PRIMARIES = frozenset((0,) + AK + M4 + AWP + RIFLE + SNIPER_OTHER + SMG + HEAVY)
_SNAPSHOT_NAMES = frozenset(loadout_from_players([]))  # every input name the player table can supply


def seconds_from_ticks(tick, freeze_end_tick):
    """Fractional seconds since the round went live at any tick (the model was trained on whole seconds)."""
    return (float(tick) - float(freeze_end_tick)) / TICK_RATE


def _num(x):
    """A finite float, or None. Booleans are not numbers here (a True where a count belongs is a bug, not a 1)."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


_TRUE = ("1", "true", "yes", "y", "t")
_FALSE = ("0", "false", "no", "n", "f")


def _flag(v):
    """1.0 / 0.0 for a clear yes/no (bool, the numbers 1 and 0, or yes/no style text); None for anything else."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, str):
        k = v.strip().lower()
        return 1.0 if k in _TRUE else 0.0 if k in _FALSE else None
    n = _num(v)
    if n is None:
        return None
    return 1.0 if n == 1.0 else 0.0 if n == 0.0 else None


# Physically possible ranges of the per-player inputs. The 31 M training snapshots stay inside them (health 1-100, armor 0-100,
# equipment value <= 8,700, at most 2 flashbangs and 1 of each other grenade), so clamping changes nothing for real data and
# only tames garbage.
PLAYER_RANGE = {"health": (0.0, 100.0), "armor": (0.0, 100.0), "equipment_value": (0.0, 12000.0), "flashbang": (0.0, 4.0),
                "smokegrenade": (0.0, 4.0), "hegrenade": (0.0, 4.0), "molotov": (0.0, 4.0), "incgrenade": (0.0, 4.0), "decoy": (0.0, 4.0)}


def _player_number(q, k):
    """The finite float behind q[k]. A bool where a number belongs, text that is not a number, or a non-finite value raises."""
    v = q[k]
    if isinstance(v, bool):
        raise ValueError(f"{k} is a bool, not a number")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"non-finite {k}")
    return f


def _is_fraction(values):
    """True when every value is within 0..1 and at least one is not a whole number: a 0..1 scale where points (0-100) belong."""
    return bool(values) and max(values) <= 1.0 and any(v != math.floor(v) for v in values)


def _clean_players(players):
    """Copies of the LIVING player dicts with out-of-range numbers clamped. Returns (clean_list, info) where info counts what was
    done: clamped values, dead rows dropped, spectator rows ignored, the unknown primary ids, and the optional keys absent per key.
    A row with health <= 0 is a dead player and is dropped BEFORE its other fields are read (a dead row may carry nulls); a row
    on no team (spectator, unassigned) is ignored. Raises on a player that is not a dict, lacks health or (living) one of
    PLAYER_REQUIRED, holds a bool / text / non-finite where a number belongs, or when the table is on the wrong scale (health or
    armor as a 0..1 fraction): the caller then treats the table as unusable (fallback model) rather than guessing numbers."""
    clean = []
    info = {"clamped": 0, "dead": 0, "spectators": 0, "unknown_primary": set(), "absent": {}}
    raw_health, raw_armor = [], []
    for p in players:
        if not isinstance(p, dict):
            raise ValueError("player is not a dict")
        if p.get("health") is None:
            raise ValueError("player row lacks health")
        q = dict(p)
        f = _player_number(q, "health")
        lo, hi = PLAYER_RANGE["health"]
        c = min(hi, max(lo, f))
        if c != f:
            info["clamped"] += 1
        q["health"] = c
        if float(q["health"]) <= 0.0:
            info["dead"] += 1
            continue
        if q.get("side") is None:
            raise ValueError("player row lacks side")
        side_key = q["side"].strip().lower() if isinstance(q["side"], str) else q["side"]
        if not isinstance(side_key, bool) and isinstance(side_key, (str, int)) and str(side_key) in NON_PLAYER_SIDES:
            info["spectators"] += 1
            continue
        for k in PLAYER_REQUIRED:
            if q.get(k) is None:
                raise ValueError(f"player row lacks {k}")
        raw_health.append(f)
        for k, (lo, hi) in PLAYER_RANGE.items():
            if k == "health" or q.get(k) is None:
                continue
            f = _player_number(q, k)
            if k == "armor":
                raw_armor.append(f)
            c = min(hi, max(lo, f))
            if c != f:
                info["clamped"] += 1
                q[k] = c
        prim = _player_number(q, "primary")
        if prim < 0 or prim != math.floor(prim):
            raise ValueError("primary is not an item id")
        if int(prim) not in KNOWN_PRIMARIES:
            info["unknown_primary"].add(int(prim))
        for k in PLAYER_OPTIONAL:
            if q.get(k) is None:
                info["absent"][k] = info["absent"].get(k, 0) + 1
        clean.append(q)
    if _is_fraction(raw_health):
        raise ValueError("health looks like a 0..1 fraction (points 0-100 expected)")
    if _is_fraction(raw_armor):
        raise ValueError("armor looks like a 0..1 fraction (points 0-100 expected)")
    return clean, info


def _label(v):
    """None or blank text means 'not given'; any other non-text value is an unknown label, never a silent default."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return ""
    return v.strip().lower() if isinstance(v, str) else "<non-text value>"


def _map_col(name):
    """'de_mirage' -> 'map_is_mirage' (the one-hot input of that map)."""
    return "map_is_" + (name[3:] if name[2:3] == "_" else name)


def _scope_of(meta, features):
    """(maps, modes, source) the model was trained for: meta.scope first, then the map_is_* features, the module defaults last."""
    sc = meta.get("scope") if isinstance(meta.get("scope"), dict) else {}
    maps = sc.get("maps")
    if isinstance(maps, (list, tuple)) and maps and all(isinstance(m, str) and m.strip() for m in maps):
        maps_t, src = tuple(m.strip().lower() for m in maps), "meta.scope.maps"
    else:
        cols = [f for f in features if f.startswith("map_is_")]
        if cols:
            maps_t, src = tuple("de_" + c[len("map_is_"):] for c in cols), "meta.features map_is_*"
        else:
            maps_t, src = IN_SCOPE_MAPS, "default constant"
    modes = sc.get("modes")
    if isinstance(modes, (list, tuple)) and modes and all(isinstance(m, str) and m.strip() for m in modes):
        modes_t = tuple(m.strip().lower() for m in modes)
    else:
        modes_t = tuple(m for m in IN_SCOPE_MODES if m)
    modes_t += tuple(a for a, t in MODE_ALIASES.items() if t in modes_t and a not in modes_t)
    return maps_t, modes_t, src


class _Model:
    def __init__(self, booster, meta):
        self.booster = booster
        self.meta = meta
        self.features = list(meta["features"])
        self.feature_set = frozenset(self.features)
        self.best_iteration = int(meta["best_iteration"])
        inp = meta.get("input")  # an explicit "players" / "counts" in the meta wins over the feature names
        self.needs_players = (inp == "players") if inp in ("players", "counts") else any(f.startswith("st_") or f in _SNAPSHOT_NAMES for f in self.features)
        self.map_cols = tuple(f for f in self.features if f.startswith("map_is_"))
        self.lock = threading.Lock()  # Booster.predict is not documented as thread-safe

    @classmethod
    def load(cls, path):
        import xgboost as xgb
        b = xgb.Booster()
        b.set_param({"device": "cpu", "nthread": 1})  # single-row calls: one thread beats spinning up the pool
        b.load_model(path)
        with open(os.path.splitext(path)[0] + ".meta.json", encoding="utf-8") as f:
            return cls(b, json.load(f))

    def row(self, vals):
        return np.array([[vals.get(f, np.nan) for f in self.features]], dtype=np.float32)

    def predict_rows(self, X):
        with self.lock:
            return self.booster.inplace_predict(X, iteration_range=(0, self.best_iteration + 1))

    def contribs(self, x):
        import xgboost as xgb
        with self.lock:
            return self.booster.predict(xgb.DMatrix(x, nthread=1), pred_contribs=True, iteration_range=(0, self.best_iteration + 1))[0]


class WinProb:
    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback
        self.meta = primary.meta
        self.features = primary.features
        try:
            prior = float(primary.meta.get("prior_ct_win", 0.5))
        except (TypeError, ValueError):
            prior = 0.5
        self.prior = prior if math.isfinite(prior) and 0.0 <= prior <= 1.0 else 0.5
        self.scope_maps, self.scope_modes, self.scope_source = _scope_of(primary.meta, primary.features)
        cols = {_map_col(m) for m in self.scope_maps}
        self.scope_issues = [f"scope map {m} has no model input {_map_col(m)}: it would be answered as the all-zero 'other' row"
                             for m in self.scope_maps if primary.map_cols and _map_col(m) not in primary.feature_set]
        self.scope_issues += [f"model input {c} has no map in scope" for c in primary.map_cols if c not in cols]
        if fallback is not None and fallback.map_cols:
            self.scope_issues += [f"fallback model has no input {_map_col(m)} for scope map {m}" for m in self.scope_maps if _map_col(m) not in fallback.feature_set]
        # which optional player key feeds a feature the (snapshot) model actually lists
        self._opt_used = {k: any(f.startswith(f"st_{fld}_") for f in primary.features for fld in fields) for k, fields in PLAYER_OPTIONAL.items()}

    @classmethod
    def load(cls, path, fallback_path=None):
        return cls(_Model.load(path), _Model.load(fallback_path) if fallback_path else None)

    # ------------------------------------------------------------------ public
    def predict(self, state):
        try:
            prep = self._prepare(state)
            if "early" in prep:
                return prep["early"]
            p = float(prep["model"].predict_rows(prep["x"])[0])
            return self._out(p, prep["status"], prep["reliable"], prep["warn"])
        except Exception as exc:  # the contract: never raise
            return self._out(self.prior, "invalid_input", False, [f"internal error: {type(exc).__name__}"])

    def predict_many(self, states):
        """Same results as predict() for every state, with one booster call per model for all the ordinary rows."""
        try:
            states = list(states)
        except Exception:  # not iterable: nothing to predict, and never raise
            return []
        out = [None] * len(states)
        groups = {}
        for i, s in enumerate(states):
            try:
                prep = self._prepare(s)
            except Exception as exc:
                out[i] = self._out(self.prior, "invalid_input", False, [f"internal error: {type(exc).__name__}"])
                continue
            if "early" in prep:
                out[i] = prep["early"]
            else:
                groups.setdefault(id(prep["model"]), (prep["model"], []))[1].append((i, prep))
        for model, items in groups.values():
            try:
                p = model.predict_rows(np.vstack([pr["x"][0] for _, pr in items]))
                for (i, pr), v in zip(items, p):
                    out[i] = self._out(float(v), pr["status"], pr["reliable"], pr["warn"])
            except Exception:
                for i, _ in items:
                    out[i] = self.predict(states[i])
        return out

    def explain(self, state):
        """predict() plus the contribution of every input to the log-odds of a CT win. contributions[feature] > 0 pushes toward
        CT; base_logit is the model's average starting point; base_logit + sum(contributions) = logit(p_ct_win). Never raises.
        For terminal / invalid states contributions is {}."""
        try:
            prep = self._prepare(state)
            if "early" in prep:
                out = dict(prep["early"])
                out.update({"contributions": {}, "base_logit": None})
                return out
            m = prep["model"]
            c = m.contribs(prep["x"])
            tot = float(c.sum())
            p = float("nan") if not math.isfinite(tot) else float(1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, tot)))))
            out = self._out(p, prep["status"], prep["reliable"], prep["warn"])
            if out["status"] == "invalid_input" and prep["status"] != "invalid_input":  # non-finite model output: no contributions to show
                out.update({"contributions": {}, "base_logit": None})
                return out
            out["contributions"] = {f: float(v) for f, v in zip(m.features, c[:-1])}
            out["base_logit"] = float(c[-1])
            return out
        except Exception as exc:
            out = self._out(self.prior, "invalid_input", False, [f"internal error: {type(exc).__name__}"])
            out.update({"contributions": {}, "base_logit": None})
            return out

    def describe(self):
        """What this predictor was loaded with (for reports): version, scope and where it came from, inputs, issues found at load."""
        return {"version": self.meta.get("version", "?"), "maps": list(self.scope_maps), "modes": list(self.scope_modes),
                "scope_source": self.scope_source, "features": list(self.features), "needs_players": self.primary.needs_players,
                "fallback": None if self.fallback is None else self.fallback.meta.get("version", "?"), "scope_issues": list(self.scope_issues)}

    # ------------------------------------------------------------------ internals
    def _out(self, p, status, reliable, warnings):
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = float("nan")
        if not math.isfinite(p):  # a non-finite model output is never served as an answer
            return {"p_ct_win": self.prior, "status": "invalid_input", "reliable": False,
                    "warnings": list(warnings) + ["the model output was not a finite number; the prior is returned"],
                    "model_version": self.meta.get("version", "?")}
        p = float(min(1.0, max(0.0, p)))
        return {"p_ct_win": p, "status": status, "reliable": bool(reliable), "warnings": warnings,
                "model_version": self.meta.get("version", "?")}

    def _invalid(self, warn, msg):
        return {"early": self._out(self.prior, "invalid_input", False, warn + [msg])}

    @staticmethod
    def _read(s, key, warn):
        """A finite number from s[key] or None; text numbers are accepted with a warning."""
        v = s.get(key)
        n = _num(v)
        if n is not None and isinstance(v, str):
            warn.append(f"{key} was given as text; converted to a number")
        return n

    def _inputs(self, model, vals, map_name):
        """The full input dict for one model: the shared values plus that model's own map one-hot (an unknown / blank map is all zeros)."""
        v = dict(vals)
        for c in model.map_cols:
            v[c] = 1.0 if map_name and c == _map_col(map_name) else 0.0
        return v

    def _prepare(self, s):
        """Validate and clamp a state. Returns {"early": result} for invalid/terminal states, else {"model", "x", ...}."""
        if not isinstance(s, dict):
            return {"early": self._out(self.prior, "invalid_input", False, ["state is not a dict"])}
        warn = []
        out_of_scope = False
        mode = _label(s.get("mode"))
        if not mode:
            out_of_scope = True
            warn.append("mode not given: the competitive 5v5 scope cannot be confirmed")
        elif mode not in self.scope_modes:
            out_of_scope = True
            warn.append(f"mode '{mode[:40]}' is outside the competitive 5v5 scope")
        map_name = _label(s.get("map_name"))
        if not map_name:
            out_of_scope = True
            warn.append("map not given: the map one-hot is all zeros, a row never seen in training")
        elif map_name not in self.scope_maps:
            out_of_scope = True
            warn.append(f"map '{map_name[:40]}' is outside the {len(self.scope_maps)} trained maps")

        secs = self._read(s, "seconds_since_round_live", warn)
        if secs is None:
            return self._invalid(warn, "seconds_since_round_live is a required number")
        if abs(secs) > MAX_SECONDS:
            return self._invalid(warn, f"seconds_since_round_live {secs:g} is beyond +-{MAX_SECONDS:g} s: not seconds (ticks or milliseconds? see seconds_from_ticks)")
        if secs < 0:
            warn.append("seconds_since_round_live < 0 (before the round went live): clamped to 0")
        secs = max(0.0, secs)

        # snapshot features from the living-player table, when the table is usable
        snap, players_ok, table_note, table_degraded = None, False, None, False
        players = s.get("players")
        if isinstance(players, (list, tuple)) and 0 < len(players) <= 40:
            try:
                clean, info = _clean_players(players)
                snap = loadout_from_players(clean)
                sides = [_side(p.get("side")) for p in clean]
                n_ct, n_t = sides.count(SIDE_CT), sides.count(SIDE_T)
                players_ok = True
                if info["clamped"]:
                    warn.append(f"{info['clamped']} player value(s) outside the physical range were clamped (units?)")
                    table_degraded = True
                if info["dead"]:
                    warn.append(f"{info['dead']} row(s) with health <= 0 were dropped (the table should list living players only)")
                if info["spectators"]:
                    warn.append(f"{info['spectators']} spectator / unassigned row(s) were ignored (they are on no team)")
                if info["unknown_primary"]:
                    warn.append(f"unknown primary item id(s) {sorted(info['unknown_primary'])[:5]}: counted in no weapon group")
                if info["absent"] and self.primary.needs_players:
                    warn.append("optional player key(s) absent, counted as 0/False: " + ", ".join(f"{k} ({n} of {len(clean)} rows)" for k, n in sorted(info["absent"].items())))
                    every = sorted(k for k, n in info["absent"].items() if n == len(clean) and self._opt_used.get(k))
                    if every:
                        warn.append("absent from EVERY row while the model uses it: " + ", ".join(every) + " (answer not reliable)")
                        table_degraded = True
            except Exception as exc:
                snap = None
                table_note = str(exc)[:90] if isinstance(exc, ValueError) else type(exc).__name__
        elif players is not None:
            table_note = "not a list of 1-40 rows"
        if players_ok:
            ac, at = float(n_ct), float(n_t)
            if ac + at == 0:
                return self._invalid(warn, "the player table holds no living player")
            given_ac, given_at = _num(s.get("alive_ct")), _num(s.get("alive_t"))
            if (given_ac is not None and given_ac != ac) or (given_at is not None and given_at != at):
                warn.append("alive_ct / alive_t differ from the player table; the table was used")
        else:
            ac, at = self._read(s, "alive_ct", warn), self._read(s, "alive_t", warn)
            if ac is None or at is None or ac < 0 or at < 0:
                if players is not None:
                    warn.append("player table unusable" + (f" ({table_note})" if table_note else ""))
                return self._invalid(warn, "need a valid player table, or alive_ct and alive_t as non-negative numbers")
            if ac != round(ac) or at != round(at):
                warn.append("alive_ct / alive_t are not whole numbers; rounded")
            if players is not None:
                warn.append("player table unusable" + (f" ({table_note})" if table_note else ""))
        if ac > 5 or at > 5:
            out_of_scope = True
            warn.append("more than 5 alive on a side (not a 5v5 round); clamped to 5")
        ac, at = min(5.0, round(ac)), min(5.0, round(at))

        if "bomb_planted" not in s:
            return self._invalid(warn, "bomb_planted not given: the bomb state is a required input")
        planted = _flag(s.get("bomb_planted"))
        if planted is None:  # a text like 'maybe', NaN, None, or a number other than 0 and 1: never guess which way it falls
            return self._invalid(warn, "bomb_planted must be 0/1 or true/false")
        if planted:
            sp = self._read(s, "seconds_since_plant", warn)
            if sp is None or sp < 0:
                return self._invalid(warn, "bomb planted but seconds_since_plant is missing or negative")
            if sp > MAX_SECONDS:
                return self._invalid(warn, f"seconds_since_plant {sp:g} is beyond {MAX_SECONDS:g} s: not seconds (ticks or milliseconds?)")
            if sp > secs + CLOCK_SLACK or secs - sp > PLANT_LATEST_MAX:
                return self._invalid(warn, f"impossible clock: bomb planted {sp:g} s ago on a round clock of {secs:g} s (plant before the round started or after it ended)")
        else:
            sp = -1.0
        defused = None
        if s.get("bomb_defused") is not None:  # optional event flag (the caller owns events): a clear yes/no or invalid_input
            defused = _flag(s.get("bomb_defused"))
            if defused is None:
                return self._invalid(warn, "bomb_defused must be 0/1 or true/false")

        reliable = not out_of_scope
        if defused:
            if not planted:
                warn.append("bomb_defused is true but bomb_planted is 0")
            return {"early": self._out(1.0, "terminal", reliable, warn + ["bomb defused: the round is already decided (CT win)"])}
        if ac == 0:
            return {"early": self._out(0.0, "terminal", reliable, warn)}
        if at == 0 and not planted:
            return {"early": self._out(1.0, "terminal", reliable, warn)}
        if planted and sp >= BOMB_TIMER_MAX:  # at or past the bomb timer: it has exploded
            return {"early": self._out(0.0, "terminal", reliable, warn + [f"bomb planted {BOMB_TIMER_MAX:.0f} s ago or more: the round is already decided (T win)"])}
        if not planted and secs >= ROUND_CLOCK_MAX:  # at or past the round clock with no plant: time ran out
            return {"early": self._out(1.0, "terminal", reliable, warn + [f"round clock at {ROUND_CLOCK_MAX:.0f} s or more with no plant: the round is already decided (CT win)"])}

        vals = {"alive_ct": ac, "alive_t": at, "ct_alive_minus_t": ac - at, "bomb_planted": planted,
                "seconds_since_round_live": secs, "seconds_since_plant": sp}
        if snap is not None:
            vals.update(snap)
        for k, v in s.items():  # optional extra inputs a model may list (e.g. hp_ct, hp_t): numbers only, never overwrite the above
            if k not in vals and k != "players" and isinstance(k, str):
                n = _num(v)
                if n is not None:
                    vals[k] = n

        model, status = self.primary, ("out_of_scope" if out_of_scope else "ok")
        if model.needs_players and snap is None:
            if self.fallback is None:
                return self._invalid(warn, "this model needs a valid player table")
            model = self.fallback
            status = "out_of_scope" if out_of_scope else "fallback"
            warn.append("no usable player table: answered by the alive-count model" + (f" ({table_note})" if table_note else ""))
        full = self._inputs(model, vals, map_name)
        missing = [f for f in model.features if f not in full]
        if missing and model is self.primary and self.fallback is not None and all(f in self._inputs(self.fallback, vals, map_name) for f in self.fallback.features):
            warn.append(f"{len(missing)} model input(s) not given (e.g. {missing[0]}): answered by the alive-count model")
            model = self.fallback
            status = "out_of_scope" if out_of_scope else "fallback"
            full = self._inputs(model, vals, map_name)
            missing = []
        if missing:  # never send NaN to the booster as if it were a measurement
            return self._invalid(warn, f"{len(missing)} model input(s) not given: {', '.join(missing[:5])}")
        if map_name in self.scope_maps and model.map_cols and _map_col(map_name) not in model.feature_set:
            out_of_scope, status = True, "out_of_scope"
            warn.append(f"map {map_name} is in scope but the model has no input {_map_col(map_name)}: answered as 'other'")
        degraded = table_degraded and model.needs_players and snap is not None
        return {"model": model, "x": model.row(full), "status": status, "reliable": not out_of_scope and not degraded, "warn": warn}
