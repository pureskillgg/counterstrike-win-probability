"""Run context of e2e_checks.py: paths, parameters, lazily loaded tables. Nothing here hard-codes a store, an era, a build or a map."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(HERE)  # scripts/model
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from checks import core  # noqa: E402

# Every threshold is a parameter with a default (override with --param name=value).
DEFAULTS = {
    # --- vacuity: a check on fewer rows than this FAILs (a pass on an empty population proves nothing)
    "min_used_matches": 1, "min_used_rounds": 1, "min_grid_rows": 1, "min_sample_rows": 1,
    # --- game constants of the recorded data (CS2 5v5 competitive)
    "tick_rate": 64, "players_per_round": 10, "players_per_side": 5, "clock_cap_s": 116, "side_t": 2, "side_ct": 3,
    # --- store consistency
    "channels": "round_end,round_start,round_state,player_death,player_info,bomb_state",
    "tome_newer_than_tables_slack_s": 60.0, "manifest_sha_max_mb": 64.0,
    # --- round completeness / numbering (documented known limitation: WARN; FAIL only past a hard bound or a worse-than-baseline share)
    "completeness_hard_bound": 0.90, "completeness_baseline_tol": 0.02, "completeness_min_matches": 20, "more_rounds_than_total_max_share": 0.01,
    # --- labels
    "inferred_side_max_share": 0.10, "wingman_disagree_max_share": 0.005, "recorded_only_max_share": 0.01, "relabel_outside_after_swap_max": 0, "relabel_warn_share": 0.001, "relabel_hard_share": 0.15,
    "reconcile_min_share": 0.995, "reconcile_min_matches": 10, "signal_min_n": 30, "signal_fail_below": 0.85, "signal_warn_below": 0.97,
    # --- duplicates
    "dup_tick_tol": 128, "dup_min_rounds": 16,
    # --- grid
    "grid_recompute_rounds": 30000, "clock_tol": 1e-4, "causal_rounds": 300, "causal_rows_per_round": 0,
    # --- split
    "test_frac": 0.2, "val_frac": 0.1, "split_seed": 0,
    # --- snapshot features / status table
    "bounds_hp_per_player": 100.0, "bounds_armor_per_player": 100.0, "bounds_equip_max": 12000.0, "bounds_grenade_per_player": 4.0,
    "bounds_max_violations": 0, "status_alive_mismatch_max": 0, "status_max_files": 0, "status_fast_files": 3, "status_min_age_s": 120.0,
    # --- model
    "extra_features": "hp_ct,hp_t",
    "repro_tol": 1e-5, "slope_lo": 0.9, "slope_hi": 1.1, "ece_max": 0.02, "leak_margin": 0.01,
    "subgroup_min_rows": 500, "subgroup_abs_tol": 0.02, "subgroup_ece_max": 0.03, "subgroup_ece_min_rows": 5000,
    "discrimination_margin": 0.01, "auc_floor": 0.6, "determinism_tol": 1e-7,
}


class Ctx:
    def __init__(self, root, store, model, P, R, tomes=None, models_dir=None, status_dir=None, sample=None, maps=None,
                 split_manifest=None, baseline=None, fast=False, only=None):
        self.root = os.path.abspath(root)
        self.store, self.model_name, self.P, self.R, self.fast = store, model, P, R, fast
        self.data_dir = os.path.join(self.root, "data", store)
        self.tomes_dir = tomes or os.path.join(self.root, "tomes", store, "tome", "csds")
        self.models_dir = models_dir or os.path.join(self.root, "models")
        self.status_dir = status_dir or os.path.join(self.root, "data", "status_1hz")
        self.sample_path = sample or os.path.join(self.data_dir, f"export_{model}_test_sample.parquet")
        self.maps_arg = [m.strip() for m in maps.split(",") if m.strip()] if maps else None
        self.split_manifest_arg = split_manifest
        self.baseline = baseline or {}
        self.only = [s.strip() for s in only.split(",") if s.strip()] if only else None
        R.only = self.only  # Reporter.run skips the checks of a selected group that the id prefixes do not select
        self._c = {}
        self.measured = {}  # numbers worth recording as a baseline (--write-baseline)

    # ---- selection (--only)
    def want(self, cid):
        """A group (or check) is selected by --only when an id prefix covers it or it covers the id prefix (--only C9 selects group C, and then only check C9)."""
        return self.only is None or any(cid.startswith(p) or p.startswith(cid) for p in self.only)

    # ---- paths
    def p(self, name):
        return os.path.join(self.data_dir, name)

    def missing(self, *names):
        return [n for n in names if not os.path.exists(n if os.path.isabs(n) else self.p(n))]

    @property
    def model_path(self):
        return os.path.join(self.models_dir, self.model_name + ".json")

    @property
    def meta_path(self):
        return os.path.join(self.models_dir, self.model_name + ".meta.json")

    # ---- cached tables
    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    @property
    def pl(self):
        import polars as pl
        return pl

    @property
    def matches(self):
        return self._get("matches", lambda: self.pl.read_parquet(self.p("matches.parquet")))

    @property
    def rounds(self):
        return self._get("rounds", lambda: self.pl.read_parquet(self.p("rounds.parquet")))

    @property
    def used(self):
        pl = self.pl
        return self._get("used", lambda: self.rounds.filter(pl.col("use").fill_null(False)))

    @property
    def used_matches(self):
        pl = self.pl
        return self._get("used_matches", lambda: self.matches.filter(pl.col("match_key").is_in(self.used["match_key"].unique().to_list())))

    @property
    def meta(self):
        def load():
            if not os.path.isfile(self.meta_path):
                return None
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        return self._get("meta", load)

    @property
    def maps(self):
        """The maps in scope: --maps, else the model's scope.maps / map_is_* features. None when neither is known."""
        def derive():
            if self.maps_arg:
                return self.maps_arg
            m = self.meta
            if m:
                sc = (m.get("scope") or {}).get("maps")
                if sc:
                    return list(sc)
                fs = [f[len("map_is_"):] for f in m.get("features", []) if f.startswith("map_is_")]
                if fs:
                    return ["de_" + x for x in fs]
            return None
        return self._get("maps", derive)

    @property
    def wp(self):
        """wp_common, pointed at this run's root."""
        def load():
            import wp_common as W
            W.ROOT = self.root
            return W
        return self._get("wp", load)

    def grid_scan(self, cols=None):
        lf = self.pl.scan_parquet(self.p("grid.parquet"))
        return lf.select(cols) if cols else lf

    def split_manifest_path(self):
        """--split-manifest, else the store's own data/<store>/split_manifest.json when it exists; 'none' disables."""
        a = self.split_manifest_arg
        if a is not None:
            return None if str(a).lower() == "none" else a
        cand = self.p("split_manifest.json")
        return cand if os.path.isfile(cand) else None

    def channel_dir(self, ch):
        d = os.path.join(self.tomes_dir, f"basic_{ch}.COMBINED")
        if os.path.isdir(d):
            return d
        c = [x for x in sorted(glob.glob(os.path.join(self.tomes_dir, f"basic_{ch}.*"))) if os.path.isdir(x)]
        return c[0] if len(c) == 1 else None

    def header_dir(self):
        d = os.path.join(self.tomes_dir, "header")
        return d if os.path.isdir(d) else None

    def baseline_get(self, section, key):
        return (self.baseline.get(section) or {}).get(key)

    def record(self, section, key, value):
        self.measured.setdefault(section, {})[key] = value
