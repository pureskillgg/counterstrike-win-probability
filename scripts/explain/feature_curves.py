"""Per-input effect curves for the 83-input model, from the saved TreeSHAP sample (no retraining).

For every input: its SHAP contribution converted to percentage points per second (100 * [sigmoid(z) - sigmoid(z - phi)]),
binned against the input's value, plus a summary per input:
  slope_pp_per_unit   linear slope of the effect against the value (pp per unit of the input)
  low_decile_pp       mean effect over rows in the bottom 10% of the value
  high_decile_pp      mean effect over rows in the top 10% of the value
  change_p10_p90_pp   high_decile_pp minus low_decile_pp (moving from low to high values)

Run after `xgb83_explain.py --stage explain` (needs WORK/shap_sample.npz and WORK/xs.npy).
Outputs: docs/explainability/tables/feature_curves_all83.csv (binned) and feature_slopes_all83.csv (summary),
and figures under docs/explainability/figures/.
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb83_explain import DOCS, GROUPS, MDIR, NOTES, WORK, sig  # noqa: E402

UNITS = {
    "st_hp": "team HP sum (0 to 500)",
    "st_equip": "team equipment value (dollars)",
    "st_armor_n": "players carrying armor (0 to 5)",
    "st_armor_sum": "team armor points (0 to 500)",
    "st_helmet_n": "players with helmet (0 to 5)",
    "st_kit_n": "players with defuse kit (0 to 5)",
    "st_defusing": "players defusing (0 or 1)",
    "st_ak": "AK-47 count (0 to 5)",
    "st_m4": "M4 count (0 to 5)",
    "st_awp": "AWP count (0 to 5)",
    "st_rifle": "rifle count (0 to 5)",
    "st_sniper_other": "other sniper count (0 to 5)",
    "st_smg": "SMG count (0 to 5)",
    "st_heavy": "heavy weapon count (0 to 5)",
    "st_no_primary": "players with no primary weapon (0 to 5)",
    "st_flash": "flashbang count (0 to 5)",
    "st_smoke": "smoke count (0 to 5)",
    "st_he": "HE grenade count (0 to 5)",
    "st_fire": "fire grenade count (0 to 5)",
    "st_decoy": "decoy count (0 to 5)",
}


def unit_of(name):
    if name.endswith("_diff"):
        base = name[: -len("_diff")]
        side = "CT minus T"
    elif name.endswith("_ct") or name.endswith("_t"):
        base = name[: -len("_ct")] if name.endswith("_ct") else name[: -len("_t")]
        side = "CT" if name.endswith("_ct") else "T"
    else:
        base, side = None, None
    if base in UNITS:
        return f"{UNITS[base]}, {side}"
    if name in ("alive_ct", "alive_t"):
        return "players alive (0 to 5)"
    if name == "ct_alive_minus_t":
        return "CT alive minus T alive"
    if name in ("bomb_planted", "map_is_ancient", "map_is_anubis", "map_is_cache", "map_is_dust2",
                "map_is_inferno", "map_is_mirage", "map_is_nuke"):
        return "indicator (0 or 1)"
    if name in ("seconds_since_round_live", "seconds_since_plant"):
        return "seconds (-1 when no bomb planted)" if name == "seconds_since_plant" else "seconds"
    if name == "defuse_slack":
        return "seconds (defuse race slack; missing when no bomb planted)"
    if "dist" in name or "spread" in name or "nearest" in name or "site" in name or "enemy_dist" in name:
        return "distance (game units)"
    if "n_bomb" in name or "sees" in name or "exposed" in name or "in_contact" in name or "blind_n" in name:
        return "count"
    if "blind_sum" in name:
        return "sum (cache units)"
    return "cache units"


def main():
    meta = json.load(open(os.path.join(MDIR, "meta.json")))
    feats = meta["feats_all"]
    S = np.load(os.path.join(WORK, "shap_sample.npz"))
    phi = S["phi"].astype(np.float64)
    z = S["z"].astype(np.float64)
    xs = np.load(os.path.join(WORK, "xs.npy")).astype(np.float64)
    p = sig(z)
    grp_of = {f: g for g, cols in GROUPS.items() for f in cols}
    tables = os.path.join(DOCS, "tables")
    os.makedirs(tables, exist_ok=True)

    long_rows, summary_rows = [], []
    for j, f in enumerate(feats):
        x = xs[:, j]
        pp = 100.0 * (p - sig(z - phi[:, j]))
        ok = ~np.isnan(x)
        valid = x[ok]
        uniq = np.unique(valid)

        # binned curve
        if len(uniq) <= 15:
            bins = [(u, u) for u in uniq]
            masks = [ok & (x == u) for u, _ in bins]
        else:
            edges = np.unique(np.quantile(valid, np.linspace(0, 1, 21)))
            bins = list(zip(edges[:-1], edges[1:]))
            masks = []
            for k, (lo, hi) in enumerate(bins):
                last = k == len(bins) - 1
                masks.append(ok & (x >= lo) & (x <= hi if last else x < hi))
        for (lo, hi), m in zip(bins, masks):
            if m.sum() == 0:
                continue
            long_rows.append({"feature": f, "bundle": grp_of[f], "unit": unit_of(f), "x_lo": float(lo), "x_hi": float(hi),
                              "n": int(m.sum()), "mean_value": float(x[m].mean()),
                              "mean_signed_effect_pp": float(pp[m].mean()), "mean_abs_effect_pp": float(np.abs(pp[m]).mean())})
        if (~ok).any():
            m = ~ok
            long_rows.append({"feature": f, "bundle": grp_of[f], "unit": unit_of(f), "x_lo": None, "x_hi": None,
                              "n": int(m.sum()), "mean_value": None, "mean_signed_effect_pp": float(pp[m].mean()),
                              "mean_abs_effect_pp": float(np.abs(pp[m]).mean())})

        # summary
        if len(valid) > 10 and len(uniq) > 1:
            slope = float(np.polyfit(valid, pp[ok], 1)[0])
            p10, p50, p90 = np.percentile(valid, [10, 50, 90])
            low = float(pp[ok & (x <= p10)].mean())
            high = float(pp[ok & (x >= p90)].mean())
        else:
            slope = p10 = p50 = p90 = low = high = None
        summary_rows.append({"feature": f, "bundle": grp_of[f], "unit": unit_of(f), "note": NOTES.get(f, ""),
                             "n_unique": int(len(uniq)), "nan_rate": float((~ok).mean()),
                             "p10_value": float(p10) if p10 is not None else None,
                             "p50_value": float(p50) if p50 is not None else None,
                             "p90_value": float(p90) if p90 is not None else None,
                             "slope_pp_per_unit": slope, "low_decile_pp": low, "high_decile_pp": high,
                             "change_p10_p90_pp": (high - low) if (high is not None and low is not None) else None,
                             "mean_abs_effect_pp": float(np.abs(pp).mean())})

    pl.DataFrame(long_rows).write_csv(os.path.join(tables, "feature_curves_all83.csv"))
    summ = pl.DataFrame(summary_rows).sort("mean_abs_effect_pp", descending=True)
    summ.write_csv(os.path.join(tables, "feature_slopes_all83.csv"))
    print(summ.select(["feature", "unit", "mean_abs_effect_pp", "slope_pp_per_unit", "change_p10_p90_pp", "nan_rate"])
          .head(40).to_pandas().to_string(index=False, float_format=lambda v: f"{v:.5g}"))


if __name__ == "__main__":
    main()
