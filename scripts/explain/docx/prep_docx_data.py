import json
import os

import polars as pl
from PIL import Image

REPO = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability"
T = f"{REPO}/docs/explainability/tables"
F = f"{REPO}/docs/explainability/figures"
X57 = f"{REPO}/docs/explainability/xgb57/tables"
OUT = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/816f32ee-27ce-47da-985a-83230418259e/scratchpad/report_data.json"

LOADOUT_PANELS = ["st_hp_diff", "st_equip_diff", "st_armor_n_diff", "st_rifle_diff", "st_awp_diff", "st_smg_diff",
                  "st_hp_ct", "st_hp_t", "st_equip_ct", "st_equip_t", "st_armor_n_ct", "st_armor_n_t"]
DISAGREE = ["defuse_slack", "seconds_since_plant", "st_hp_diff", "alive_ct", "alive_t", "ct_alive_minus_t"]
WHY = {
    "defuse_slack": "Missing for 83% of seconds (bomb not planted). Shuffling puts values into rows the model never saw.",
    "seconds_since_plant": "Uses -1 when no bomb is planted. Shuffling creates impossible combinations.",
    "st_hp_diff": "Exact difference of two inputs. Shuffling one breaks the link to the other.",
    "alive_ct": "Little direct credit: alive information is probably carried by the loadout inputs.",
    "alive_t": "Little direct credit, as for alive_ct.",
    "ct_alive_minus_t": "Partly a function of the two alive counts, so credit is shared.",
}
FLAGS = [
    ("st_armor_sum_ct / st_armor_sum_t", "Dropped by request (notes).", "CT 0.44 pp, T 0.54 pp", "CT 0.0041, T 0.0030",
     "not yet measured"),
    ("VIEW: c/t sees_enemy, c/t exposed", "Unreliable (notes): remove with sees-enemy.", "0.63 pp", "0.00099",
     "VIEW alone -0.00067 vs base"),
    ("BLIND: c/t blind_n, blind_sum", "Source unclear (notes): rebuild from player blind events.", "0.03 pp", "0.000025",
     "BLIND alone -0.00002 vs base"),
    ("defuse_slack", "Crude (notes): rebuild as time to reach and defuse.", "1.28 pp", "0.072 (see disagreement)",
     "GEO alone -0.00313 vs base"),
    ("st_heavy_ct / st_heavy_t", "Dropped in v6 (drop heavy).", "0.07 and 0.05 pp", "about 0", "n/a"),
]
ABLATION = [
    ("base (59 inputs)", "0", "reference"),
    ("base + GEO", "-0.00313", "GEO alone"),
    ("base + VIEW", "-0.00067", "VIEW alone"),
    ("base + BLIND", "-0.00002", "BLIND alone"),
    ("base + GEO + VIEW", "-0.00348", ""),
    ("base + GEO + BLIND", "-0.00312", ""),
    ("base + VIEW + BLIND", "-0.00066", ""),
    ("full (83 inputs)", "-0.00344", "all position groups"),
]


def fmt(v, nd=2):
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return f"{v:.{nd}f}"


def img(path):
    w, h = Image.open(path).size
    return {"path": path.replace("\\", "/"), "w": w, "h": h}


def main():
    b = pl.read_csv(f"{T}/bundle_importance_and_effects.csv")
    fe = pl.read_csv(f"{T}/feature_importance_and_effects.csv")
    sl = pl.read_csv(f"{T}/feature_slopes_all83.csv")
    fe_map = {r["feature"]: r for r in fe.iter_rows(named=True)}
    sl_map = {r["feature"]: r for r in sl.iter_rows(named=True)}
    fe_rank = {r["feature"]: i + 1 for i, r in enumerate(fe.sort("mean_abs_effect_pp", descending=True).iter_rows(named=True))}

    bundle_rows = []
    for r in b.sort("mean_abs_effect_pp", descending=True).iter_rows(named=True):
        bundle_rows.append([r["bundle"], str(r["n_inputs"]), fmt(r["mean_abs_effect_pp"]),
                            fmt(r["mean_signed_effect_pp"]), fmt(r["perm_delta_logloss"], 5)])

    loadout_rows = []
    for f in LOADOUT_PANELS:
        s = sl_map[f]
        loadout_rows.append([f, s["unit"], fmt(s["mean_abs_effect_pp"]),
                             f"{s['slope_pp_per_unit']:+.4g}" if s["slope_pp_per_unit"] is not None else "",
                             fmt(s["change_p10_p90_pp"]), fmt(fe_map[f]["perm_delta_logloss"], 4) if f in fe_map else ""])

    top_rows = []
    for f in sl.sort("mean_abs_effect_pp", descending=True).head(24)["feature"].to_list():
        s = sl_map[f]
        top_rows.append([f, s["bundle"], s["unit"], fmt(s["mean_abs_effect_pp"]),
                         f"{s['slope_pp_per_unit']:+.4g}" if s["slope_pp_per_unit"] is not None else "",
                         fmt(s["change_p10_p90_pp"]), fmt(fe_map[f]["perm_delta_logloss"], 4) if f in fe_map else "",
                         str(fe_rank.get(f, ""))])

    disagree_rows = []
    for f in DISAGREE:
        disagree_rows.append([f, str(fe_rank.get(f, "")), fmt(fe_map[f]["mean_abs_effect_pp"]),
                              fmt(fe_map[f]["perm_delta_logloss"], 4), WHY[f]])

    data = {
        "bundle_rows": bundle_rows, "loadout_rows": loadout_rows, "top_rows": top_rows,
        "disagree_rows": disagree_rows, "flags": [list(x) for x in FLAGS], "ablation": [list(x) for x in ABLATION],
        "figs": {
            "bundles": img(f"{F}/bundle_effects.png"),
            "curves_loadout": img(f"{F}/curves_loadout.png"),
            "curves_top_a": img(f"{F}/curves_top_a.png"),
            "curves_top_b": img(f"{F}/curves_top_b.png"),
            "shap_vs_perm": img(f"{F}/shap_vs_permutation.png"),
        },
    }

    # 57-input model, if its explanation has finished
    cmp_path = f"{X57}/compare_83_vs_57_bundles.csv"
    meta57 = f"{X57}/explain_meta_57.json"
    if os.path.exists(cmp_path) and os.path.exists(meta57):
        c = pl.read_csv(cmp_path)
        data["cmp57_rows"] = [[r["bundle"], fmt(r["effect_pp_83"]), fmt(r["effect_pp_57"]),
                               fmt(r["perm_83"], 4), fmt(r["perm_57"], 4)] for r in c.iter_rows(named=True)]
        data["meta57"] = json.load(open(meta57, encoding="utf-8"))
        f57 = pl.read_csv(f"{X57}/feature_effects_57.csv").sort("mean_abs_effect_pp", descending=True).head(20)
        data["top57_rows"] = [[r["feature"], r["bundle"], fmt(r["mean_abs_effect_pp"]), fmt(r["mean_signed_effect_pp"]),
                               fmt(r["perm_delta_logloss"], 4)] for r in f57.iter_rows(named=True)]
    json.dump(data, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT, "| 57 section:", "cmp57_rows" in data)


if __name__ == "__main__":
    main()
