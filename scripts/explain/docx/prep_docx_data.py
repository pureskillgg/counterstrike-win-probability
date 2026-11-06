import json
import math
import os

import polars as pl
from PIL import Image

REPO = "C:/Users/13144/Documents/pureskill/counterstrike-win-probability"
T = f"{REPO}/docs/explainability/tables"
F = f"{REPO}/docs/explainability/figures"
L = f"{REPO}/docs/explainability/local"
X57 = f"{REPO}/docs/explainability/xgb57/tables"
OUT = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/816f32ee-27ce-47da-985a-83230418259e/scratchpad/report_data.json"

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
# Last column: the GDT drop test from drop_ablation_83_gdt.csv (mean of three seeds, paired with the full model).
FLAGS = [
    ("st_armor_sum_ct / st_armor_sum_t", "Dropped by request (notes).", "CT 0.44 pp, T 0.54 pp", "CT 0.0041, T 0.0030",
     "Drop: GDT +0.0001"),
    ("VIEW: c/t sees_enemy, c/t exposed", "Unreliable (notes): remove with sees-enemy.", "0.63 pp", "0.00099",
     "Drop: GDT -0.0005"),
    ("BLIND: c/t blind_n, blind_sum", "Source unclear (notes): rebuild from player blind events.", "0.03 pp", "0.000025",
     "Drop: GDT 0.0000"),
    ("defuse_slack", "Crude (notes): rebuild as time to reach and defuse.", "1.28 pp", "0.072 (see disagreement)",
     "Only in bomb geometry: GDT -0.0005"),
    ("st_heavy_ct / st_heavy_t", "Dropped in v6 (drop heavy).", "0.07 and 0.05 pp", "about 0", "n/a"),
]
# Earlier sprint run (eta 0.05, base of 59 inputs), log loss only. Superseded for decisions by the GDT table.
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
GDT_LABEL = {
    "full": "Full model (83)",
    "no_position": "Non-positional (59)",
    "checklist_77": "Checklist fallback (77)",
    "minus_alive": "Minus alive counts",
    "minus_armor_sums": "Minus armour sums",
    "minus_blind": "Minus blind",
    "minus_round_state": "Minus round state",
    "minus_view": "Minus VIEW",
    "minus_bomb_geo": "Minus bomb geometry (incl. defuse slack)",
    "loadout_only": "Loadout only (46)",
    "round_state_only": "Round state only (6)",
    "minus_site_spread": "Minus site and spread (74)",
    "minus_contact": "Minus contact (80)",
}
STATE_LABEL = {
    "1v5_any_bomb": "Any 1v5 state (either bomb status)",
    "1v5_bomb_planted": "Bomb planted",
    "1v5_no_bomb": "No bomb planted",
}


def fmt(v, nd=2):
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return f"{v:.{nd}f}"


def signed(v, nd=4):
    # Signed value; a value that rounds to zero prints as unsigned zero (no "-0.0000").
    if round(v, nd) == 0:
        return f"{0:.{nd}f}"
    return f"{v:+.{nd}f}"


def img(path):
    w, h = Image.open(path).size
    return {"path": path.replace("\\", "/"), "w": w, "h": h}


def gdt_rows():
    g = {r["set"]: r for r in pl.read_csv(f"{T}/drop_ablation_83_gdt.csv").iter_rows(named=True)}
    a = {r["set"]: r for r in pl.read_csv(f"{T}/drop_ablation_83.csv").iter_rows(named=True)}
    rows = []
    for s, label in GDT_LABEL.items():
        if s not in g:
            continue
        gr, ar = g[s], a[s]
        if s == "full":
            rows.append([label, str(ar["n_inputs"]), f"{gr['mean_GDT']:.4f}", "reference", "", "reference"])
        else:
            rows.append([label, str(ar["n_inputs"]), f"{gr['mean_GDT']:.4f}",
                         signed(gr["mean_paired_GDT_difference_vs_full"]),
                         f"{signed(gr['paired_difference_min'])} to {signed(gr['paired_difference_max'])}",
                         signed(ar["mean_delta_vs_full"], 6)])
    ci = (g["full"]["mean_GDT_ci_low"], g["full"]["mean_GDT_ci_high"])
    return rows, f"{ci[0]:.4f} to {ci[1]:.4f}"


def comparison_rows():
    """Positional versus non-positional, and the 83 against the checklist fallback, on log loss and on GDT."""
    g = {r["set"]: r for r in pl.read_csv(f"{T}/drop_ablation_83_gdt.csv").iter_rows(named=True)}
    a = {r["set"]: r for r in pl.read_csv(f"{T}/drop_ablation_83.csv").iter_rows(named=True)}
    order = [("full", "Full model (83): all inputs"), ("checklist_77", "Checklist fallback (77): no armour sums, no VIEW"),
             ("minus_round_state", "Minus round state (77): no round state"),
             ("no_position", "Non-positional (59): no position inputs"),
             ("loadout_only", "Loadout only (46)")]
    rows = []
    for s, label in order:
        if s not in a or s not in g:
            continue
        ll = a[s]["mean_val_logloss"]
        dll = 0.0 if s == "full" else a[s]["mean_delta_vs_full"]
        if s == "full":
            gdt_txt, diff_txt = f"{g[s]['mean_GDT']:.4f}", "reference"
        else:
            gdt_txt = f"{g[s]['mean_GDT']:.4f}"
            diff_txt = f"{signed(g[s]['mean_paired_GDT_difference_vs_full'])} ({signed(g[s]['paired_difference_min'])} to {signed(g[s]['paired_difference_max'])})"
        rows.append([label, str(a[s]["n_inputs"]), f"{ll:.6f}", "reference" if s == "full" else signed(dll, 5), gdt_txt, diff_txt])
    z57 = json.load(open(f"{X57}/explain_meta_57.json", encoding="utf-8"))
    rows.append(["v6 submission model (57): no position, heavy dropped", "57", f"{z57['validation_logloss_all_rows']:.6f}",
                 signed(z57["validation_logloss_all_rows"] - a["full"]["mean_val_logloss"], 5), "not scored", ""])
    return rows


def per_seed_rows():
    p = pl.read_csv(f"{T}/drop_ablation_83_per_seed.csv")
    order = list(GDT_LABEL)
    p = p.with_columns(pl.col("set").replace_strict(order, list(range(len(order))), default=99).alias("o")).sort(["o", "seed"])
    return [[GDT_LABEL.get(r["set"], r["set"]), str(r["seed"]), str(r["n_inputs"]), f"{r['val_logloss']:.6f}",
             signed(r["delta_vs_full_same_seed"], 6), str(r["best_iteration"])] for r in p.iter_rows(named=True)]


def contrast_rows():
    c = pl.read_csv(f"{L}/contrast_confident_ct_vs_t.csv").head(6)
    return [[str(r["rank"]), r["feature"], r["bundle"], f"{r['value_a']:,.1f}", f"{r['value_b']:,.1f}",
             signed(r["logit_a_minus_b"], 2)] for r in c.iter_rows(named=True)]


def match_block():
    """Per-match explanations: the largest moves in P(CT wins) and each bundle's share of the absolute contribution."""
    s = pl.read_csv(f"{T}/match_swings.csv")
    swings = []
    for r in s.filter(pl.col("match") == "largest_loss").head(5).iter_rows(named=True):
        swings.append([str(r["round"]), f"{r['second_from']} to {r['second_to']}", f"{r['p_from']:.2f} to {r['p_to']:.2f}",
                       signed(r["change_pp"], 1) + " pp", f"{r['biggest_mover']} ({signed(r['mover_logit'], 2)})",
                       f"{r['second_mover']} ({signed(r['second_logit'], 2)})", "CT won" if r["label_ct_win"] else "T won"])
    b = pl.read_csv(f"{T}/match_bundle_summary.csv")
    matches = ["largest_loss", "positional_help", "worst_vs_table"]
    order = ["loadout difference", "CT loadout", "T loadout", "round state", "map", "bomb geometry", "site and spread",
             "contact", "view", "blindness"]
    share = {m: {r["bundle"]: r["share_of_abs"] for r in b.filter(pl.col("match") == m).iter_rows(named=True)} for m in matches}
    loadout = ["loadout difference", "CT loadout", "T loadout"]
    position = ["bomb geometry", "site and spread", "contact", "view", "blindness"]
    rows = [[o] + [f"{100 * share[m][o]:.1f}%" for m in matches] for o in order]
    return {"swing_rows": swings, "share_rows": rows,
            "loadout_total": {m: sum(share[m][o] for o in loadout) for m in matches},
            "position_total": {m: sum(share[m][o] for o in position) for m in matches}}


def interaction_rows():
    i = pl.read_csv(f"{L}/interactions_top_pairs.csv").head(8)
    return [[str(r["rank"]), r["feature_a"], r["feature_b"], f"{r['mean_abs_interaction_logit']:.3f}",
             f"{r['mean_abs_shap_a_logit']:.2f}", f"{r['mean_abs_shap_b_logit']:.2f}"] for r in i.iter_rows(named=True)]


def key_numbers():
    """The few numbers the prose quotes, read from the tables so the text cannot drift from them."""
    g = {r["set"]: r for r in pl.read_csv(f"{T}/drop_ablation_83_gdt.csv").iter_rows(named=True)}
    a = {r["set"]: r for r in pl.read_csv(f"{T}/drop_ablation_83.csv").iter_rows(named=True)}
    k = {}
    for s in ["full", "no_position", "checklist_77", "minus_round_state", "loadout_only", "minus_view", "minus_bomb_geo",
              "minus_alive", "minus_armor_sums", "minus_blind", "round_state_only", "minus_site_spread", "minus_contact"]:
        if s not in g or s not in a:
            continue
        k[s] = {"n": int(a[s]["n_inputs"]), "logloss": float(a[s]["mean_val_logloss"]),
                "dlogloss": float(a[s]["mean_delta_vs_full"]), "gdt": float(g[s]["mean_GDT"]),
                "dgdt": float(g[s]["mean_paired_GDT_difference_vs_full"]),
                "dgdt_min": float(g[s]["paired_difference_min"]), "dgdt_max": float(g[s]["paired_difference_max"])}
    z57 = json.load(open(f"{X57}/explain_meta_57.json", encoding="utf-8"))
    k["v6_57"] = {"n": 57, "logloss": float(z57["validation_logloss_all_rows"])}
    return k


def gdt_summary():
    p = f"{T}/gdt_examples_summary.json"
    if not os.path.exists(p):
        return None
    s = json.load(open(p, encoding="utf-8"))
    out = {"matches": s["matches"], "datapoints": s["datapoints_seed0"],
           "share_beats_table": s["share_matches_full_beats_table"], "median_gain": s["median_match_gain_full"]}
    if "share_matches_full_beats_no_position" in s:
        out["share_beats_no_position"] = s["share_matches_full_beats_no_position"]
        out["median_diff_position"] = s["median_match_diff_position"]
    return out


def examples_rows():
    p = f"{T}/gdt_examples.csv"
    if not os.path.exists(p):
        return []
    rows = []
    for r in pl.read_csv(p).iter_rows(named=True):
        match_id = r["match"].split("/")[-1]
        rows.append([r["example"], f"{match_id[:8]}…", r["date"], f"{r['datapoints']:,}", signed(r["gain_full"], 4),
                     "" if r.get("gain_no_position") is None else signed(r["gain_no_position"], 4),
                     "" if r.get("diff_position") is None else signed(r["diff_position"], 4)])
    return rows


def calibration_rows():
    d = pl.read_csv(f"{T}/calibration_deciles_shipped.csv")
    rows = []
    for r in d.iter_rows(named=True):
        gap = r["observed_ct_win_rate"] - r["mean_predicted_p_ct_win"]
        rows.append([str(r["decile"]), f"{r['n']:,}", f"{r['mean_predicted_p_ct_win']:.4f}",
                     f"{r['observed_ct_win_rate']:.4f}", f"{r['observed_ci_low']:.4f} to {r['observed_ci_high']:.4f}",
                     signed(gap)])
    return rows


def onev5_rows():
    m = {r["state"]: r for r in pl.read_csv(f"{T}/calibration_1v5_shipped.csv").iter_rows(named=True)}
    return [[STATE_LABEL[s], f"{m[s]['rounds']:,}", f"{m[s]['matches']:,}",
             f"{m[s]['expected_ct_win_rounds']:.2f}", str(m[s]["observed_ct_win_rounds"])]
            for s in STATE_LABEL]


def local_rows():
    """The four example seconds and the two contrasts, from docs/explainability/local/."""
    ex = pl.read_csv(f"{L}/examples.csv")
    cols = ex.columns
    rows = [[str(v) if v is not None else "" for v in r] for r in ex.iter_rows()]
    return cols, rows


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

    gdt, gdt_ci = gdt_rows()
    cols, local = local_rows()

    data = {
        "bundle_rows": bundle_rows, "loadout_rows": loadout_rows, "top_rows": top_rows,
        "disagree_rows": disagree_rows, "flags": [list(x) for x in FLAGS], "ablation": [list(x) for x in ABLATION],
        "gdt_rows": gdt, "gdt_ci": gdt_ci, "comparison_rows": comparison_rows(), "gdt_summary": gdt_summary(),
        "key": key_numbers(), "per_seed_rows": per_seed_rows(), "contrast_rows": contrast_rows(),
        "match": match_block(), "interaction_rows": interaction_rows(),
        "examples_rows": examples_rows(), "calib_rows": calibration_rows(), "onev5_rows": onev5_rows(),
        "local_cols": cols, "local_rows": local,
        "figs": {
            "bundles": img(f"{F}/bundle_effects.png"),
            "top25": img(f"{F}/top25_inputs.png"),
            "curves_loadout": img(f"{F}/curves_loadout.png"),
            "curves_top_a": img(f"{F}/curves_top_a.png"),
            "curves_top_b": img(f"{F}/curves_top_b.png"),
            "shap_vs_perm": img(f"{F}/shap_vs_permutation.png"),
            "gdt_ablation": img(f"{F}/gdt_ablation.png"),
            "calibration": img(f"{F}/calibration_deciles.png"),
            "waterfall": img(f"{F}/waterfall_largest_loss.png"),
        },
    }
    for key, name in [("gdt_by_alive", "gdt_by_alive.png"), ("gdt_per_match", "gdt_per_match.png"),
                      ("gdt_example_round", "gdt_example_round.png")]:
        p = f"{F}/{name}"
        if os.path.exists(p):
            data["figs"][key] = img(p)

    for key, name in [("match_largest_loss", "match_largest_loss.png"), ("match_positional_help", "match_positional_help.png"),
                      ("match_worst_vs_table", "match_worst_vs_table.png")]:
        p = f"{F}/{name}"
        if os.path.exists(p):
            data["figs"][key] = img(p)

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
    print("wrote", OUT, "| 57 section:", "cmp57_rows" in data, "| gdt rows:", len(gdt),
          "| examples:", len(data["examples_rows"]), "| comparison rows:", len(data["comparison_rows"]))


if __name__ == "__main__":
    main()
