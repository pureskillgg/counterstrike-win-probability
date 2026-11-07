"""Y1: (1) what the snapshot model relies on (mean |TreeSHAP| per feature and per feature family, match-cluster bootstrap) and
(2) the T-side AK-47 versus CT-side M4 question at round start (second 0), from the 1 Hz player snapshots (2026, used rounds).

  python -I -X utf8 scripts\\exp\\Y1_importance_weapons.py        (CPU env; reads only G:)  ->  data/viz/Y1_importance_weapons.json
Round-start loadouts: the snapshot at second 0 of each round (post buy phase). "Price" is the data's own equipment_value_calc of the
player (everything carried), not the weapon's list price. Rounds with an inferred roster side (missing_side) are left out.
"""
import glob
import json
import os
import re
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, os.path.join(r"G:\pureskill\csds-cs2", "scripts", "model"))
import numpy as np
import polars as pl
import xgboost as xgb

from wp_items import AK, AWP, M4, RIFLE, SMG

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
rng = np.random.default_rng(0)
out = {}

# ------------------------------------------------------------------ 1. importance
meta = json.load(open(os.path.join(ROOT, "models", "winprob_v1.meta.json"), encoding="utf-8"))
feats, best = meta["features"], meta["best_iteration"]
b = xgb.Booster()
b.load_model(os.path.join(ROOT, "models", "winprob_v1.json"))
S = pl.read_parquet(os.path.join(ROOT, "data", "combined", "export_winprob_v1_test_sample.parquet"))
X = S.select(feats).to_numpy().astype(np.float32)
contrib = b.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=(0, best + 1))[:, :-1]
A = np.abs(contrib)
WEAPONS = {"ak", "m4", "awp", "rifle", "sniper_other", "smg", "heavy", "no_primary"}
GREN = {"flash", "smoke", "he", "fire", "decoy"}


def family(f):
    if f in ("alive_ct", "alive_t", "ct_alive_minus_t"):
        return "alive counts"
    if f in ("bomb_planted", "seconds_since_plant"):
        return "bomb state"
    if f == "seconds_since_round_live":
        return "clock"
    if f.startswith("map_is_"):
        return "map"
    m = re.match(r"^st_(.*)_(ct|t|diff)$", f)
    base = m.group(1)
    return ("health" if base == "hp" else "armor" if base.startswith("armor") else "helmet" if base == "helmet_n" else
            "defuse kit" if base in ("kit_n", "defusing") else "primary weapons" if base in WEAPONS else "grenades" if base in GREN else
            "equipment value" if base == "equip" else base)


fam = np.array([family(f) for f in feats])
mk = S["match_key"].to_numpy()
um, inv = np.unique(mk, return_inverse=True)
per_match = np.zeros((len(um), len(feats)))
np.add.at(per_match, inv, A)
cnt = np.bincount(inv).astype(float)
tot = per_match.sum(0) / cnt.sum()
famnames = sorted(set(fam))
fam_share_pt = {n: float(tot[fam == n].sum() / tot.sum()) for n in famnames}
boots = {n: [] for n in famnames}
for _ in range(300):
    pick = rng.integers(0, len(um), len(um))
    t = per_match[pick].sum(0) / cnt[pick].sum()
    for n in famnames:
        boots[n].append(t[fam == n].sum() / t.sum())
out["importance"] = {
    "n_rows": int(len(S)), "n_matches": int(len(um)),
    "families": sorted([{"family": n, "share": fam_share_pt[n], "ci95": [float(np.percentile(boots[n], 2.5)), float(np.percentile(boots[n], 97.5))]} for n in famnames],
                       key=lambda r: -r["share"]),
    "top_features": [{"feature": feats[i], "mean_abs_logodds": float(tot[i]), "share": float(tot[i] / tot.sum())} for i in np.argsort(-tot)[:15]],
    "weapon_features": [{"feature": feats[i], "mean_abs_logodds": float(tot[i]), "share": float(tot[i] / tot.sum())} for i in np.argsort(-tot) if re.match(r"^st_(ak|m4|awp|rifle|smg|sniper_other|heavy|no_primary)_", feats[i])],
}
print("== feature families (share of mean |SHAP|, 95% match-cluster interval)")
for r in out["importance"]["families"]:
    print(f"  {r['family']:<18} {100 * r['share']:5.1f}%  [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}]")
print("== top features")
for r in out["importance"]["top_features"][:10]:
    print(f"  {r['feature']:<22} {100 * r['share']:5.1f}%")

# ------------------------------------------------------------------ 2. weapons at round start
R = pl.read_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet"), columns=["match_key", "round", "missing_side", "use", "final"]).filter(pl.col("use") & pl.col("missing_side").is_null())
lab = R.select("match_key", "round", (pl.col("final") == 3).cast(pl.Int8).alias("ct_win"))
frames = []
for f in sorted(glob.glob(os.path.join(ROOT, "data", "status_1hz", "*.parquet"))):
    d = pl.scan_parquet(f).filter(pl.col("second") == 0).select("match_key", "round", "side", "inv_primary", "armor", "has_helmet", "equipment_value_calc").collect()
    frames.append(d)
P = pl.concat(frames).join(lab, on=["match_key", "round"], how="inner")
P = P.with_columns(
    pl.when(pl.col("inv_primary").is_in(list(AK))).then(pl.lit("AK-47"))
    .when(pl.col("inv_primary").is_in(list(M4))).then(pl.lit("M4 (A4/A1-S)"))
    .when(pl.col("inv_primary").is_in(list(AWP))).then(pl.lit("AWP"))
    .when(pl.col("inv_primary").is_in(list(RIFLE))).then(pl.lit("other rifle"))
    .when(pl.col("inv_primary").is_in(list(SMG))).then(pl.lit("SMG"))
    .when(pl.col("inv_primary") == 0).then(pl.lit("no primary")).otherwise(pl.lit("other")).alias("wclass"))
print(f"\n== round-start snapshots: {P['match_key'].n_unique()} matches, {P.select('match_key', 'round').unique().height} rounds, {P.height} player rows")
price = P.group_by(["side", "wclass"]).agg(pl.len().alias("n"), pl.col("equipment_value_calc").median().alias("median_equip"), pl.col("equipment_value_calc").mean().alias("mean_equip"),
                                           (pl.col("armor") > 0).mean().alias("armor"), pl.col("has_helmet").mean().alias("helmet")).sort(["side", "n"], descending=[False, True])
print("== per player at second 0 by primary weapon (equipment value = everything carried)")
for r in price.iter_rows(named=True):
    print(f"  {'CT' if r['side'] == 3 else 'T ':<3} {r['wclass']:<14} n {r['n']:>7}  median ${r['median_equip']:>6.0f}  mean ${r['mean_equip']:>6.0f}  armor {100 * r['armor']:.0f}%  helmet {100 * r['helmet']:.0f}%")
out["per_player_by_weapon"] = price.to_dicts()
ak_share = P.filter(pl.col("wclass").is_in(["AK-47", "M4 (A4/A1-S)", "other rifle"])).group_by("side", "wclass").len().to_dicts()
out["rifle_choice_counts"] = ak_share

# team level
team = P.group_by(["match_key", "round", "side"]).agg(
    (pl.col("wclass") == "AK-47").sum().alias("ak"), (pl.col("wclass") == "M4 (A4/A1-S)").sum().alias("m4"), (pl.col("wclass") == "AWP").sum().alias("awp"),
    pl.col("equipment_value_calc").sum().alias("equip"), pl.col("ct_win").first().alias("ct_win"))
ct = team.filter(pl.col("side") == 3).select("match_key", "round", pl.col("ak").alias("ct_ak"), pl.col("m4").alias("ct_m4"), pl.col("awp").alias("ct_awp"), pl.col("equip").alias("ct_equip"), "ct_win")
t = team.filter(pl.col("side") == 2).select("match_key", "round", pl.col("ak").alias("t_ak"), pl.col("m4").alias("t_m4"), pl.col("awp").alias("t_awp"), pl.col("equip").alias("t_equip"))
T = ct.join(t, on=["match_key", "round"], how="inner").with_columns((pl.col("ct_equip") - pl.col("t_equip")).alias("equip_gap"))


def cluster_ci(df, col="ct_win"):
    g = df.group_by("match_key").agg(pl.col(col).sum().alias("w"), pl.len().alias("n"))
    w, n = g["w"].to_numpy().astype(float), g["n"].to_numpy().astype(float)
    r = w.sum() / n.sum()
    bs = []
    for _ in range(400):
        p = rng.integers(0, len(w), len(w))
        bs.append(w[p].sum() / n[p].sum())
    return float(r), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], int(n.sum()), int(len(w))


groups = {
    "all rounds": T,
    "both teams full-ish (each >= 20,000)": T.filter((pl.col("ct_equip") >= 20000) & (pl.col("t_equip") >= 20000)),
    "T: 4+ AK-47  vs  CT: 4+ M4": T.filter((pl.col("t_ak") >= 4) & (pl.col("ct_m4") >= 4)),
    "T: 4+ AK-47  vs  CT: 4+ AK-47": T.filter((pl.col("t_ak") >= 4) & (pl.col("ct_ak") >= 4)),
    "T: 4+ AK-47  vs  CT: 0-1 rifles of any kind (eco/force)": T.filter((pl.col("t_ak") >= 4) & (pl.col("ct_m4") + pl.col("ct_ak") <= 1)),
    "CT: 4+ M4  vs  T: 0-1 AK-47 (T eco/force)": T.filter((pl.col("ct_m4") >= 4) & (pl.col("t_ak") <= 1)),
}
print("\n== CT win rate by round-start weapon matchup (match-cluster 95% interval)")
rows = []
for name, d in groups.items():
    if d.height < 200:
        continue
    r, ci, n, m = cluster_ci(d)
    gap = float(d["equip_gap"].median())
    rows.append({"group": name, "ct_win_rate": r, "ci95": ci, "rounds": n, "matches": m, "median_ct_minus_t_equipment": gap})
    print(f"  {name:<58} CT wins {100 * r:5.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]  rounds {n:>6}  median CT-T equipment {gap:+,.0f}")
out["matchups"] = rows
# equipment gap effect overall
T = T.with_columns(pl.col("equip_gap").cut([-12000, -6000, -2000, 2000, 6000, 12000]).alias("gap_bin"))
print("\n== CT win rate by CT-minus-T equipment value")
gb = []
for k in sorted(T["gap_bin"].unique().to_list(), key=lambda s: str(s)):
    d = T.filter(pl.col("gap_bin") == k)
    r, ci, n, m = cluster_ci(d)
    gb.append({"gap_bin": str(k), "ct_win_rate": r, "ci95": ci, "rounds": n})
    print(f"  {str(k):<22} CT wins {100 * r:5.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]  rounds {n}")
out["equipment_gap"] = gb
json.dump(out, open(os.path.join(ROOT, "data", "viz", "Y1_importance_weapons.json"), "w", encoding="utf-8"), indent=1, default=str)
print("\n-> data/viz/Y1_importance_weapons.json")
