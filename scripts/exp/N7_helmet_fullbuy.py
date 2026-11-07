"""Do CTs forgo helmets on full buys? (note: "on full buys CTs forgo helmets because Ts carry AKs that kill through them")

Population: data/status_1hz (2026, used rounds), first live second (second == 0), alive players only.
Full buy side-round = the side's equipment_value_calc summed over its players >= 20000 (same cut-off as N2; 17000 and 23000 shown).
Reported: share of players with armor > 0 and with a helmet, by side, for full-buy side-rounds, and for all non-pistol side-rounds.
Match-cluster bootstrap, 500 resamples, seed 0.
"""
import glob
import json
import os
import sys

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
fs = sorted(glob.glob(os.path.join(ROOT, "data", "status_1hz", "*.parquet")))
S = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "second", "side", "armor", "has_helmet", "equipment_value_calc", "inv_primary"]).filter(pl.col("second") == 0) for f in fs])
rn = pl.read_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet")).select("match_key", "round", "max_rounds_final")
S = S.join(rn, on=["match_key", "round"], how="left")
S = S.with_columns(((pl.col("round") == 1) | (pl.col("round") == pl.col("max_rounds_final") // 2 + 1)).alias("pistol"))
team = S.group_by(["match_key", "round", "side"]).agg(pl.col("equipment_value_calc").sum().alias("team_eq"), pl.len().alias("n_players"))
S = S.join(team, on=["match_key", "round", "side"], how="left").filter(pl.col("n_players") == 5)
print("player rows at second 0:", S.height, " matches:", S["match_key"].n_unique(), " side-rounds:", S.select("match_key", "round", "side").unique().height)

mk = S["match_key"].to_numpy()
uniq, mid = np.unique(mk, return_inverse=True)
M = len(uniq)
idx = np.random.default_rng(0).integers(0, M, size=(500, M))


def share(mask, y):
    num = np.bincount(mid, weights=(mask & y).astype(float), minlength=M)
    den = np.bincount(mid, weights=mask.astype(float), minlength=M)
    reps = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(num.sum() / max(den.sum(), 1)), [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], int(den.sum())


side = S["side"].to_numpy()
armor = (S["armor"].to_numpy() > 0)
helm = S["has_helmet"].to_numpy().astype(bool)
eq = S["team_eq"].to_numpy()
pist = S["pistol"].to_numpy().astype(bool)
prim = S["inv_primary"].to_numpy()
out = {"rows": []}
print(f"\n{'population':<34}{'side':<5}{'players':>9}{'armor':>10}{'helmet':>10}   helmet 95%")
for lab, mask in [("full buy (team >= 20000)", eq >= 20000), ("full buy (>= 17000)", eq >= 17000), ("full buy (>= 23000)", eq >= 23000), ("all non-pistol side-rounds", ~pist)]:
    for sd, nm in ((3, "CT"), (2, "T")):
        m = mask & (side == sd)
        a, aci, n = share(m, armor)
        h, hci, _ = share(m, helm)
        out["rows"].append({"population": lab, "side": nm, "n": n, "armor": a, "armor_ci95": aci, "helmet": h, "helmet_ci95": hci})
        print(f"{lab:<34}{nm:<5}{n:>9}{100 * a:>9.1f}%{100 * h:>9.1f}%   {100 * hci[0]:.1f}-{100 * hci[1]:.1f}")
# by primary weapon among full-buy sides: rifle holders only
RIFLES_CT, RIFLES_T = [16, 60], [7]
for lab, sd, codes in (("CT with M4 (full buy)", 3, RIFLES_CT), ("T with AK-47 (full buy)", 2, RIFLES_T)):
    m = (eq >= 20000) & (side == sd) & np.isin(prim, codes)
    h, hci, n = share(m, helm)
    out["rows"].append({"population": lab, "n": n, "helmet": h, "helmet_ci95": hci})
    print(f"{lab:<34}{'':<5}{n:>9}{'':>10}{100 * h:>9.1f}%   {100 * hci[0]:.1f}-{100 * hci[1]:.1f}")
json.dump(out, open(os.path.join(ROOT, "data", "viz", "N7_helmet_fullbuy.json"), "w", encoding="utf-8"), indent=1)
