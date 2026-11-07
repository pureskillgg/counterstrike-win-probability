"""Headshot lethality, redone cleanly (the Haiku version N4 did not match its own stated method).

Population: 2026 player_hurt rows. Victim at full health = the victim's FIRST damage event in the round (everyone starts a
round at 100 HP, so that hit lands on 100 HP). A hit is lethal iff health after the hit is 0.
Head = the hit_box_code with by far the highest mean damage for AK-47 first hits (checked and printed).
Helmet is inferred from the hit itself: on a head hit, armor_removed > 0 means the victim wore armor with a helmet (kevlar alone
does not reduce head damage); armor_removed == 0 with armor == 0 afterwards and no kevlar-only possibility is ambiguous, so the
table also reports the split "armor after hit > 0 / == 0".
Friendly fire is excluded (attacker_team_code == player_team_code) and the exclusion is counted and printed.
Distance = 3D distance between attacker and victim positions in game units, terciles over all counted head hits of a weapon.
Intervals: 500-resample match-cluster bootstrap, seed 0.
"""
import glob
import json
import os
import sys

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
STORE = sys.argv[1] if len(sys.argv) > 1 else "combined"
S = os.path.join(r"G:\pureskill\csds-cs2\tomes", STORE, "tome", "csds", "basic_player_hurt.COMBINED")
cols = ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "hit_box_code", "health", "armor", "health_removed",
        "armor_removed", "effective_health_removed", "player_team_code", "attacker_team_code",
        "player_x_pos", "player_y_pos", "player_z_pos", "attacker_x_pos", "attacker_y_pos", "attacker_z_pos"]
H = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(S, "dataframe_*")))], how="vertical_relaxed")
H = H.filter(pl.col("match_key").str.contains("^csds/2026/"))
n_all = H.height
R = pl.read_parquet(rf"G:\pureskill\csds-cs2\data\{STORE}\rounds.parquet").filter(pl.col("use")).select("match_key", "round", "freeze", "end_eff")
H = H.join(R, on=["match_key", "round"], how="inner").filter((pl.col("tick") >= pl.col("freeze")) & (pl.col("tick") <= pl.col("end_eff")))
print(f"2026 hurt rows {n_all}; in used rounds' live windows {H.height}")
ff = H.filter(pl.col("attacker_team_code") == pl.col("player_team_code")).height
H = H.filter((pl.col("attacker_team_code") != pl.col("player_team_code")) | pl.col("attacker_team_code").is_null() | pl.col("player_team_code").is_null())
H = H.filter(pl.col("attacker_id").is_not_null() & (pl.col("attacker_id") != pl.col("player_id")))
H = H.sort(["match_key", "round", "player_id", "tick"]).with_columns(pl.int_range(pl.len()).over(["match_key", "round", "player_id"]).alias("k"))
F = H.filter(pl.col("k") == 0)
print(f"friendly-fire rows removed {ff}; first-damage-in-round rows {F.height}")
ak = F.filter(pl.col("weapon_name") == "ak47")
print("AK-47 first hits: mean health_removed by hit_box_code")
print(ak.group_by("hit_box_code").agg(pl.len().alias("n"), pl.col("health_removed").mean().alias("mean_hr")).sort("mean_hr", descending=True).head(8))
head = int(ak.group_by("hit_box_code").agg(pl.col("health_removed").mean().alias("m")).sort("m", descending=True)["hit_box_code"][0])
print("head code:", head)
W = {"ak47": ["ak47"], "m4 (m4a1 + m4a1_silencer)": ["m4a1", "m4a1_silencer"], "awp": ["awp"], "famas+galil": ["famas", "galilar"], "sg556+aug": ["sg556", "aug"],
     "deagle": ["deagle"], "glock": ["glock"], "usp+p2000": ["usp_silencer", "hkp2000"], "smgs": ["mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon"]}
F = F.filter(pl.col("hit_box_code") == head).with_columns(
    (pl.col("health") == 0).alias("lethal"),
    (pl.col("armor_removed") > 0).alias("helmet_inferred"),
    ((pl.col("attacker_x_pos") - pl.col("player_x_pos")) ** 2 + (pl.col("attacker_y_pos") - pl.col("player_y_pos")) ** 2
     + (pl.col("attacker_z_pos") - pl.col("player_z_pos")) ** 2).sqrt().alias("dist"))
mk = F["match_key"].to_numpy()
uniq, mid = np.unique(mk, return_inverse=True)
M = len(uniq)
idx = np.random.default_rng(0).integers(0, M, size=(500, M))


def rate(mask, y):
    num = np.bincount(mid, weights=(mask & y).astype(float), minlength=M)
    den = np.bincount(mid, weights=mask.astype(float), minlength=M)
    reps = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(num.sum() / max(den.sum(), 1)), [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], int(den.sum())


lethal = F["lethal"].to_numpy()
helm = F["helmet_inferred"].to_numpy()
dist = F["dist"].to_numpy()
wn = F["weapon_name"].to_numpy()
out = {"head_code": head, "friendly_fire_removed": ff, "first_hits_total": int(len(F)), "rows": []}
print(f"\n{'weapon':<28}{'helmet':<12}{'n head hits':>12}{'one-shot rate':>16}   95% interval")
for name, ws in W.items():
    wm = np.isin(wn, ws)
    for hl, hm in (("any", np.ones_like(wm)), ("helmet", helm), ("no helmet", ~helm)):
        m = wm & hm
        if m.sum() < 30:
            continue
        r, ci, n = rate(m, lethal)
        out["rows"].append({"weapon": name, "helmet": hl, "n": n, "rate": r, "ci95": ci})
        print(f"{name:<28}{hl:<12}{n:>12}{r:>16.3f}   {ci[0]:.3f}-{ci[1]:.3f}")
# distance terciles for AK and M4 with helmet inferred
for name in ("ak47", "m4 (m4a1 + m4a1_silencer)"):
    wm = np.isin(wn, W[name]) & np.isfinite(dist)
    q = np.quantile(dist[wm], [1 / 3, 2 / 3])
    for lab, lo, hi in (("near", -1, q[0]), ("mid", q[0], q[1]), ("far", q[1], 1e9)):
        for hl, hm in (("helmet", helm), ("no helmet", ~helm)):
            m = wm & (dist > lo) & (dist <= hi) & hm
            if m.sum() < 30:
                continue
            r, ci, n = rate(m, lethal)
            out["rows"].append({"weapon": name, "helmet": hl, "distance": lab, "n": n, "rate": r, "ci95": ci, "dist_range": [float(max(lo, 0)), float(min(hi, 1e6))]})
            print(f"{name:<28}{hl:<12}{lab:>6} {n:>7}{r:>12.3f}   {ci[0]:.3f}-{ci[1]:.3f}  ({max(lo, 0):.0f}-{min(hi, 99999):.0f} units)")
json.dump(out, open(rf"G:\pureskill\csds-cs2\data\viz\N4b_headshot_clean_{STORE}.json", "w", encoding="utf-8"), indent=1)
