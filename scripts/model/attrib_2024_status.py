"""Is the lower 2024 equipment and grenade level (vs 2026) the game changing, or the 2024 status fields?

Revised after a first run whose "round start" was after the buy phase (money far above the pistol starting value in both eras).

Check 1 (field consistency, decisive): a player's money plus equipment value is conserved by buying. At the first status second of a
pistol round, every player's (money + equipment_value_calc) should equal the same starting wealth in both eras. A difference between
eras would mean the status fields disagree, not the game.
Check 2 (economy, only if check 1 passes): non-pistol mid-round (20 to 30 s after live start) loadout features from the snapshot caches
(equipment, flashes, smokes, HE, rifles), 2024 against 2026.

Verdict rules (pre-registered before the run):
  FIELD  pistol start wealth differs between eras by more than 5%, or any 2024 status field has a null rate above 1%
  GAME   pistol start wealth agrees within 5%, and non-pistol mid-round loadout means differ by more than 5% in at least one feature
  MIXED  otherwise

Writes data/diagnostics/attrib_2024_status.json.
Run in the cs2gpu environment (polars):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/attrib_2024_status.py
"""
import glob
import json
import os
import sys
import time

import polars as pl

ROOT = "G:/pureskill/csds-cs2"
OUT = os.path.join(ROOT, "data", "diagnostics", "attrib_2024_status.json")
CACHE = {"2024": os.path.join(ROOT, "data", "combined", "cache_snapshot_2024.parquet"),
         "2026": os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet")}
STATUS = {"2024": os.path.join(ROOT, "data", "status_1hz_2024"), "2026": os.path.join(ROOT, "data", "status_1hz")}
STATUS_COLS = ["money", "equipment_value_calc", "health", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_primary"]
MID_COLS = ["st_equip_ct", "st_equip_t", "st_flash_ct", "st_flash_t", "st_smoke_ct", "st_smoke_t", "st_he_ct", "st_he_t",
            "st_rifle_ct", "st_rifle_t"]
TOL = 0.05


def pistol_rounds(era):
    c = pl.read_parquet(CACHE[era], columns=["match_key", "round", "is_pistol_round"])
    return c.group_by(["match_key", "round"]).agg(pl.col("is_pistol_round").max().alias("pistol"))


def status_check(era):
    files = sorted(glob.glob(os.path.join(STATUS[era], "*.parquet")))
    s = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "second"] + STATUS_COLS) for f in files])
    nulls = {c: float(s[c].is_null().mean()) for c in STATUS_COLS}
    s = s.join(pistol_rounds(era), on=["match_key", "round"], how="inner")
    first = s.group_by(["match_key", "round"]).agg(pl.col("second").min().alias("s0"))
    start = s.join(first, on=["match_key", "round"]).filter((pl.col("second") == pl.col("s0")) & (pl.col("pistol") == 1))
    wealth = (start["money"].cast(pl.Float64) + start["equipment_value_calc"].cast(pl.Float64)).drop_nulls()
    return {"player_seconds": s.height, "null_share": nulls, "pistol_start_players": wealth.len(),
            "pistol_start_wealth_mean": float(wealth.mean()), "pistol_start_wealth_median": float(wealth.median()),
            "pistol_start_wealth_share_within_5pct_of_mean": float(((wealth - wealth.median()).abs() <= 0.05 * wealth.median()).mean())}


def mid_check(era):
    c = pl.read_parquet(CACHE[era], columns=["match_key", "round", "seconds_since_round_live", "is_pistol_round"] + MID_COLS)
    mid = c.filter((pl.col("is_pistol_round") == 0) & (pl.col("seconds_since_round_live") >= 20) & (pl.col("seconds_since_round_live") <= 30))
    return {col: float(mid[col].cast(pl.Float64).drop_nans().drop_nulls().mean()) for col in MID_COLS} | {"rows": mid.height}


def main():
    t0 = time.time()
    res = {"status": {era: status_check(era) for era in ["2024", "2026"]},
           "midround_nonpistol": {era: mid_check(era) for era in ["2024", "2026"]}}
    print(f"checks done ({time.time() - t0:.0f} s)", flush=True)
    w24 = res["status"]["2024"]["pistol_start_wealth_median"]
    w26 = res["status"]["2026"]["pistol_start_wealth_median"]
    wealth_rel = (w24 - w26) / abs(w26)
    null_bad = any(v > 0.01 for v in res["status"]["2024"]["null_share"].values())
    rel = {col: (res["midround_nonpistol"]["2024"][col] - res["midround_nonpistol"]["2026"][col]) / abs(res["midround_nonpistol"]["2026"][col])
           for col in MID_COLS}
    econ = any(abs(v) > TOL for v in rel.values())
    if abs(wealth_rel) > TOL or null_bad:
        verdict = "FIELD"
    elif econ:
        verdict = "GAME"
    else:
        verdict = "MIXED"
    res["pistol_wealth_relative_diff_2024_vs_2026"] = wealth_rel
    res["midround_relative_diff_2024_vs_2026"] = rel
    res["null_rate_over_1pct_in_2024"] = null_bad
    res["verdict"] = verdict
    res["rules"] = {"tolerance": TOL}
    res["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"verdict": verdict, "pistol_wealth_median": {"2024": w24, "2026": w26}, "wealth_rel": wealth_rel,
                      "midround_rel": rel}, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
