# N6_verify: independent recomputation of primary-weapon drop/pickup claims (2026 era, clean 5v5 rounds)
import glob, json, time
from collections import deque
import numpy as np
import polars as pl

T0 = time.time()
BASE = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
OUT = "G:/pureskill/csds-cs2/data/viz/N6_verify_pickups_results.json"

def files(ch):
    return sorted(glob.glob(BASE + "/basic_" + ch + ".COMBINED/dataframe_*"))

def lf26(ch, cols=None):
    lf = pl.scan_parquet(files(ch)).filter(pl.col("match_key").str.starts_with("csds/2026/"))
    if cols:
        lf = lf.select(cols)
    return lf

# ASSUMPTION: primary = rifles + SMGs + snipers, keyed by item_dropped def_index (display names).
RIFLE = {7, 8, 10, 13, 16, 39, 60}          # AK-47, AUG, FAMAS, Galil AR, M4A4, SG 553, M4A1
SMG = {17, 19, 23, 24, 26, 33, 34}          # MAC-10, P90, MP5-SD, UMP-45, PP-Bizon, MP7, MP9
SNIPER = {9, 11, 38, 40}                    # AWP, G3SG1, SCAR-20, SSG 08
PRIMARY = RIFLE | SMG | SNIPER
ALT_PRIMARY = PRIMARY | {14, 28, 25, 27, 29, 35}  # sensitivity: + LMG (M249, Negev) + shotguns
REASONS = ["death", "manual", "replaced"]

# ---- rounds ----
re_all = lf26("round_end", ["match_key", "round", "tick", "player_count"]).collect()
clean = (re_all.filter(pl.col("player_count") == 10)
         .select(["match_key", "round", pl.col("tick").alias("round_end_tick")]))
N_CLEAN = clean.height
print("clean 5v5 rounds:", N_CLEAN, " matches:", clean["match_key"].n_unique())
st = lf26("round_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["round_freeze_end", "round_start"])).collect()
fz = (st.filter(pl.col("event_type") == "round_freeze_end")
      .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("freeze_end_tick")))
rs = (st.filter(pl.col("event_type") == "round_start")
      .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("round_start_tick")))
print("round_start event count per (match,round): ", rs.height, " rows:", st.filter(pl.col("event_type")=="round_start").height)
clean = clean.join(fz, on=["match_key", "round"], how="left").join(rs, on=["match_key", "round"], how="left")
print("clean rounds with freeze_end:", clean["freeze_end_tick"].is_not_null().sum(),
      " with round_start:", clean["round_start_tick"].is_not_null().sum())
clean_keys = clean.select(["match_key", "round"])

drops_all = lf26("item_dropped", ["match_key", "round", "tick", "player_id", "entity_id", "def_index", "item",
                                  "reason", "player_team_code", "x_pos", "y_pos"]).collect()
picks_all = lf26("item_pickup", ["match_key", "round", "tick", "player_id", "def_index", "item",
                                 "player_team_code", "player_x_pos", "player_y_pos"]).collect()

# ---- window grid: which definition reproduces the claimed denominators? ----
def window_counts(dfr, pkr, lo_col, hi_col="round_end_tick"):
    j = clean.select(["match_key", "round", lo_col, hi_col])
    d = dfr.join(j, on=["match_key", "round"], how="inner")
    p = pkr.join(j, on=["match_key", "round"], how="inner")
    dw = d.filter((pl.col("tick") >= pl.col(lo_col)) & (pl.col("tick") <= pl.col(hi_col)))
    pw = p.filter((pl.col("tick") >= pl.col(lo_col)) & (pl.col("tick") <= pl.col(hi_col)))
    return {
        "drops_all": dw.height, "drops_primary": dw.filter(pl.col("def_index").is_in(list(PRIMARY))).height,
        "pickups_all": pw.height, "pickups_primary": pw.filter(pl.col("def_index").is_in(list(PRIMARY))).height,
    }
grid = {}
grid["no_window_clean_rounds"] = {
    "drops_all": drops_all.join(clean_keys, on=["match_key","round"]).height,
    "drops_primary": drops_all.join(clean_keys, on=["match_key","round"]).filter(pl.col("def_index").is_in(list(PRIMARY))).height,
    "pickups_all": picks_all.join(clean_keys, on=["match_key","round"]).height,
    "pickups_primary": picks_all.join(clean_keys, on=["match_key","round"]).filter(pl.col("def_index").is_in(list(PRIMARY))).height,
}
# round_end-bounded only (tick <= round_end), no lower bound
clean_lo0 = clean.with_columns(pl.lit(0).alias("lo0"))
grid["le_round_end_only"] = window_counts(drops_all, picks_all, "lo0")
clean_fz = clean.with_columns(pl.col("freeze_end_tick").fill_null(0).alias("lo_fz"))
clean.drop_nulls  # noop
clean_join_fz = clean.select(["match_key","round","round_end_tick", "freeze_end_tick"]).with_columns(pl.col("freeze_end_tick").fill_null(0).alias("lo_fz"))
grid["freeze_end_to_round_end"] = window_counts(drops_all, picks_all, "lo_fz")
clean_join_rs = clean.select(["match_key","round","round_end_tick","round_start_tick"]).with_columns(pl.col("round_start_tick").fill_null(0).alias("lo_rs"))
grid["round_start_to_round_end"] = window_counts(drops_all, picks_all, "lo_rs")
for k, v in grid.items():
    print("WINDOW", k, v, " per-round drops_all:", round(v["drops_all"]/N_CLEAN, 4), " pickups_all:", round(v["pickups_all"]/N_CLEAN, 4))
print("CLAIM targets: drops_all_per_round=25.5018 (=817,103 rows), pickups_all_per_round=44.0488 (=1,411,368 rows), drops_primary_per_round=6.8939 (=220,887 rows)")

json.dump({"window_grid": grid, "N_CLEAN": N_CLEAN}, open("G:/pureskill/csds-cs2/data/viz/N6_verify_pickups_grid.json", "w"), indent=1)
print("elapsed", round(time.time()-T0,1), "s")
