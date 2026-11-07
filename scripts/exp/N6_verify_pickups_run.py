# N6_verify: independent recomputation of primary-weapon drop/pickup claims.
# Era: 2026 only (match_key starts with csds/2026/). Unit of analysis: clean 5v5 round (round_end.player_count == 10).
# Written from raw event tomes; does not read the analyst script or its output.
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
ALT_PRIMARY = PRIMARY | {14, 28, 25, 27, 29, 35}  # sensitivity: + M249, Negev, shotguns
PRIM_L = sorted(PRIMARY)
ALT_L = sorted(ALT_PRIMARY)
REASONS = ["death", "manual", "replaced"]

# ---------------- rounds ----------------
re_all = lf26("round_end", ["match_key", "round", "tick", "player_count"]).collect()
clean = (re_all.filter(pl.col("player_count") == 10)
         .select(["match_key", "round", pl.col("tick").alias("round_end_tick")]))
N_CLEAN = clean.height
M_MATCH = clean["match_key"].n_unique()
print("clean 5v5 rounds:", N_CLEAN, " matches:", M_MATCH)

st = lf26("round_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["round_freeze_end", "round_start"])).collect()
fz = (st.filter(pl.col("event_type") == "round_freeze_end")
      .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("freeze_end_tick")))
rs = (st.filter(pl.col("event_type") == "round_start")
      .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("round_start_tick")))
clean = clean.join(fz, on=["match_key", "round"], how="left").join(rs, on=["match_key", "round"], how="left")
print("clean rounds with freeze_end:", clean["freeze_end_tick"].is_not_null().sum(),
      " with round_start:", clean["round_start_tick"].is_not_null().sum())
fz_gap = clean.filter(pl.col("freeze_end_tick").is_not_null() & pl.col("round_start_tick").is_not_null()).with_columns(
    ((pl.col("freeze_end_tick") - pl.col("round_start_tick")) / 64.0).alias("gap_s"))
print("freeze_end - round_start (s): median", fz_gap["gap_s"].median(), " min", fz_gap["gap_s"].min(), " max", fz_gap["gap_s"].max())

# ---------------- events ----------------
drops_all = lf26("item_dropped", ["match_key", "round", "tick", "player_id", "entity_id", "def_index", "item",
                                  "reason", "player_team_code", "x_pos", "y_pos"]).collect()
picks_all = lf26("item_pickup", ["match_key", "round", "tick", "player_id", "def_index", "item",
                                 "player_team_code", "player_x_pos", "player_y_pos"]).collect()
print("2026 item_dropped rows:", drops_all.height, " item_pickup rows:", picks_all.height)

BOUND_COLS = ["match_key", "round", "round_end_tick", "round_start_tick", "freeze_end_tick"]

def window_join(df, mode):
    j = df.join(clean.select(BOUND_COLS), on=["match_key", "round"], how="inner")
    if mode == "none":
        return j
    lo = {"round_start": pl.col("round_start_tick"), "freeze_end": pl.col("freeze_end_tick"),
          "le_end": pl.lit(0)}[mode].fill_null(0)
    return j.filter((pl.col("tick") >= lo) & (pl.col("tick") <= pl.col("round_end_tick")))

# ---------------- window grid (which definition matches the claimed denominators?) ----------------
grid = {}
for mode in ["none", "le_end", "freeze_end", "round_start"]:
    d = window_join(drops_all, mode)
    p = window_join(picks_all, mode)
    grid[mode] = {
        "drops_all": d.height,
        "drops_primary": d.filter(pl.col("def_index").is_in(PRIM_L)).height,
        "pickups_all": p.height,
        "pickups_primary": p.filter(pl.col("def_index").is_in(PRIM_L)).height,
        "drops_all_per_round": round(d.height / N_CLEAN, 4),
        "drops_primary_per_round": round(d.filter(pl.col("def_index").is_in(PRIM_L)).height / N_CLEAN, 4),
        "pickups_all_per_round": round(p.height / N_CLEAN, 4),
    }
    print("GRID", mode, grid[mode])

# ---------------- main configuration: window = round_start .. round_end ----------------
def build(mode_win, link_mode):
    dc = window_join(drops_all, mode_win).with_row_index("drid").with_columns(pl.col("drid").cast(pl.Int64))
    pc = window_join(picks_all, mode_win).with_row_index("pkid").with_columns(pl.col("pkid").cast(pl.Int64))

    def gkey(df):
        return pl.concat_str([pl.col("match_key"), pl.col("round").cast(pl.Utf8), pl.col("def_index").cast(pl.Utf8)],
                             separator="|")
    ev_d = dc.select([gkey(dc).alias("g"), pl.col("tick"), pl.lit(1).cast(pl.Int8).alias("kind"),
                      pl.col("drid").alias("did"), pl.lit(-1).cast(pl.Int64).alias("pid")])
    ev_p = pc.select([gkey(pc).alias("g"), pl.col("tick"), pl.lit(0).cast(pl.Int8).alias("kind"),
                      pl.lit(-1).cast(pl.Int64).alias("did"), pl.col("pkid").alias("pid")])
    ev = pl.concat([ev_d, ev_p]).sort(["g", "tick", "kind"])
    G = ev["g"].to_list(); K = ev["kind"].to_list(); D = ev["did"].to_list(); P = ev["pid"].to_list()
    del ev, ev_d, ev_p
    avail = deque(); prev = None; out_p = []; out_d = []
    for g, k, d, p in zip(G, K, D, P):
        if g != prev:
            avail.clear(); prev = g
        if k == 1:
            avail.append(d)
        elif avail:
            dd = avail.popleft() if link_mode == "fifo" else avail.pop()
            out_p.append(p); out_d.append(dd)
    del G, K, D, P
    pairs = pl.DataFrame({"pkid": out_p, "drid": out_d}, schema={"pkid": pl.Int64, "drid": pl.Int64})

    dsmall = dc.select(["drid", "match_key", "round", "def_index", "tick", "reason",
                        pl.col("player_team_code").alias("drop_team"), pl.col("player_id").alias("drop_pid")]
                       ).rename({"tick": "drop_tick"})
    psmall = pc.select(["pkid", "tick", pl.col("player_team_code").alias("pick_team"),
                        pl.col("player_id").alias("pick_pid"), "freeze_end_tick"]).rename({"tick": "pick_tick"})
    pf = pairs.join(dsmall, on="drid", how="inner").join(psmall, on="pkid", how="inner")
    pf = pf.with_columns([
        ((pl.col("pick_tick") - pl.col("drop_tick")) / 64.0).alias("delay_s"),
        (pl.col("pick_tick") < pl.col("freeze_end_tick")).fill_null(False).alias("pick_in_freeze"),
        (pl.col("drop_team").is_not_null() & pl.col("pick_team").is_not_null()).alias("team_known"),
        ((pl.col("drop_team") == pl.col("pick_team")) & pl.col("drop_team").is_not_null()
         & pl.col("pick_team").is_not_null()).alias("teammate"),
        (pl.col("drop_pid") == pl.col("pick_pid")).alias("same_player"),
    ])

    pk_max = pc.group_by(["match_key", "round", "def_index"]).agg(pl.col("tick").max().alias("max_pick_tick"))
    dc2 = dc.join(pk_max, on=["match_key", "round", "def_index"], how="left").with_columns([
        (pl.col("tick") < pl.col("max_pick_tick")).fill_null(False).alias("r1"),
        pl.col("drid").is_in(pairs["drid"]).alias("r2"),
    ])
    return dc2, pc, pairs, pf

dc2, pc, pairs, pf = build("round_start", "fifo")
print("main window FIFO: drops", dc2.height, " pickups", pc.height, " linked pairs (all defs)", pairs.height)

def summarize(dc2, pc, pf, pset, pairs):
    dp = dc2.filter(pl.col("def_index").is_in(pset))
    pfp = pf.filter(pl.col("def_index").is_in(pset))
    pcp = pc.filter(pl.col("def_index").is_in(pset))
    out = {"n_drops_primary": dp.height, "n_r1_any_later_pickup": dp.filter(pl.col("r1")).height,
           "n_r2_one_to_one_linked": dp.filter(pl.col("r2")).height, "n_pickups_primary_all": pcp.height,
           "n_ground_primary_pickups_linked": pfp.height,
           "n_rounds_with_ground_primary_pickup": pfp.select(["match_key", "round"]).unique().height,
           "by_reason": {}}
    for r in REASONS + ["all"]:
        sub = dp if r == "all" else dp.filter(pl.col("reason") == r)
        subp = pfp if r == "all" else pfp.filter(pl.col("reason") == r)
        n = sub.height
        n_r2 = sub.filter(pl.col("r2")).height
        n_r1 = sub.filter(pl.col("r1")).height
        known = subp.filter(pl.col("team_known")).height
        team = subp.filter(pl.col("teammate")).height
        med = float(subp["delay_s"].median()) if subp.height else None
        out["by_reason"][r] = {
            "n_drops": n, "n_r1": n_r1, "share_r1": round(n_r1 / n, 4) if n else None,
            "n_r2": n_r2, "share_r2": round(n_r2 / n, 4) if n else None,
            "n_r2_team_known": known, "n_r2_teammate": team,
            "n_r2_enemy": known - team,
            "share_teammate_of_known": round(team / known, 4) if known else None,
            "share_teammate_of_all_r2": round(team / subp.height, 4) if subp.height else None,
            "n_r2_null_team": subp.height - known,
            "median_delay_s": round(med, 3) if med is not None else None,
            "n_r2_picked_in_freeze": subp.filter(pl.col("pick_in_freeze")).height,
        }
    out["pickup_decomposition"] = {
        "all_pickups_clean_window": pc.height,
        "linked_ground_pickups_any_def": pairs.height,
        "unlinked_pickups_any_def_buy_spawn_or_unmatched": pc.height - pairs.height,
        "ground_share_of_primary_pickups": round(pfp.height / pcp.height, 4) if pcp.height else None,
    }
    return out

main = summarize(dc2, pc, pf, PRIMARY, pairs)
print("MAIN primary summary:", json.dumps(main["by_reason"]["all"]))
print("MAIN n_drops_primary:", main["n_drops_primary"], " n_r1:", main["n_r1_any_later_pickup"],
      " n_r2:", main["n_r2_one_to_one_linked"], " rounds w/ ground prim pickup:", main["n_rounds_with_ground_primary_pickup"])
print("MAIN per-reason:", {r: (main["by_reason"][r]["n_drops"], main["by_reason"][r]["n_r2"]) for r in REASONS})

# ---------------- sensitivity runs ----------------
sens = {}
sens["main_fifo_primary_alt_set"] = summarize(dc2, pc, pf, ALT_PRIMARY, pairs)
dc2_l, pc_l, pairs_l, pf_l = build("round_start", "lifo")
sens["lifo_primary"] = summarize(dc2_l, pc_l, pf_l, PRIMARY, pairs_l)
dc2_n, pc_n, pairs_n, pf_n = build("none", "fifo")
sens["no_window_fifo_primary"] = summarize(dc2_n, pc_n, pf_n, PRIMARY, pairs_n)
sens["no_window_fifo_primary_r1_only_drops"] = {"n_drops_primary": sens["no_window_fifo_primary"]["n_drops_primary"],
                                                 "n_r1": sens["no_window_fifo_primary"]["n_r1_any_later_pickup"]}
print("SENS lifo n_r2:", sens["lifo_primary"]["n_r2_one_to_one_linked"], " alt_set n_drops:", sens["main_fifo_primary_alt_set"]["n_drops_primary"])
print("SENS no-window n_drops_primary:", sens["no_window_fifo_primary"]["n_drops_primary"],
      " n_r2:", sens["no_window_fifo_primary"]["n_r2_one_to_one_linked"])
del dc2_l, pc_l, pairs_l, pf_l, dc2_n, pc_n, pairs_n, pf_n

# ---------------- diagnostics ----------------
diag = {}
name_norm = lambda s: s.lower().replace("-", "").replace(" ", "") if s else s
pn = pf.filter(pl.col("def_index").is_in(PRIM_L)).select(["match_key", "round", "def_index", "drop_pid"])
# naive text matching: compare drop item text vs pickup item text for linked primary pairs
linked_txt = (pairs.join(dc2.select(["drid", "item"]), on="drid", how="inner")
              .join(pc.select(["pkid", "item"]).rename({"item": "pick_item"}), on="pkid", how="inner")
              .filter(pl.col("def_index").is_in(PRIM_L) if "def_index" in pairs.columns else pl.lit(True)))
pf_items = pf.filter(pl.col("def_index").is_in(PRIM_L)).join(
    dc2.select(["drid", "item"]), on="drid", how="inner").join(pc.select(["pkid", "item"]).rename({"item": "pick_item"}), on="pkid", how="inner")
n_prim_linked = pf_items.height
name_fail = pf_items.filter(pl.col("item").str.to_lowercase().str.replace_all("[- ]", "") !=
                            pl.col("pick_item").str.to_lowercase().str.replace_all("[- ]", "")).height
diag["linked_primary_pairs"] = n_prim_linked
diag["linked_primary_pairs_name_match_fails"] = name_fail
diag["linked_primary_name_fail_share"] = round(name_fail / n_prim_linked, 4) if n_prim_linked else None
# identity diagnostics
diag["clean_window_drops_player_id_gt9"] = dc2.filter(pl.col("player_id") > 9).height
diag["clean_window_drops_player_id_max"] = int(dc2["player_id"].max())
diag["r2_primary_same_player_by_player_id"] = int(pf.filter(pl.col("def_index").is_in(PRIM_L))["same_player"].sum())
# null team
diag["r2_primary_null_team"] = int(pf.filter(pl.col("def_index").is_in(PRIM_L)).filter(~pl.col("team_known")).height)
diag["primary_drops_null_team"] = int(dc2.filter(pl.col("def_index").is_in(PRIM_L)).filter(pl.col("player_team_code").is_null()).height)
# negative delays guard
diag["negative_delay_pairs"] = int(pf.filter(pl.col("delay_s") < 0).height)
# drops/pickups before round_start (window check)
pre_rs = (drops_all.join(clean.select(BOUND_COLS), on=["match_key", "round"], how="inner")
          .filter(pl.col("tick") < pl.col("round_start_tick")).height)
post_re_d = (drops_all.join(clean.select(BOUND_COLS), on=["match_key", "round"], how="inner")
             .filter(pl.col("tick") > pl.col("round_end_tick")).height)
post_re_p = (picks_all.join(clean.select(BOUND_COLS), on=["match_key", "round"], how="inner")
             .filter(pl.col("tick") > pl.col("round_end_tick")).height)
diag["clean_drops_before_round_start"] = pre_rs
diag["clean_drops_after_round_end"] = post_re_d
diag["clean_pickups_after_round_end"] = post_re_p
print("DIAG", diag)

# ---------------- match-cluster bootstrap (B=500, seed 0) ----------------
mlist = clean["match_key"].unique().sort().to_list()
midx = {m: i for i, m in enumerate(mlist)}
M = len(mlist)
def per_match_count(df, name, filt=None):
    x = df if filt is None else df.filter(filt)
    g = x.group_by("match_key").len().rename({"len": name})
    a = np.zeros(M)
    for m, v in zip(g["match_key"].to_list(), g[name].to_list()):
        a[midx[m]] = v
    return a

rounds_m = np.zeros(M)
g = clean.group_by("match_key").len()
for m, v in zip(g["match_key"].to_list(), g["len"].to_list()):
    rounds_m[midx[m]] = v

pset_expr = pl.col("def_index").is_in(PRIM_L)
drops_all_m = per_match_count(dc2, "n")
drops_prim_m = per_match_count(dc2, "n", pset_expr)
pick_all_m = per_match_count(pc, "n")
pick_prim_m = per_match_count(pc, "n", pl.col("def_index").is_in(PRIM_L))
r1_m = per_match_count(dc2, "n", pset_expr & pl.col("r1"))
r2_m = per_match_count(dc2, "n", pset_expr & pl.col("r2"))
r2_freeze_m = per_match_count(pf, "n", pl.col("def_index").is_in(PRIM_L) & pl.col("pick_in_freeze"))
known_m = per_match_count(pf, "n", pl.col("def_index").is_in(PRIM_L) & pl.col("team_known"))
team_m = per_match_count(pf, "n", pl.col("def_index").is_in(PRIM_L) & pl.col("teammate"))
gpp_m = per_match_count(pf, "n", pl.col("def_index").is_in(PRIM_L))
rw_m = np.zeros(M)
rw = pf.filter(pl.col("def_index").is_in(PRIM_L)).select(["match_key", "round"]).unique().group_by("match_key").len()
for m, v in zip(rw["match_key"].to_list(), rw["len"].to_list()):
    rw_m[midx[m]] = v
reason_drop_m = {r: per_match_count(dc2, "n", pset_expr & (pl.col("reason") == r)) for r in REASONS}
reason_r2_m = {r: per_match_count(dc2, "n", pset_expr & pl.col("r2") & (pl.col("reason") == r)) for r in REASONS}
all_drop_r2_m = per_match_count(dc2, "n", pl.col("r2") & pset_expr)
# delays per match for bootstrap median
delays_by_m = [np.array([], dtype=float) for _ in range(M)]
dp_ = pf.filter(pl.col("def_index").is_in(PRIM_L)).select(["match_key", "delay_s"])
for m, grp in dp_.group_by("match_key"):
    delays_by_m[midx[m[0]]] = grp["delay_s"].to_numpy()

rng = np.random.default_rng(0)
B = 500
IDX = rng.integers(0, M, size=(B, M))

def ratio_boot(num, den):
    n = num[IDX].sum(axis=1)
    d = den[IDX].sum(axis=1)
    r = np.where(d > 0, n / np.where(d > 0, d, 1), np.nan)
    point = float(num.sum() / den.sum()) if den.sum() > 0 else None
    return {"point": point, "ci95_lo": float(np.nanpercentile(r, 2.5)), "ci95_hi": float(np.nanpercentile(r, 97.5)),
            "n_num": float(num.sum()), "n_den": float(den.sum())}

boot = {}
boot["share_r2_all"] = ratio_boot(r2_m, drops_prim_m)
boot["share_r1_all"] = ratio_boot(r1_m, drops_prim_m)
for r in REASONS:
    boot[f"share_r2_{r}"] = ratio_boot(reason_r2_m[r], reason_drop_m[r])
boot["teammate_share_of_known_r2"] = ratio_boot(team_m, known_m)
boot["freeze_share_of_r2"] = ratio_boot(r2_freeze_m, r2_m)
boot["drops_primary_per_round"] = ratio_boot(drops_prim_m, rounds_m)
boot["drops_all_per_round"] = ratio_boot(drops_all_m, rounds_m)
boot["pickups_all_per_round"] = ratio_boot(pick_all_m, rounds_m)
boot["rounds_with_ground_primary_pickup_share"] = ratio_boot(rw_m, rounds_m)
boot["ground_primary_pickups_per_round"] = ratio_boot(gpp_m, rounds_m)
# bootstrap median delay
meds = []
for b in range(B):
    sel = IDX[b]
    arr = np.concatenate([delays_by_m[i] for i in sel]) if len(sel) else np.array([])
    meds.append(float(np.median(arr)) if arr.size else np.nan)
boot["median_delay_s_r2_primary"] = {"point": float(np.median(dp_["delay_s"].to_numpy())),
                                     "ci95_lo": float(np.nanpercentile(meds, 2.5)),
                                     "ci95_hi": float(np.nanpercentile(meds, 97.5))}
for k, v in boot.items():
    print("BOOT", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})

# ---------------- claims check ----------------
def verdict(claim, mine, is_rate):
    if mine is None:
        return "CANNOT_RECOMPUTE"
    if is_rate:
        return "AGREE" if abs(mine - claim) <= 0.005 or abs(mine - claim) <= 0.01 * abs(claim) else "DISAGREE"
    return "AGREE" if abs(mine - claim) <= 0.01 * abs(claim) else "DISAGREE"

m_all = main["by_reason"]["all"]
claims = [
    ("drops_primary_per_round", 6.8939, main["n_drops_primary"] / N_CLEAN, False),
    ("drops_all_per_round", 25.5018, dc2.height / N_CLEAN, False),
    ("pickups_all_per_round", 44.0488, pc.height / N_CLEAN, False),
    ("primary_drop_picked_share_all", 0.2623, m_all["share_r2"], True),
    ("primary_drop_picked_share_death", 0.201, main["by_reason"]["death"]["share_r2"], True),
    ("primary_drop_picked_share_manual", 0.6536, main["by_reason"]["manual"]["share_r2"], True),
    ("primary_drop_picked_share_replaced", 0.2337, main["by_reason"]["replaced"]["share_r2"], True),
    ("picked_primary_drops_by_teammate_share_of_picked", 0.4536, m_all["share_teammate_of_known"], True),
]
claim_out = []
for name, c, mine, rate in claims:
    claim_out.append({"name": name, "claimed": c, "mine_one_to_one_fifo_main": (round(mine, 6) if mine is not None else None),
                      "verdict": verdict(c, mine, rate), "is_rate": rate})
# upper-bound (any later pickup) version of the share, for reference
claim_out.append({"name": "primary_drop_picked_share_all_upper_bound_r1", "claimed": 0.2623,
                  "mine_one_to_one_fifo_main": round(m_all["share_r1"], 6), "verdict": verdict(0.2623, m_all["share_r1"], True),
                  "is_rate": True})
for c in claim_out:
    print("CLAIM", c)

res = {
    "analysis_id": "N6_verify_pickups",
    "era": "2026 only (csds/2026/*)",
    "unit": "clean 5v5 rounds (round_end.player_count == 10)",
    "N_clean_rounds": N_CLEAN, "N_matches_clean": M_MATCH,
    "main_window": "round_start tick .. round_end tick (ASSUMPTION: round_start = min tick of round_state round_start per round)",
    "primary_def_index_sets": {"rifle": sorted(RIFLE), "smg": sorted(SMG), "sniper": sorted(SNIPER)},
    "grid": grid,
    "main": main,
    "sensitivity": sens,
    "bootstrap_match_cluster_B500_seed0": boot,
    "diagnostics": diag,
    "claims": claim_out,
    "runtime_s": round(time.time() - T0, 1),
}
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT, "runtime", round(time.time() - T0, 1), "s")
