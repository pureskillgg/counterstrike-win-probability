"""N6_pickups: ground pickups of dropped weapons, 2026 era, clean 5v5 rounds.

Run: <python> -I -X utf8 G:\\pureskill\\csds-cs2\\scripts\\exp\\N6_pickups_analysis.py
Writes: G:\\pureskill\\csds-cs2\\data\\viz\\N6_pickups_results.json
Uses only polars and numpy. Match-cluster bootstrap, 500 resamples, seed 0.
"""
import glob
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
VIZ = r"G:\pureskill\csds-cs2\data\viz"
OUT_JSON = os.path.join(VIZ, "N6_pickups_results.json")
NBOOT = 500
SEED = 0
TICK = 64.0
T0 = time.time()

# Primary = rifles, SMGs, snipers (task definition). Shotguns and LMGs are excluded.
PRIMARY_NAMES = {
    7: "AK-47", 8: "AUG", 10: "FAMAS", 13: "Galil AR", 16: "M4A4", 60: "M4A1", 39: "SG 553",
    9: "AWP", 11: "G3SG1", 38: "SCAR-20", 40: "SSG 08",
    17: "MAC-10", 19: "P90", 23: "MP5-SD", 24: "UMP-45", 26: "PP-Bizon", 33: "MP7", 34: "MP9",
}
PISTOL = {1, 2, 3, 4, 30, 32, 36, 61, 63, 64}
SHOTGUN_LMG = {14, 25, 27, 28, 29, 35}
EQUIP = {49, 50, 51, 55}
GRENADE = {43, 44, 45, 46, 47, 48}
CATS = ["primary", "pistol", "shotgun_lmg", "knife", "grenade", "equipment", "taser", "other"]
REASONS = ["death", "manual", "replaced"]


def cat_of(d):
    if d in PRIMARY_NAMES:
        return "primary"
    if d in PISTOL:
        return "pistol"
    if d in SHOTGUN_LMG:
        return "shotgun_lmg"
    if d == 42 or d == 59 or d >= 500:
        return "knife"
    if d in GRENADE:
        return "grenade"
    if d in EQUIP:
        return "equipment"
    if d == 31:
        return "taser"
    return "other"


def load(ch, cols):
    files = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}.COMBINED", "dataframe_*")))
    if not files:
        raise SystemExit("no pages for " + ch)
    lf = pl.scan_parquet(files).filter(pl.col("match_key").str.starts_with("csds/2026/"))
    return lf.select(["match_key"] + cols).collect()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- rounds
rend = load("round_end", ["round", "tick", "player_count"])
N_ROUND_END_2026 = rend.height
clean0 = rend.filter(pl.col("player_count") == 10).select(
    ["match_key", "round", pl.col("tick").alias("end_tick")])
N_CLEAN_ROUNDS = clean0.height
assert clean0.unique(["match_key", "round"]).height == N_CLEAN_ROUNDS, "duplicate clean rounds"
del rend
mk = clean0.select("match_key").unique().sort("match_key").with_row_index("midx")
M = mk.height
mk = mk.with_columns(
    pl.when(pl.col("match_key").str.slice(10, 5) == "10/08")
    .then(pl.lit("10/08")).otherwise(pl.lit("10/01-07")).alias("dgrp"))
clean = clean0.join(mk, on="match_key", how="inner")
DGRP = mk["dgrp"].to_numpy()
N_MATCH_DAYS = {g: int((DGRP == g).sum()) for g in ["10/01-07", "10/08"]}
log(f"2026 round_end rows={N_ROUND_END_2026} clean 5v5 rounds={N_CLEAN_ROUNDS} matches={M} {N_MATCH_DAYS}")


def in_clean(df):
    j = df.join(clean.select(["match_key", "round", "end_tick", "midx"]),
                on=["match_key", "round"], how="inner")
    return j.drop("match_key")


def per_match(df):
    out = np.zeros(M, dtype=np.float64)
    if df.height:
        g = df.group_by("midx").agg(pl.len().alias("n"))
        out[g["midx"].to_numpy().astype(np.int64)] = g["n"].to_numpy().astype(np.float64)
    return out


R = per_match(clean.select(["midx", "round"]))  # clean rounds per match

# ---------------------------------------------------------------- drops
dr_raw = load("item_dropped", ["round", "tick", "player_id", "item", "def_index",
                               "x_pos", "y_pos", "reason", "player_team_code"])
N_DROP_2026_ALL = dr_raw.height
dr = in_clean(dr_raw)
N_DROP_CLEAN_ALL = dr.height
N_DROP_AFTER_ROUND_END = dr.filter(pl.col("tick") > pl.col("end_tick")).height
dr = dr.filter(pl.col("tick") <= pl.col("end_tick"))
N_DROP_IN = dr.height
del dr_raw
log(f"item_dropped 2026 rows={N_DROP_2026_ALL} in clean rounds={N_DROP_CLEAN_ALL} "
    f"after round end (excluded)={N_DROP_AFTER_ROUND_END} kept={N_DROP_IN}")

# ---------------------------------------------------------------- pickups
pk_raw = load("item_pickup", ["round", "tick", "player_id", "item", "def_index",
                              "player_x_pos", "player_y_pos", "player_team_code"])
N_PICK_2026_ALL = pk_raw.height
pk = in_clean(pk_raw)
N_PICK_CLEAN_ALL = pk.height
N_PICK_AFTER_ROUND_END = pk.filter(pl.col("tick") > pl.col("end_tick")).height
pk = pk.filter(pl.col("tick") <= pl.col("end_tick"))
N_PICK_IN = pk.height
del pk_raw
log(f"item_pickup 2026 rows={N_PICK_2026_ALL} in clean rounds={N_PICK_CLEAN_ALL} "
    f"after round end (excluded)={N_PICK_AFTER_ROUND_END} kept={N_PICK_IN}")

# category by def_index (def_index is the key: item text differs between tables)
defs = sorted(set(dr["def_index"].unique().to_list()) | set(pk["def_index"].unique().to_list()))
defmap = pl.DataFrame({"def_index": defs, "cat": [cat_of(d) for d in defs]},
                      schema={"def_index": pl.Int64, "cat": pl.String})
dr = dr.join(defmap, on="def_index", how="left")
pk = pk.join(defmap, on="def_index", how="left")
log(f"distinct defs={len(defs)} cat counts drops={dr.group_by('cat').agg(pl.len().alias('n')).sort('cat').to_dicts()}")

# round start (buy-window check only)
rs_raw = load("round_start", ["round", "tick"])
rs = in_clean(rs_raw).group_by(["midx", "round"]).agg(pl.col("tick").min().alias("start_tick"))
del rs_raw
pk = pk.join(rs, on=["midx", "round"], how="left")
pk = pk.with_columns(((pl.col("tick") - pl.col("start_tick")).cast(pl.Float64) / TICK).alias("t_since_start_s"))

# deaths (validation of the 'death' drop reason)
de = in_clean(load("player_death", ["round", "tick", "player_id"]))

# ---------------------------------------------------------------- ground rule
# ground pickup: a drop of the same def_index exists in the same match and round at an earlier tick
fd = dr.group_by(["midx", "round", "def_index"]).agg(pl.col("tick").min().alias("first_drop_tick"))
pk = pk.join(fd, on=["midx", "round", "def_index"], how="left")
pk = pk.with_columns((pl.col("first_drop_tick") < pl.col("tick")).fill_null(False).alias("ground"))
N_GROUND_ALL = pk.filter(pl.col("ground")).height
log(f"ground pickups (all cats) = {N_GROUND_ALL} of {N_PICK_IN}")

# ---------------------------------------------------------------- one-to-one drop-pickup link (primary only)
pdr = dr.filter(pl.col("cat") == "primary").with_row_index("did").with_columns(
    pl.col("did").cast(pl.Int64))
groups = defaultdict(list)
for did, mi, rnd, dfi, tk, x, y in pdr.select(
        ["did", "midx", "round", "def_index", "tick", "x_pos", "y_pos"]).rows():
    groups[(mi, rnd, dfi)].append((tk, x, y, did))


def _dist(c, px, py):
    if c[1] is None or c[2] is None:
        return math.inf
    return math.hypot(c[1] - px, c[2] - py)


gpk = pk.filter((pl.col("cat") == "primary") & pl.col("ground")).with_row_index("pid").with_columns(
    pl.col("pid").cast(pl.Int64)).sort(["midx", "round", "def_index", "tick", "pid"])
gp_rows = gpk.select(["pid", "midx", "round", "def_index", "tick", "player_x_pos", "player_y_pos",
                      "player_id", "player_team_code"]).rows()

used = set()
assign = {}          # did -> (pick_tick, pick_player_id, pick_team, dist, pid)
unassigned_pids = []
for pid, mi, rnd, dfi, tk, px, py, ppid, pteam in gp_rows:
    cands = [c for c in groups.get((mi, rnd, dfi), ()) if c[0] < tk and c[3] not in used]
    if not cands:
        unassigned_pids.append(pid)
        continue
    if px is None or py is None:
        best = min(cands, key=lambda c: c[0])  # no position: earliest unconsumed drop
        dist = None
    else:
        best = min(cands, key=lambda c: _dist(c, px, py))  # nearest unconsumed earlier drop
        dist = _dist(best, px, py)
        if math.isinf(dist):
            dist = None
    used.add(best[3])
    assign[best[3]] = (tk, ppid, pteam, dist, pid)
N_GROUND_PRIM = len(gp_rows)
N_GROUND_PRIM_LINKED = len(assign)
N_GROUND_PRIM_UNLINKED = len(unassigned_pids)
log(f"ground primary pickups={N_GROUND_PRIM} linked={N_GROUND_PRIM_LINKED} unlinked={N_GROUND_PRIM_UNLINKED}")

ids = list(assign.keys())
asg = pl.DataFrame(
    {
        "did": pl.Series(ids, dtype=pl.Int64),
        "pick_tick": pl.Series([assign[d][0] for d in ids], dtype=pl.Int64),
        "pick_pid": pl.Series([assign[d][1] for d in ids], dtype=pl.Int64),
        "pick_team": pl.Series([assign[d][2] for d in ids], dtype=pl.Int64),
        "dist": pl.Series([assign[d][3] for d in ids], dtype=pl.Float64),
        "pid": pl.Series([assign[d][4] for d in ids], dtype=pl.Int64),
    })
pdr2 = pdr.join(asg, on="did", how="left").with_columns(
    pl.col("pick_tick").is_not_null().alias("matched"),
    ((pl.col("pick_tick") - pl.col("tick")).cast(pl.Float64) / TICK).alias("delta_s"),
)
pdr2 = pdr2.with_columns(
    pl.when(~pl.col("matched")).then(pl.lit("unmatched"))
    .when(pl.col("pick_pid") == pl.col("player_id")).then(pl.lit("self"))
    .when(pl.col("player_team_code").is_in([2, 3]) & pl.col("pick_team").is_in([2, 3])
          & (pl.col("player_team_code") == pl.col("pick_team"))).then(pl.lit("teammate"))
    .when(pl.col("player_team_code").is_in([2, 3]) & pl.col("pick_team").is_in([2, 3])).then(pl.lit("enemy"))
    .otherwise(pl.lit("unknown_team")).alias("teamcat"))
N_PDR = pdr2.height
N_PDR_MATCHED = int(pdr2["matched"].sum())
log(f"primary drops={N_PDR} matched={N_PDR_MATCHED}")

# validation: death drops vs player_death at (nearly) the same tick
dd = dr.filter(pl.col("reason") == "death").with_row_index("k").with_columns(pl.col("k").cast(pl.Int64))
ddj = dd.select(["k", "midx", "round", "player_id", "tick"]).join(
    de.select(["midx", "round", "player_id", pl.col("tick").alias("dtick")]),
    on=["midx", "round", "player_id"], how="left")
ddj = ddj.with_columns((pl.col("tick") - pl.col("dtick")).abs().alias("ad"))
bestd = ddj.group_by("k").agg(pl.col("ad").min())
N_DEATH_DROPS = dd.height
N_DEATH_EXACT = bestd.filter(pl.col("ad") == 0).height
N_DEATH_WITHIN8 = bestd.filter(pl.col("ad") <= 8).height
log(f"death drops={N_DEATH_DROPS} with player_death same tick={N_DEATH_EXACT} within 8 ticks={N_DEATH_WITHIN8}")

# validation: self-pickups should have equal team codes
self_rows = pdr2.filter(pl.col("teamcat") == "self")
N_SELF = self_rows.height
N_SELF_TEAM_KNOWN = self_rows.filter(pl.col("player_team_code").is_not_null() & pl.col("pick_team").is_not_null()).height
N_SELF_TEAM_EQUAL = self_rows.filter(pl.col("player_team_code") == pl.col("pick_team")).height
log(f"self-pickups={N_SELF} team known={N_SELF_TEAM_KNOWN} team equal={N_SELF_TEAM_EQUAL}")

# buy-window and late-pickup checks (rounds 2+; round 1 start tick is a known placeholder)
pk_r2 = pk.filter(pl.col("round") >= 2)
prim_nonground_r2 = pk_r2.filter((pl.col("cat") == "primary") & (~pl.col("ground")))
N_PNG_R2 = prim_nonground_r2.height
N_PNG_R2_LE30 = prim_nonground_r2.filter(pl.col("t_since_start_s") <= 30).height
N_PNG_R2_GT60 = prim_nonground_r2.filter(pl.col("t_since_start_s") > 60).height
gp_r2 = pk_r2.filter((pl.col("cat") == "primary") & pl.col("ground"))
N_GP_R2 = gp_r2.height
N_GP_R2_LT15 = gp_r2.filter(pl.col("t_since_start_s") < 15).height
log(f"non-ground primary pickups r2+={N_PNG_R2} <=30s={N_PNG_R2_LE30} >60s={N_PNG_R2_GT60}; "
    f"ground primary r2+={N_GP_R2} <15s={N_GP_R2_LT15}")

# ---------------------------------------------------------------- per-match arrays
drops_all = per_match(dr.select("midx"))
drops_by_reason = {r: per_match(dr.filter(pl.col("reason") == r).select("midx")) for r in REASONS}
drops_by_cat = {c: per_match(dr.filter(pl.col("cat") == c).select("midx")) for c in CATS}
drops_prim = per_match(pdr2.select("midx"))
drops_prim_by_reason = {r: per_match(pdr2.filter(pl.col("reason") == r).select("midx")) for r in REASONS}
drops_prim_matched = per_match(pdr2.filter(pl.col("matched")).select("midx"))
drops_prim_matched_by_reason = {
    r: per_match(pdr2.filter(pl.col("matched") & (pl.col("reason") == r)).select("midx")) for r in REASONS}
TEAMCATS = ["self", "teammate", "enemy", "unknown_team"]
drops_prim_by_teamcat = {t: per_match(pdr2.filter(pl.col("teamcat") == t).select("midx")) for t in TEAMCATS}

picks_all = per_match(pk.select("midx"))
picks_ground = per_match(pk.filter(pl.col("ground")).select("midx"))
picks_by_cat = {c: per_match(pk.filter(pl.col("cat") == c).select("midx")) for c in CATS}
picks_by_cat_ground = {c: per_match(pk.filter((pl.col("cat") == c) & pl.col("ground")).select("midx"))
                       for c in CATS}
picks_prim = picks_by_cat["primary"]
picks_prim_ground = picks_by_cat_ground["primary"]
picks_prim_nonground = picks_prim - picks_prim_ground
rounds_with_ground_prim = per_match(gpk.select(["midx", "round"]).unique())
rounds_with_prim_pick = per_match(pk.filter(pl.col("cat") == "primary").select(["midx", "round"]).unique())
prim_unlinked_per_match = per_match(gpk.filter(pl.col("pid").is_in(unassigned_pids)).select("midx"))

# ---------------------------------------------------------------- bootstrap helpers
def boot_index(n_units):
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n_units, size=(NBOOT, n_units))


def ratio_ci(num, den):
    """Ratio of sums over matches, match-cluster bootstrap (percentile 95%)."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    n = len(num)
    d0 = den.sum()
    point = float(num.sum() / d0) if d0 > 0 else float("nan")
    idx = boot_index(n)
    stats = np.empty(NBOOT)
    for b in range(NBOOT):
        w = np.bincount(idx[b], minlength=n).astype(np.float64)
        d = w @ den
        stats[b] = (w @ num) / d if d > 0 else np.nan
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def median_ci(vals, mi):
    """Median of drop-level values; match-cluster bootstrap over matches (weights per match)."""
    vals = np.asarray(vals, dtype=np.float64)
    mi = np.asarray(mi, dtype=np.int64)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(vals))
    order = np.argsort(vals, kind="mergesort")
    v = vals[order]
    m = mi[order]
    idx = boot_index(M)
    stats = np.empty(NBOOT)
    for b in range(NBOOT):
        w = np.bincount(idx[b], minlength=M).astype(np.int64)
        ww = w[m]
        cw = np.cumsum(ww)
        T = int(cw[-1])
        if T == 0:
            stats[b] = np.nan
            continue
        p1 = (T - 1) // 2
        p2 = T // 2
        k1 = np.searchsorted(cw, p1, side="right")
        k2 = np.searchsorted(cw, p2, side="right")
        stats[b] = 0.5 * (v[k1] + v[k2])
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


NUMBERS = []
TABLE = []


def add_ratio(name, num, den, definition, sel=None):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if sel is not None:
        num, den = num[sel], den[sel]
    point, lo, hi = ratio_ci(num, den)
    n_num = float(num.sum())
    n_den = float(den.sum())
    row = {"name": name, "value": point, "ci95_low": lo, "ci95_high": hi,
           "n_num": n_num, "n_rows": n_den, "n_matches": int(len(num)) if sel is None else int(sel.sum()),
           "definition": definition}
    NUMBERS.append(row)
    return row


def add_median(name, vals, mi, n_rows, definition):
    point, lo, hi = median_ci(vals, mi)
    row = {"name": name, "value": point, "ci95_low": lo, "ci95_high": hi,
           "n_rows": int(n_rows), "n_matches": int(len(np.unique(mi))) if len(mi) else 0,
           "definition": definition}
    NUMBERS.append(row)
    return row


# ---------------------------------------------------------------- results (1): counts per round
per_round_defs = [
    ("drops_all_per_round", drops_all, "all item_dropped rows per clean round"),
    ("drops_death_per_round", drops_by_reason["death"], "item_dropped reason=death per clean round"),
    ("drops_manual_per_round", drops_by_reason["manual"], "item_dropped reason=manual per clean round"),
    ("drops_replaced_per_round", drops_by_reason["replaced"], "item_dropped reason=replaced per clean round"),
    ("drops_primary_per_round", drops_prim, "dropped primary weapons per clean round"),
    ("drops_primary_death_per_round", drops_prim_by_reason["death"], "dropped primary, reason=death, per round"),
    ("drops_primary_manual_per_round", drops_prim_by_reason["manual"], "dropped primary, reason=manual, per round"),
    ("drops_primary_replaced_per_round", drops_prim_by_reason["replaced"], "dropped primary, reason=replaced, per round"),
    ("pickups_all_per_round", picks_all, "all item_pickup rows per clean round (buy/spawn + ground)"),
    ("pickups_ground_per_round", picks_ground, "item_pickup rows with an earlier same-def drop in the round (any category)"),
    ("pickups_nonground_per_round", picks_all - picks_ground, "item_pickup rows without an earlier same-def drop (buy/spawn/unexplained)"),
    ("pickups_primary_per_round", picks_prim, "primary item_pickup rows per clean round (buy/spawn + ground)"),
    ("pickups_primary_ground_per_round", picks_prim_ground, "ground primary pickups per clean round"),
    ("pickups_primary_nonground_per_round", picks_prim_nonground, "non-ground primary pickups per clean round"),
]
for name, arr, d in per_round_defs:
    add_ratio(name, arr, R, d)

for c in CATS:
    add_ratio(f"pickups_{c}_per_round", picks_by_cat[c], R, f"item_pickup category={c} per clean round")
    add_ratio(f"pickups_{c}_ground_per_round", picks_by_cat_ground[c], R,
              f"ground item_pickup category={c} per clean round")
    add_ratio(f"drops_{c}_per_round", drops_by_cat[c], R, f"item_dropped category={c} per clean round")

# ---------------------------------------------------------------- results (2): share of primary drops picked up
add_ratio("primary_drop_picked_share_all", drops_prim_matched, drops_prim,
          "share of primary drops with a linked ground pickup later in the same round")
for r in REASONS:
    add_ratio(f"primary_drop_picked_share_{r}", drops_prim_matched_by_reason[r], drops_prim_by_reason[r],
              f"share of primary drops with reason={r} that were picked up later in the round")
add_ratio("primary_drop_picked_share_dropper_died", drops_prim_matched_by_reason["death"],
          drops_prim_by_reason["death"], "dropper died (reason=death)")
add_ratio("primary_drop_picked_share_dropper_alive",
          drops_prim_matched_by_reason["manual"] + drops_prim_matched_by_reason["replaced"],
          drops_prim_by_reason["manual"] + drops_prim_by_reason["replaced"],
          "dropper alive (reason=manual or replaced)")
for t in ["self", "teammate", "enemy"]:
    add_ratio(f"primary_drop_picked_by_{t}_share_of_all_drops", drops_prim_by_teamcat[t], drops_prim,
              f"share of ALL primary drops picked up by a {t} (dropper team vs picker team)")
for t in ["self", "teammate", "enemy", "unknown_team"]:
    add_ratio(f"picked_primary_drops_by_{t}_share_of_picked", drops_prim_by_teamcat[t], drops_prim_matched,
              f"among picked-up primary drops, share picked by a {t}")

# ---------------------------------------------------------------- results (3): time to pickup
mpk = pdr2.filter(pl.col("matched"))
mv = mpk["delta_s"].to_numpy()
mm = mpk["midx"].to_numpy().astype(np.int64)
add_median("median_seconds_drop_to_pickup_all", mv, mm, len(mv), "median seconds drop->pickup, all picked primary drops")
for r in REASONS:
    sub = mpk.filter(pl.col("reason") == r)
    add_median(f"median_seconds_drop_to_pickup_{r}", sub["delta_s"].to_numpy(),
               sub["midx"].to_numpy().astype(np.int64), sub.height,
               f"median seconds drop->pickup, picked primary drops with reason={r}")
tm_team = mpk.filter(pl.col("teamcat").is_in(["self", "teammate"]))
add_median("median_seconds_drop_to_pickup_same_team", tm_team["delta_s"].to_numpy(),
           tm_team["midx"].to_numpy().astype(np.int64), tm_team.height,
           "median seconds, picked primary drops picked by self or teammate")
tm_en = mpk.filter(pl.col("teamcat") == "enemy")
add_median("median_seconds_drop_to_pickup_enemy", tm_en["delta_s"].to_numpy(),
           tm_en["midx"].to_numpy().astype(np.int64), tm_en.height,
           "median seconds, picked primary drops picked by an enemy")

# ---------------------------------------------------------------- results (4): rounds with ground primary pickup
add_ratio("share_rounds_with_ground_primary_pickup", rounds_with_ground_prim, R,
          "share of clean rounds with >=1 ground primary pickup")
add_ratio("mean_ground_primary_pickups_per_round", picks_prim_ground, R,
          "mean ground primary pickups per clean round")

# ---------------------------------------------------------------- results (5): fraction of primary acquisitions from ground
add_ratio("share_primary_acquisitions_from_ground", picks_prim_ground, picks_prim,
          "ground primary pickups / all primary pickups (buy/spawn + ground), per clean round set")
add_ratio("share_all_pickups_from_ground", picks_ground, picks_all,
          "ground pickups / all item_pickup rows, all categories")
add_ratio("share_rounds_with_any_primary_acquisition", rounds_with_prim_pick, R,
          "share of clean rounds with >=1 primary pickup of any kind")

# ---------------------------------------------------------------- day split (robustness)
DAY_TABLE = []
for g in ["10/01-07", "10/08"]:
    sel = DGRP == g
    if sel.sum() == 0:
        continue
    for name, num, den, d in [
        ("primary_drop_picked_share_all", drops_prim_matched, drops_prim, "primary drops picked up later in round"),
        ("share_primary_acquisitions_from_ground", picks_prim_ground, picks_prim, "ground / all primary pickups"),
        ("mean_ground_primary_pickups_per_round", picks_prim_ground, R, "ground primary pickups per round"),
        ("share_rounds_with_ground_primary_pickup", rounds_with_ground_prim, R, "rounds with >=1 ground primary pickup"),
    ]:
        point, lo, hi = ratio_ci(np.asarray(num)[sel], np.asarray(den)[sel])
        DAY_TABLE.append({"section": "day_split", "group": g, "measure": name, "definition": d,
                          "value": point, "ci95_low": lo, "ci95_high": hi,
                          "n_num": float(np.asarray(num)[sel].sum()), "n_rows": float(np.asarray(den)[sel].sum()),
                          "n_matches": int(sel.sum())})

# ---------------------------------------------------------------- weapon table
wt = pdr2.group_by(["def_index", "item"]).agg(
    pl.len().alias("n_drops"),
    pl.col("matched").sum().alias("n_picked"),
    (pl.col("reason") == "death").sum().alias("n_death_drops")).sort("n_drops", descending=True)
gpk_def = gpk.group_by("def_index").agg(pl.len().alias("n_ground_pickups"))
wt = wt.join(gpk_def, on="def_index", how="left").with_columns(
    pl.col("n_ground_pickups").fill_null(0),
    (pl.col("n_picked") / pl.col("n_drops")).alias("picked_share"))
WEAPON_TABLE = []
for row in wt.iter_rows(named=True):
    WEAPON_TABLE.append({"section": "weapon", "def_index": int(row["def_index"]),
                         "weapon_name_in_drops": row["item"],
                         "n_drops": int(row["n_drops"]), "n_picked": int(row["n_picked"]),
                         "picked_share": float(row["picked_share"]),
                         "n_death_drops": int(row["n_death_drops"]),
                         "n_ground_pickups": int(row["n_ground_pickups"])})

# ---------------------------------------------------------------- validation and checks
CHECKS = {
    "clean_rounds": N_CLEAN_ROUNDS,
    "clean_matches": M,
    "round_end_2026_rows": N_ROUND_END_2026,
    "matches_day_10_01_07": N_MATCH_DAYS["10/01-07"],
    "matches_day_10_08": N_MATCH_DAYS["10/08"],
    "drops_2026_all": N_DROP_2026_ALL,
    "drops_in_clean_rounds_before_cutoff": N_DROP_IN,
    "drops_excluded_after_round_end": N_DROP_AFTER_ROUND_END,
    "pickups_2026_all": N_PICK_2026_ALL,
    "pickups_in_clean_rounds_before_cutoff": N_PICK_IN,
    "pickups_excluded_after_round_end": N_PICK_AFTER_ROUND_END,
    "ground_pickups_all_cats": N_GROUND_ALL,
    "ground_primary_pickups": N_GROUND_PRIM,
    "ground_primary_linked_to_a_drop": N_GROUND_PRIM_LINKED,
    "ground_primary_unlinked_no_free_drop": N_GROUND_PRIM_UNLINKED,
    "primary_drops": N_PDR,
    "primary_drops_picked": N_PDR_MATCHED,
    "death_drops": N_DEATH_DROPS,
    "death_drops_with_player_death_same_tick": N_DEATH_EXACT,
    "death_drops_with_player_death_within_8_ticks": N_DEATH_WITHIN8,
    "self_pickups_primary": N_SELF,
    "self_pickups_team_known": N_SELF_TEAM_KNOWN,
    "self_pickups_team_equal": N_SELF_TEAM_EQUAL,
    "non_ground_primary_pickups_r2plus": N_PNG_R2,
    "non_ground_primary_pickups_r2plus_within_30s": N_PNG_R2_LE30,
    "non_ground_primary_pickups_r2plus_after_60s": N_PNG_R2_GT60,
    "ground_primary_pickups_r2plus": N_GP_R2,
    "ground_primary_pickups_r2plus_under_15s": N_GP_R2_LT15,
}
ASSUMPTIONS = [
    "ASSUMPTION primary = rifles, SMGs and snipers as the task defines it (rifles 7,8,10,13,16,39,60; snipers 9,11,38,40; SMGs 17,19,23,24,26,33,34). Shotguns (25,27,29,35) and LMGs (14,28) are excluded.",
    "ASSUMPTION item_pickup and item_dropped are joined on def_index, not on item text. The text differs between tables (USP-S in item_dropped vs hkp2000 in item_pickup; M4A1 vs m4a1; Galil AR vs galilar), so a name match would miss pairs.",
    "ASSUMPTION a pickup is a ground pickup if an item_dropped row with the same match, round and def_index exists at a strictly earlier tick (the task rule, using def_index). Same-tick drop/pickup pairs (weapon swaps) are not counted as ground.",
    "ASSUMPTION drop-to-pickup links are one-to-one: each drop can be linked once. A ground pickup takes the unconsumed earlier drop of the same def with the nearest position to the picker (fallback: earliest drop if positions are null). Links can be wrong when several same-def weapons lie in one round; this affects drop-side shares and times, not the ground pickup counts.",
    "ASSUMPTION a 'weapon change' is a primary acquisition in a round (item_pickup of a primary, whether buy/spawn or ground). item_equip (weapon switches) was not used. This denominator is the basis of the one-sentence conclusion.",
    "ASSUMPTION the dropper is item_dropped.player_id and the picker is item_pickup.player_id. Both are raw per-match indices. They are not compared across matches. Identity is checked by self-pickup team agreement and by death-drop links to player_death.",
    "ASSUMPTION team = player_team_code on the event row (SIDE code 2=T, 3=CT). Rows with a null or 0 code are classed unknown_team and excluded from the teammate/enemy split.",
    "ASSUMPTION dropper died = reason 'death'. manual and replaced are treated as dropper alive.",
    "ASSUMPTION events after round_end.tick are excluded as post-round logging (the handoff reports 2-4% of weapon rows after round end). Cutoff: tick <= round_end.tick.",
    "ASSUMPTION median seconds use tick/64. Position units are CS2 world units (about 2.54 cm each).",
    "ASSUMPTION clean rounds are round_end.player_count == 10 (task rule). All 2026 matches in combined_test are used (days 10/01 to 10/08). Days 09 and 10 are absent from this folder.",
    "ASSUMPTION bootstrap: resample matches with replacement, 500 draws, seed 0 (numpy default_rng). Ratio CIs use the ratio of summed match counts. Median CIs use the weighted median over resampled matches. CIs are 2.5/97.5 percentiles.",
]
CAVEATS = [
    "2026-10-08 holds 1,135 of 1,752 matches (about 65%) and the handoff flags it as a batch with gaps in bullet_damage and player_sound. The day split above checks this.",
    "The task's '150+ matches' refers to the 1 Hz grid. This analysis uses every 2026 match in the combined event tome, so n is far larger.",
    "item_pickup includes buy and spawn equips, so ground pickups are only a subset. Non-ground primary pickups are not verified as purchases. The buy-window checks are indirect.",
    "The one-sentence conclusion depends on the denominator in ASSUMPTION 5. Primary weapon switches (item_equip) were not measured.",
    "Round 1 round_start tick is a placeholder in about 39% of matches (handoff). Buy-window checks use rounds 2 and up only.",
    "Dropped grenades and C4 are in item_dropped but are not primary, so they are excluded from the primary shares.",
]

log("writing JSON")
os.makedirs(VIZ, exist_ok=True)


def clean_num(x):
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (np.floating,)):
        return None if (np.isnan(x) or np.isinf(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return clean_num(obj)


TABLE = [
    {"section": "numbers", "measure": n["name"], "value": n["value"], "ci95_low": n["ci95_low"],
     "ci95_high": n["ci95_high"], "n_num": n.get("n_num"), "n_rows": n.get("n_rows"),
     "n_matches": n.get("n_matches"), "definition": n.get("definition")}
    for n in NUMBERS
]
TABLE += DAY_TABLE
out = {
    "id": "N6_pickups",
    "script": r"G:\pureskill\csds-cs2\scripts\exp\N6_pickups_analysis.py",
    "source": r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds (2026 era, basic_item_dropped, basic_item_pickup, basic_round_end, basic_round_start, basic_player_death)",
    "primary_weapons": {str(k): v for k, v in PRIMARY_NAMES.items()},
    "checks": CHECKS,
    "numbers": NUMBERS,
    "table": TABLE,
    "weapon_table": WEAPON_TABLE,
    "assumptions": ASSUMPTIONS,
    "caveats": CAVEATS,
    "bootstrap": {"resamples": NBOOT, "seed": SEED, "cluster": "match"},
}
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(sanitize(out), fh, indent=1)
log(f"wrote {OUT_JSON}")

# ---------------------------------------------------------------- console summary
print("CHECKS")
for k, v in CHECKS.items():
    print(f"  {k}: {v}")
print("NUMBERS")
for n in NUMBERS:
    v = n["value"]
    lo, hi = n["ci95_low"], n["ci95_high"]
    fmt = lambda x: "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    print(f"  {n['name']}: {fmt(v)} [{fmt(lo)}, {fmt(hi)}] n_rows={n.get('n_rows')} n_num={n.get('n_num')} matches={n.get('n_matches')}")
print("DAY SPLIT")
for r in DAY_TABLE:
    print(f"  {r['group']} {r['measure']}: {r['value']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] n_num={r['n_num']} n_rows={r['n_rows']} matches={r['n_matches']}")
print("WEAPON TABLE")
for w in WEAPON_TABLE:
    print(f"  def {w['def_index']} {w['weapon_name_in_drops']}: drops={w['n_drops']} picked={w['n_picked']} share={w['picked_share']:.3f} death_drops={w['n_death_drops']} ground_pickups={w['n_ground_pickups']}")
