"""N6_pickups v2: ground pickups of dropped primary weapons, 2026 era, clean 5v5 rounds.

Task rule (headline): a pickup is 'ground' if an item_dropped row with the same match, round and
def_index exists at a strictly earlier tick. Links to drops are one-to-one (nearest unconsumed drop).
Strict variant (sensitivity): the link must also satisfy distance(picker, drop) <= radius (R in units).

Run: <python> -I -X utf8 G:\\pureskill\\csds-cs2\\scripts\\exp\\N6_pickups_analysis_v2.py
Writes: G:\\pureskill\\csds-cs2\\data\\viz\\N6_pickups_results_v2.json
Uses polars and numpy only. Match-cluster bootstrap, 500 resamples, seed 0.
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
OUT_JSON = os.path.join(VIZ, "N6_pickups_results_v2.json")
NBOOT = 500
SEED = 0
TICK = 64.0
FREEZE_S = 20.0           # ASSUMPTION: header.freeze_time = 20 (schema_dictionary.md)
STRICT_R = 200.0          # ASSUMPTION: 200 world units (about 5 m) as the strict link radius
RADII = {"task": None, "r400": 400.0, "r200": 200.0, "r100": 100.0}
T0 = time.time()

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
TEAMCATS = ["self", "teammate", "enemy", "unknown_team"]


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
log(f"clean 5v5 rounds={N_CLEAN_ROUNDS} matches={M} {N_MATCH_DAYS}")


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


R = per_match(clean.select(["midx", "round"]))                                # rounds per match
R2 = per_match(clean.filter(pl.col("round") >= 2).select(["midx", "round"]))  # rounds 2+ per match

# ---------------------------------------------------------------- drops
dr_raw = load("item_dropped", ["round", "tick", "player_id", "item", "def_index",
                               "x_pos", "y_pos", "reason", "player_team_code"])
N_DROP_2026_ALL = dr_raw.height
dr_all_clean = in_clean(dr_raw)
N_DROP_CLEAN_ALL = dr_all_clean.height
dr_post = dr_all_clean.filter(pl.col("tick") > pl.col("end_tick"))
N_DROP_POST = dr_post.height
POST_DROP_BY_REASON = {r: int(dr_post.filter(pl.col("reason") == r).height) for r in REASONS}
POST_DROP_PRIMARY = int(dr_post.filter(pl.col("def_index").is_in(list(PRIMARY_NAMES))).height)
dr = dr_all_clean.filter(pl.col("tick") <= pl.col("end_tick"))
N_DROP_IN = dr.height
del dr_raw, dr_all_clean, dr_post
log(f"item_dropped 2026={N_DROP_2026_ALL} clean-round={N_DROP_CLEAN_ALL} post-end={N_DROP_POST} kept={N_DROP_IN}")

# ---------------------------------------------------------------- pickups
pk_raw = load("item_pickup", ["round", "tick", "player_id", "item", "def_index",
                              "player_x_pos", "player_y_pos", "player_team_code"])
N_PICK_2026_ALL = pk_raw.height
pk_all_clean = in_clean(pk_raw)
N_PICK_CLEAN_ALL = pk_all_clean.height
pk_post = pk_all_clean.filter(pl.col("tick") > pl.col("end_tick")).with_columns(
    ((pl.col("tick") - pl.col("end_tick")).cast(pl.Float64) / TICK).alias("after_s"))
N_PICK_POST = pk_post.height
POST_PICK_PRIMARY = int(pk_post.filter(pl.col("def_index").is_in(list(PRIMARY_NAMES))).height)
POST_PICK_AFTER_MEDIAN_S = float(np.median(pk_post["after_s"].to_numpy())) if pk_post.height else float("nan")
POST_PICK_TOP_ITEMS = pk_post.group_by("item").agg(pl.len().alias("n")).sort("n", descending=True).head(6).to_dicts()
pk = pk_all_clean.filter(pl.col("tick") <= pl.col("end_tick"))
N_PICK_IN = pk.height
del pk_raw, pk_all_clean, pk_post
log(f"item_pickup 2026={N_PICK_2026_ALL} clean-round={N_PICK_CLEAN_ALL} post-end={N_PICK_POST} kept={N_PICK_IN}")

# category by def_index (def_index is the key: the item text differs between tables)
defs = sorted(set(dr["def_index"].unique().to_list()) | set(pk["def_index"].unique().to_list()))
defmap = pl.DataFrame({"def_index": defs, "cat": [cat_of(d) for d in defs]},
                      schema={"def_index": pl.Int64, "cat": pl.String})
dr = dr.join(defmap, on="def_index", how="left")
pk = pk.join(defmap, on="def_index", how="left")

# round start (freeze-window split only; round 1 start is a placeholder in ~39% of matches)
rs_raw = load("round_start", ["round", "tick"])
rs = in_clean(rs_raw).group_by(["midx", "round"]).agg(pl.col("tick").min().alias("start_tick"))
del rs_raw
pk = pk.join(rs, on=["midx", "round"], how="left").with_columns(
    ((pl.col("tick") - pl.col("start_tick")).cast(pl.Float64) / TICK).alias("ts"),
    ((pl.col("tick") - pl.col("start_tick")).cast(pl.Float64) / TICK < FREEZE_S).fill_null(False).alias("early"))
dr = dr.join(rs, on=["midx", "round"], how="left")

# deaths (validation of the 'death' drop reason)
de = in_clean(load("player_death", ["round", "tick", "player_id"]))

# ---------------------------------------------------------------- ground rule (task)
fd = dr.group_by(["midx", "round", "def_index"]).agg(pl.col("tick").min().alias("first_drop_tick"))
pk = pk.join(fd, on=["midx", "round", "def_index"], how="left")
pk = pk.with_columns((pl.col("first_drop_tick") < pl.col("tick")).fill_null(False).alias("ground"))
N_GROUND_ALL = pk.filter(pl.col("ground")).height
log(f"ground pickups (all cats, task rule)={N_GROUND_ALL} of {N_PICK_IN}")

# ---------------------------------------------------------------- primary drops and ground primary pickups
pdr = dr.filter(pl.col("cat") == "primary").with_row_index("did").with_columns(
    pl.col("did").cast(pl.Int64))
groups = defaultdict(list)
for did, mi, rnd, dfi, tk, x, y in pdr.select(
        ["did", "midx", "round", "def_index", "tick", "x_pos", "y_pos"]).rows():
    groups[(mi, rnd, dfi)].append((tk, x, y, did))

gpk = pk.filter((pl.col("cat") == "primary") & pl.col("ground")).with_row_index("pid").with_columns(
    pl.col("pid").cast(pl.Int64)).sort(["midx", "round", "def_index", "tick", "pid"])
gp_rows = gpk.select(["pid", "midx", "round", "def_index", "tick", "player_x_pos", "player_y_pos",
                      "player_id", "player_team_code"]).rows()
N_GROUND_PRIM = len(gp_rows)
N_GROUND_PRIM_EARLY = int(gpk.filter(pl.col("early") & (pl.col("round") >= 2)).height)
N_GROUND_PRIM_LATE = int(gpk.filter((~pl.col("early")) & (pl.col("round") >= 2)).height)
N_GROUND_PRIM_R1 = int(gpk.filter(pl.col("round") == 1).height)
N_GROUND_PRIM_NOPOS = int(gpk.filter(pl.col("player_x_pos").is_null()).height)
log(f"ground primary pickups={N_GROUND_PRIM} (rounds2+ early={N_GROUND_PRIM_EARLY} late={N_GROUND_PRIM_LATE}) nopos={N_GROUND_PRIM_NOPOS}")


def _dist(c, px, py):
    if c[1] is None or c[2] is None:
        return math.inf
    return math.hypot(c[1] - px, c[2] - py)


def link(radius):
    """One-to-one link of ground primary pickups to unconsumed earlier drops of the same def.
    radius None: nearest unconsumed earlier drop (task rule). radius r: only drops within r units."""
    used = set()
    assign = {}
    unassigned = []
    for pid, mi, rnd, dfi, tk, px, py, ppid, pteam in gp_rows:
        cands = []
        for c in groups.get((mi, rnd, dfi), ()):
            if c[0] >= tk or c[3] in used:
                continue
            if radius is None:
                cands.append(c)
            else:
                if px is None or py is None or c[1] is None or c[2] is None:
                    continue
                if _dist(c, px, py) <= radius:
                    cands.append(c)
        if not cands:
            unassigned.append(pid)
            continue
        if px is None or py is None:
            best = min(cands, key=lambda c: c[0])  # no position: earliest unconsumed drop
            dist = None
        else:
            best = min(cands, key=lambda c: _dist(c, px, py))
            dist = _dist(best, px, py)
            if math.isinf(dist):
                dist = None
        used.add(best[3])
        assign[best[3]] = (tk, ppid, pteam, dist, pid)
    return assign, unassigned


rs_lookup = rs  # for pick-side freeze split
LINKS = {}
for name, rad in RADII.items():
    a, u = link(rad)
    LINKS[name] = {"assign": a, "unassigned": u}
    log(f"link variant {name}: linked={len(a)} unlinked={len(u)}")


def build_variant(name):
    assign = LINKS[name]["assign"]
    unassigned = LINKS[name]["unassigned"]
    ids = list(assign.keys())
    asg = pl.DataFrame({
        "did": pl.Series(ids, dtype=pl.Int64),
        "pick_tick": pl.Series([assign[d][0] for d in ids], dtype=pl.Int64),
        "pick_pid": pl.Series([assign[d][1] for d in ids], dtype=pl.Int64),
        "pick_team": pl.Series([assign[d][2] for d in ids], dtype=pl.Int64),
        "dist": pl.Series([assign[d][3] for d in ids], dtype=pl.Float64),
        "pid": pl.Series([assign[d][4] for d in ids], dtype=pl.Int64),
    })
    pv = pdr.join(asg, on="did", how="left").with_columns(
        pl.col("pick_tick").is_not_null().alias("matched"),
        ((pl.col("pick_tick") - pl.col("tick")).cast(pl.Float64) / TICK).alias("delta_s"),
    )
    pv = pv.join(rs_lookup, on=["midx", "round"], how="left").with_columns(
        ((pl.col("pick_tick") - pl.col("start_tick")).cast(pl.Float64) / TICK).alias("pick_ts"))
    pv = pv.with_columns(
        pl.when(~pl.col("matched")).then(pl.lit("unmatched"))
        .when(pl.col("pick_pid") == pl.col("player_id")).then(pl.lit("self"))
        .when(pl.col("player_team_code").is_in([2, 3]) & pl.col("pick_team").is_in([2, 3])
              & (pl.col("player_team_code") == pl.col("pick_team"))).then(pl.lit("teammate"))
        .when(pl.col("player_team_code").is_in([2, 3]) & pl.col("pick_team").is_in([2, 3]))
        .then(pl.lit("enemy"))
        .otherwise(pl.lit("unknown_team")).alias("teamcat"),
        (pl.col("matched") & (pl.col("pick_ts") < FREEZE_S) & (pl.col("round") >= 2)).fill_null(False).alias("pick_early"),
    )
    linked_pids = [assign[d][4] for d in ids]
    gl = gpk.filter(pl.col("pid").is_in(linked_pids)) if linked_pids else gpk.head(0)
    gl = gl.with_columns(pl.col("pid").is_in(linked_pids).alias("linked"))
    A = {
        "pv": pv,
        "gl": gl,
        "drops_prim_matched": per_match(pv.filter(pl.col("matched")).select("midx")),
        "drops_prim_matched_reason": {r: per_match(pv.filter(pl.col("matched") & (pl.col("reason") == r)).select("midx"))
                                      for r in REASONS},
        "drops_prim_teamcat": {t: per_match(pv.filter(pl.col("teamcat") == t).select("midx")) for t in TEAMCATS},
        "picks_prim_ground": per_match(gl.select("midx")),
        "picks_prim_ground_early": per_match(gl.filter(pl.col("early") & (pl.col("round") >= 2)).select("midx")),
        "picks_prim_ground_late": per_match(gl.filter((~pl.col("early")) & (pl.col("round") >= 2)).select("midx")),
        "rounds_ground": per_match(gl.select(["midx", "round"]).unique()),
    }
    return A


log("building per-match arrays")
VAR = {"task": build_variant("task"), "r200": build_variant("r200")}
MAIN = VAR["task"]
STRICT = VAR["r200"]
N_PDR = pdr.height
N_PDR_MATCHED = {k: int(VAR[k]["pv"]["matched"].sum()) for k in VAR}

picks_all = per_match(pk.select("midx"))
picks_ground_all = per_match(pk.filter(pl.col("ground")).select("midx"))
picks_by_cat = {c: per_match(pk.filter(pl.col("cat") == c).select("midx")) for c in CATS}
picks_by_cat_ground = {c: per_match(pk.filter((pl.col("cat") == c) & pl.col("ground")).select("midx")) for c in CATS}
picks_prim = picks_by_cat["primary"]
picks_prim_ground_rule = picks_by_cat_ground["primary"]          # task rule (earlier same-def drop exists)
picks_prim_ground_linked = MAIN["picks_prim_ground"]             # one-to-one linked to a distinct drop
picks_prim_nonground = picks_prim - picks_prim_ground_rule
rounds_ground_rule = per_match(gpk.select(["midx", "round"]).unique())
rounds_ground_linked = MAIN["rounds_ground"]
gpk_early_rule = per_match(gpk.filter(pl.col("early") & (pl.col("round") >= 2)).select("midx"))
gpk_late_rule = per_match(gpk.filter((~pl.col("early")) & (pl.col("round") >= 2)).select("midx"))
drops_all = per_match(dr.select("midx"))
drops_by_reason = {r: per_match(dr.filter(pl.col("reason") == r).select("midx")) for r in REASONS}
drops_by_cat = {c: per_match(dr.filter(pl.col("cat") == c).select("midx")) for c in CATS}
drops_prim = per_match(pdr.select("midx"))
drops_prim_by_reason = {r: per_match(pdr.filter(pl.col("reason") == r).select("midx")) for r in REASONS}
PV_MAIN = MAIN["pv"]
drops_prim_matched_by_reason = MAIN["drops_prim_matched_reason"]
drops_prim_matched = MAIN["drops_prim_matched"]
drops_prim_by_teamcat = MAIN["drops_prim_teamcat"]
rounds_with_prim_pick = per_match(pk.filter(pl.col("cat") == "primary").select(["midx", "round"]).unique())
R2_prim_ground_early = gpk_early_rule
R2_prim_ground_late = gpk_late_rule


# ---------------------------------------------------------------- bootstrap helpers
def boot_index(n_units):
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n_units, size=(NBOOT, n_units))


def ratio_ci(num, den):
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
        cw = np.cumsum(w[m])
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


def add_ratio(name, num, den, definition, sel=None):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if sel is not None:
        num, den = num[sel], den[sel]
    point, lo, hi = ratio_ci(num, den)
    row = {"name": name, "value": point, "ci95_low": lo, "ci95_high": hi,
           "n_num": float(num.sum()), "n_rows": float(den.sum()), "n_matches": int(len(num)),
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


# ---------------------------------------------------------------- (1) counts per round
for name, arr, d in [
    ("drops_all_per_round", drops_all, "all item_dropped rows per clean round"),
    ("drops_death_per_round", drops_by_reason["death"], "item_dropped reason=death per clean round"),
    ("drops_manual_per_round", drops_by_reason["manual"], "item_dropped reason=manual per clean round"),
    ("drops_replaced_per_round", drops_by_reason["replaced"], "item_dropped reason=replaced per clean round"),
    ("drops_primary_per_round", drops_prim, "dropped primary weapons per clean round"),
    ("drops_primary_death_per_round", drops_prim_by_reason["death"], "dropped primary, reason=death, per round"),
    ("drops_primary_manual_per_round", drops_prim_by_reason["manual"], "dropped primary, reason=manual, per round"),
    ("drops_primary_replaced_per_round", drops_prim_by_reason["replaced"], "dropped primary, reason=replaced, per round"),
    ("pickups_all_per_round", picks_all, "item_pickup rows per clean round (buy/spawn + ground)"),
    ("pickups_ground_per_round", picks_ground_all, "task-rule ground item_pickup rows per clean round (all categories)"),
    ("pickups_nonground_per_round", picks_all - picks_ground_all, "non-ground item_pickup rows per clean round"),
    ("pickups_primary_per_round", picks_prim, "primary item_pickup rows per clean round (buy/spawn + ground)"),
    ("pickups_primary_ground_linked_per_round", picks_prim_ground_linked,
     "one-to-one linked ground primary pickups per clean round (conservative)"),
    ("pickups_primary_nonground_per_round", picks_prim_nonground,
     "primary pickups not satisfying the task rule (buy/spawn) per clean round"),
]:
    add_ratio(name, arr, R, d)
for c in CATS:
    if c == "primary":
        continue  # primary covered above
    add_ratio(f"pickups_{c}_per_round", picks_by_cat[c], R, f"item_pickup category={c} per clean round")
    add_ratio(f"pickups_{c}_ground_per_round", picks_by_cat_ground[c], R, f"task-rule ground item_pickup category={c} per round")
    add_ratio(f"drops_{c}_per_round", drops_by_cat[c], R, f"item_dropped category={c} per clean round")
add_ratio("pickups_primary_ground_rule_per_round", picks_prim_ground_rule, R,
          "task-rule ground primary pickups per clean round (upper bound)")

# ---------------------------------------------------------------- (2) share of primary drops picked up
add_ratio("primary_drop_picked_share_all", drops_prim_matched, drops_prim,
          "share of primary drops with a linked ground pickup later in the same round")
for r in REASONS:
    add_ratio(f"primary_drop_picked_share_{r}", drops_prim_matched_by_reason[r], drops_prim_by_reason[r],
              f"share of primary drops with reason={r} picked up later in the round")
add_ratio("primary_drop_picked_share_dropper_died", drops_prim_matched_by_reason["death"],
          drops_prim_by_reason["death"], "dropper died (reason=death)")
add_ratio("primary_drop_picked_share_dropper_alive",
          drops_prim_matched_by_reason["manual"] + drops_prim_matched_by_reason["replaced"],
          drops_prim_by_reason["manual"] + drops_prim_by_reason["replaced"],
          "dropper alive (reason=manual or replaced)")
for t in ["self", "teammate", "enemy"]:
    add_ratio(f"primary_drop_picked_by_{t}_share_of_all_drops", drops_prim_by_teamcat[t], drops_prim,
              f"share of ALL primary drops picked up by a {t}")
for t in ["self", "teammate", "enemy", "unknown_team"]:
    add_ratio(f"picked_primary_drops_by_{t}_share_of_picked", drops_prim_by_teamcat[t], drops_prim_matched,
              f"among picked-up primary drops, share picked by a {t}")

# ---------------------------------------------------------------- (3) time to pickup
mpk = PV_MAIN.filter(pl.col("matched"))
add_median("median_seconds_drop_to_pickup_all", mpk["delta_s"].to_numpy(), mpk["midx"].to_numpy().astype(np.int64),
           mpk.height, "median seconds drop->pickup, all picked primary drops (task rule)")
for r in REASONS:
    sub = mpk.filter(pl.col("reason") == r)
    add_median(f"median_seconds_drop_to_pickup_{r}", sub["delta_s"].to_numpy(),
               sub["midx"].to_numpy().astype(np.int64), sub.height,
               f"median seconds drop->pickup, picked primary drops with reason={r}")
sub = mpk.filter(pl.col("teamcat").is_in(["self", "teammate"]))
add_median("median_seconds_drop_to_pickup_same_team", sub["delta_s"].to_numpy(),
           sub["midx"].to_numpy().astype(np.int64), sub.height, "picked by self or teammate")
sub = mpk.filter(pl.col("teamcat") == "enemy")
add_median("median_seconds_drop_to_pickup_enemy", sub["delta_s"].to_numpy(),
           sub["midx"].to_numpy().astype(np.int64), sub.height, "picked by an enemy")

# ---------------------------------------------------------------- (4) rounds with a ground primary pickup
add_ratio("share_rounds_with_ground_primary_pickup_rule", rounds_ground_rule, R,
          "share of clean rounds with >=1 task-rule ground primary pickup (upper bound)")
add_ratio("share_rounds_with_ground_primary_pickup_linked", rounds_ground_linked, R,
          "share of clean rounds with >=1 one-to-one linked ground primary pickup (conservative)")
add_ratio("mean_ground_primary_pickups_per_round_rule", picks_prim_ground_rule, R,
          "mean task-rule ground primary pickups per clean round (upper bound)")
add_ratio("mean_ground_primary_pickups_per_round_r2plus_rule", picks_prim_ground_rule, R2,
          "mean task-rule ground primary pickups per round, rounds 2+ (upper bound)")
add_ratio("mean_ground_primary_pickups_per_round_r2plus_linked", picks_prim_ground_linked, R2,
          "mean one-to-one linked ground primary pickups per round, rounds 2+ (conservative)")
add_ratio("share_rounds_with_ground_primary_pickup_r2plus", rounds_ground_rule, R2,
          "share of rounds 2+ with >=1 task-rule ground primary pickup")
add_ratio("mean_ground_primary_pickups_per_round_linked", picks_prim_ground_linked, R,
          "mean one-to-one linked ground primary pickups per clean round (conservative)")

# ---------------------------------------------------------------- (5) share of primary acquisitions from ground
add_ratio("share_primary_acquisitions_from_ground_rule", picks_prim_ground_rule, picks_prim,
          "task-rule ground primary pickups / all primary pickups (upper bound)")
add_ratio("share_primary_acquisitions_from_ground_linked", picks_prim_ground_linked, picks_prim,
          "one-to-one linked ground primary pickups / all primary pickups (conservative)")
add_ratio("share_all_pickups_from_ground_rule", picks_ground_all, picks_all,
          "task-rule ground pickups / all item_pickup rows, all categories")
add_ratio("share_rounds_with_any_primary_acquisition", rounds_with_prim_pick, R,
          "share of clean rounds with >=1 primary pickup of any kind")

# ---------------------------------------------------------------- freeze-window split (rounds 2+)
add_ratio("ground_primary_pickups_freeze_window_per_round_r2plus", R2_prim_ground_early, R2,
          "task-rule ground primary pickups with pickup time < 20 s after round start, per round (rounds 2+)")
add_ratio("ground_primary_pickups_after_freeze_per_round_r2plus", R2_prim_ground_late, R2,
          "task-rule ground primary pickups with pickup time >= 20 s after round start, per round (rounds 2+)")
add_ratio("share_ground_primary_in_freeze_window", R2_prim_ground_early,
          R2_prim_ground_early + R2_prim_ground_late, "freeze-window share of ground primary pickups (rounds 2+)")

# ---------------------------------------------------------------- strict variant (r200)
S = STRICT
add_ratio("r200_primary_drop_picked_share_all", S["drops_prim_matched"], drops_prim,
          f"STRICT (link within {STRICT_R:.0f} units): primary drops picked up later in round")
for r in REASONS:
    add_ratio(f"r200_primary_drop_picked_share_{r}", S["drops_prim_matched_reason"][r], drops_prim_by_reason[r],
              f"STRICT: share of primary drops reason={r} picked up later")
for t in ["self", "teammate", "enemy"]:
    add_ratio(f"r200_primary_drop_picked_by_{t}_share_of_all_drops", S["drops_prim_teamcat"][t], drops_prim,
              f"STRICT: share of ALL primary drops picked by a {t}")
add_ratio("r200_share_primary_acquisitions_from_ground", S["picks_prim_ground"], picks_prim,
          "STRICT: ground primary pickups / all primary pickups")
add_ratio("r200_mean_ground_primary_pickups_per_round", S["picks_prim_ground"], R,
          "STRICT: ground primary pickups per clean round")
add_ratio("r200_share_rounds_with_ground_primary_pickup", S["rounds_ground"], R,
          "STRICT: share of clean rounds with >=1 ground primary pickup")
sub = S["pv"].filter(pl.col("matched"))
add_median("r200_median_seconds_drop_to_pickup_all", sub["delta_s"].to_numpy(), sub["midx"].to_numpy().astype(np.int64),
           sub.height, f"STRICT: median seconds drop->pickup (link within {STRICT_R:.0f} units)")
add_ratio("r200_ground_primary_freeze_window_share", S["picks_prim_ground_early"],
          S["picks_prim_ground_early"] + S["picks_prim_ground_late"], "STRICT: freeze-window share of ground primary pickups (rounds 2+)")

# ---------------------------------------------------------------- distance and team splits of task-rule links
pvm = PV_MAIN.filter(pl.col("matched"))
bins = [("<=100", 0, 100), ("100-200", 100, 200), ("200-400", 200, 400), ("400-800", 400, 800), (">800", 800, 1e12)]
DIST_BINS = []
for label, lo_, hi_ in bins:
    n_all = pvm.filter((pl.col("dist") > lo_) & (pl.col("dist") <= hi_)).height
    n_early = pvm.filter((pl.col("dist") > lo_) & (pl.col("dist") <= hi_) & pl.col("pick_early")).height
    DIST_BINS.append({"bin": label, "n_all": n_all, "n_freeze_window_r2plus": n_early})
DIST_BINS.append({"bin": "null position", "n_all": pvm.filter(pl.col("dist").is_null()).height,
                  "n_freeze_window_r2plus": pvm.filter(pl.col("dist").is_null() & pl.col("pick_early")).height})
TEAM_BY_WINDOW = {}
for label, cond in [("freeze_window_r2plus", pl.col("pick_early")),
                    ("after_freeze_or_r1", ~pl.col("pick_early"))]:
    sub = pvm.filter(cond)
    TEAM_BY_WINDOW[label] = {t: int(sub.filter(pl.col("teamcat") == t).height) for t in TEAMCATS}
for rad_name in ["task", "r400", "r200", "r100"]:
    LINKS[rad_name]["n_linked"] = len(LINKS[rad_name]["assign"])
    LINKS[rad_name]["share_of_primary_acq"] = len(LINKS[rad_name]["assign"]) / max(1, int(picks_prim.sum()))
RADIUS_SENS = [{"variant": k, "radius_units": RADII[k], "n_linked": LINKS[k]["n_linked"],
                "share_of_primary_pickups": LINKS[k]["share_of_primary_acq"]} for k in RADII]
log("radius sensitivity: " + "; ".join(f"{d['variant']} linked={d['n_linked']}" for d in RADIUS_SENS))

# ---------------------------------------------------------------- day split (robustness, task rule)
DAY_TABLE = []
for g in ["10/01-07", "10/08"]:
    sel = DGRP == g
    if sel.sum() == 0:
        continue
    for name, num, den, d in [
        ("primary_drop_picked_share_all", drops_prim_matched, drops_prim, "primary drops picked up later in round (linked)"),
        ("share_primary_acquisitions_from_ground_rule", picks_prim_ground_rule, picks_prim,
         "task-rule ground / all primary pickups"),
        ("mean_ground_primary_pickups_per_round_rule", picks_prim_ground_rule, R,
         "task-rule ground primary pickups per round"),
        ("share_rounds_with_ground_primary_pickup_rule", rounds_ground_rule, R,
         "rounds with >=1 task-rule ground primary pickup"),
    ]:
        point, lo, hi = ratio_ci(np.asarray(num)[sel], np.asarray(den)[sel])
        DAY_TABLE.append({"section": "day_split", "group": g, "measure": name, "definition": d,
                          "value": point, "ci95_low": lo, "ci95_high": hi,
                          "n_num": float(np.asarray(num)[sel].sum()), "n_rows": float(np.asarray(den)[sel].sum()),
                          "n_matches": int(sel.sum())})

# ---------------------------------------------------------------- weapon table (task rule)
wt = pdr.join(PV_MAIN.select(["did", "matched"]), on="did", how="left").group_by(["def_index", "item"]).agg(
    pl.len().alias("n_drops"),
    pl.col("matched").sum().alias("n_picked"),
    (pl.col("reason") == "death").sum().alias("n_death_drops")).sort("n_drops", descending=True)
gpk_def = gpk.group_by("def_index").agg(pl.len().alias("n_ground_pickups_all_pickups_rule"))
wt = wt.join(gpk_def, on="def_index", how="left").with_columns(
    pl.col("n_ground_pickups_all_pickups_rule").fill_null(0),
    (pl.col("n_picked") / pl.col("n_drops")).alias("picked_share"))
WEAPON_TABLE = []
for row in wt.iter_rows(named=True):
    WEAPON_TABLE.append({"section": "weapon", "def_index": int(row["def_index"]),
                         "weapon_name_in_drops": row["item"],
                         "n_drops": int(row["n_drops"]), "n_picked": int(row["n_picked"]),
                         "picked_share": float(row["picked_share"]),
                         "n_death_drops": int(row["n_death_drops"]),
                         "n_ground_pickups": int(row["n_ground_pickups_all_pickups_rule"])})

# ---------------------------------------------------------------- validation
dd = dr.filter(pl.col("reason") == "death").with_row_index("k").with_columns(pl.col("k").cast(pl.Int64))
ddj = dd.select(["k", "midx", "round", "player_id", "tick"]).join(
    de.select(["midx", "round", "player_id", pl.col("tick").alias("dtick")]),
    on=["midx", "round", "player_id"], how="left")
ddj = ddj.with_columns((pl.col("tick") - pl.col("dtick")).abs().alias("ad"))
bestd = ddj.group_by("k").agg(pl.col("ad").min())
N_DEATH_DROPS = dd.height
N_DEATH_EXACT = bestd.filter(pl.col("ad") == 0).height
N_DEATH_WITHIN8 = bestd.filter(pl.col("ad") <= 8).height
self_rows = PV_MAIN.filter(pl.col("teamcat") == "self")
N_SELF = self_rows.height
N_SELF_TEAM_EQUAL = self_rows.filter(pl.col("player_team_code") == pl.col("pick_team")).height
pk_r2 = pk.filter(pl.col("round") >= 2)
prim_ng_r2 = pk_r2.filter((pl.col("cat") == "primary") & (~pl.col("ground")))
N_PNG_R2 = prim_ng_r2.height
N_PNG_R2_LE30 = prim_ng_r2.filter(pl.col("ts") <= 30).height
N_PNG_R2_GT60 = prim_ng_r2.filter(pl.col("ts") > 60).height

FREEZE_REASON = {r: int(PV_MAIN.filter(pl.col("pick_early") & (pl.col("reason") == r)).height) for r in REASONS}
CHECKS = {
    "clean_rounds": N_CLEAN_ROUNDS,
    "freeze_window_linked_by_dropper_reason": FREEZE_REASON,
    "clean_matches": M,
    "matches_day_10_01_07": N_MATCH_DAYS["10/01-07"],
    "matches_day_10_08": N_MATCH_DAYS["10/08"],
    "round_end_2026_rows": N_ROUND_END_2026,
    "drops_2026_all": N_DROP_2026_ALL,
    "drops_in_clean_rounds_before_end": N_DROP_IN,
    "drops_after_round_end_excluded": N_DROP_POST,
    "drops_after_round_end_by_reason": POST_DROP_BY_REASON,
    "drops_after_round_end_primary": POST_DROP_PRIMARY,
    "pickups_2026_all": N_PICK_2026_ALL,
    "pickups_in_clean_rounds_before_end": N_PICK_IN,
    "pickups_after_round_end_excluded": N_PICK_POST,
    "pickups_after_round_end_primary": POST_PICK_PRIMARY,
    "pickups_after_round_end_median_seconds": POST_PICK_AFTER_MEDIAN_S,
    "pickups_after_round_end_top_items": POST_PICK_TOP_ITEMS,
    "ground_pickups_all_cats_task_rule": N_GROUND_ALL,
    "ground_primary_pickups_task_rule": N_GROUND_PRIM,
    "ground_primary_pickups_task_rule_linked": len(LINKS["task"]["assign"]),
    "ground_primary_pickups_task_rule_unlinked": len(LINKS["task"]["unassigned"]),
    "ground_primary_pickups_task_rule_unlinked_share": (len(LINKS["task"]["unassigned"]) / max(1, N_GROUND_PRIM)),
    "ground_primary_unlinked_freeze_window_r2plus": int(gpk.filter(
        pl.col("pid").is_in(LINKS["task"]["unassigned"]) & pl.col("early") & (pl.col("round") >= 2)).height),
    "ground_primary_unlinked_after_freeze_r2plus": int(gpk.filter(
        pl.col("pid").is_in(LINKS["task"]["unassigned"]) & (~pl.col("early")) & (pl.col("round") >= 2)).height),
    "primary_drops_round1": int(pdr.filter(pl.col("round") == 1).height),
    "primary_pickups_round1": int(pk.filter((pl.col("round") == 1) & (pl.col("cat") == "primary")).height),
    "ground_primary_pickups_round1": N_GROUND_PRIM_R1,
    "ground_primary_pickups_freeze_window_r2plus": N_GROUND_PRIM_EARLY,
    "ground_primary_pickups_after_freeze_r2plus": N_GROUND_PRIM_LATE,
    "ground_primary_pickups_null_position": N_GROUND_PRIM_NOPOS,
    "primary_drops": N_PDR,
    "primary_drops_picked_task_rule": N_PDR_MATCHED["task"],
    "primary_drops_picked_strict_r200": N_PDR_MATCHED["r200"],
    "linked_by_radius": RADIUS_SENS,
    "distance_bins_task_rule_links": DIST_BINS,
    "team_split_by_window_task_rule_links": TEAM_BY_WINDOW,
    "death_drops": N_DEATH_DROPS,
    "death_drops_with_player_death_same_tick": N_DEATH_EXACT,
    "death_drops_with_player_death_within_8_ticks": N_DEATH_WITHIN8,
    "self_pickups_task_rule": N_SELF,
    "self_pickups_team_equal": N_SELF_TEAM_EQUAL,
    "non_ground_primary_pickups_r2plus": N_PNG_R2,
    "non_ground_primary_pickups_r2plus_within_30s": N_PNG_R2_LE30,
    "non_ground_primary_pickups_r2plus_after_60s": N_PNG_R2_GT60,
}

ASSUMPTIONS = [
    "ASSUMPTION primary = rifles, SMGs and snipers (task definition): rifles 7,8,10,13,16,39,60; snipers 9,11,38,40; SMGs 17,19,23,24,26,33,34. Shotguns (25,27,29,35) and LMGs (14,28) are excluded.",
    "ASSUMPTION item_pickup and item_dropped are joined on def_index, not item text. The text differs between tables (USP-S vs hkp2000, M4A1 vs m4a1, Galil AR vs galilar, SG 553 vs sg556), so a name match would miss pairs.",
    "ASSUMPTION task rule: a pickup is 'ground' if an item_dropped row with the same match, round and def_index exists at a strictly earlier tick. Same-tick drop/pickup pairs (weapon swaps) are not ground.",
    "ASSUMPTION links are one-to-one. Each drop links at most once, to the nearest unconsumed earlier drop of the same def (task variant), or to one within 200 units (strict variant, r200). Positions come from the picker's player_x/y_pos (asof-merged) and the drop's x/y_pos.",
    "ASSUMPTION 'weapon change' = a primary acquisition in a round (item_pickup of a primary: buy/spawn or ground). item_equip (weapon switches) was not used.",
    "ASSUMPTION dropper = item_dropped.player_id and picker = item_pickup.player_id. Raw per-match indices, never compared across matches. Identity is supported by 629k of 629k death drops matching a player_death, and by self-pickups having equal team codes.",
    "ASSUMPTION team = player_team_code on each event row (2=T, 3=CT). Null or 0 codes are unknown_team. Team codes are consistent within every player-round (checked).",
    "ASSUMPTION dropper died = reason 'death'. manual and replaced are treated as dropper alive.",
    "ASSUMPTION events after round_end.tick are excluded. After-end items are next-round spawn equips (knife, glock, hkp2000, c4) logged under the previous round number about 7 s after round_end, plus next-round buys and post-round drops.",
    "ASSUMPTION freeze window = pickup time < 20 s after round_start (header.freeze_time = 20). Round 1 is excluded from the window split because its round_start tick is a placeholder in ~39% of matches.",
    "ASSUMPTION strict link radius 200 CS2 world units (about 5 m). The 100 and 400 unit variants are reported for sensitivity. The distance thresholds are judgement calls, not measured pickup radii.",
    "ASSUMPTION clean rounds = round_end.player_count == 10 (task rule). All 1,752 2026 matches in combined_test are present (days 10/01 to 10/08 only; no days 09 or 10). 1,592 matches have at least one clean round.",
    "ASSUMPTION bootstrap: resample matches with replacement, 500 draws, seed 0 (numpy default_rng). Ratio CIs use the ratio of summed match counts. Median CIs use the weighted median over resampled matches. CIs are 2.5/97.5 percentiles.",
]
CAVEATS = [
    "The task rule admits pickups that are plausibly not physical ground pickups. About 40% of ground primary pickups (rounds 2+) happen inside the 20 s freeze window and link mostly to manual drops. These look like weapon passes during freeze, which are real pickups. The strict distance variant tests how much of this is spurious.",
    "Team and distance checks: 687 of 16,669 linked freeze-window pickups (about 4%) are by an enemy of the dropper. That is implausible physically during freeze, so these are likely purchases of the same def matched to an earlier drop from the other side. After freeze, enemy pickups are 61% of links, which is normal loot-taking. Links beyond 400 units are 1,611 of 57,937 (2.8%), and 714 of 861 links beyond 800 units fall in the freeze window.",
    "Overcount of the task rule: 12,364 of 70,301 rule-positive ground primary pickups (17.6%) have no unconsumed earlier same-def drop left to link to. Of these, 11,198 are in the freeze window (40% of freeze-window rule positives) and 1,166 are after freeze (2.7%). The overcount is therefore concentrated in buy-phase pickups of a def that was dropped earlier. The one-to-one linked counts (57,937) are the conservative figure; the rule counts (70,301) are the upper bound.",
    "Rounds 2+ per-round means are the better base for primary rates, since round 1 contributes zero primary events. Section (4) reports both.",
    "Round 1 is a pistol round in this data: zero primary drops and zero primary pickups across all 1,559 clean round-1 rounds. Per-round means therefore include a zero round 1. Rounds 2+ are the relevant base for primary-weapon rates.",
    "2026-10-08 holds 1,101 of 1,592 matches with clean rounds. The day split above checks whether the headline numbers move.",
    "The '150+ matches' in the task refers to the 1 Hz grid. This analysis uses all 2026 matches in the combined event tome, so n is much larger.",
    "item_pickup includes buy and spawn equips. Non-ground primary pickups are not verified as purchases; in rounds 2+, 98.9% of non-ground primary pickups fall within 30 s of round start, consistent with buys.",
    "The one-sentence conclusion uses primary acquisitions as the denominator (ASSUMPTION 5). Weapon switches (item_equip) were not measured.",
    "Round 1 round_start tick is a placeholder in about 39% of matches (handoff), so round 1 is excluded from freeze-window splits.",
    "Dropped grenades and C4 are in item_dropped but are not primary; they are excluded from primary shares.",
    "Post-round logging: 531k pickups and 54k drops fall after round_end (mostly next-round spawn equips labelled with the previous round). They are excluded by the tick cutoff.",
]

# ---------------------------------------------------------------- output
log("writing JSON")
os.makedirs(VIZ, exist_ok=True)


def clean_num(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if (math.isnan(float(x)) or math.isinf(float(x))) else float(x)
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
] + DAY_TABLE
out = {
    "id": "N6_pickups",
    "version": "v2",
    "script": r"G:\pureskill\csds-cs2\scripts\exp\N6_pickups_analysis_v2.py",
    "source": r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds (2026 era: basic_item_dropped, basic_item_pickup, basic_round_end, basic_round_start, basic_player_death)",
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

print("CHECKS")
for k, v in CHECKS.items():
    print(f"  {k}: {v}")
print("NUMBERS")
for n in NUMBERS:
    fmt = lambda x: "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    print(f"  {n['name']}: {fmt(n['value'])} [{fmt(n['ci95_low'])}, {fmt(n['ci95_high'])}] n_rows={n.get('n_rows')} n_num={n.get('n_num')} matches={n.get('n_matches')}")
print("DAY SPLIT")
for r in DAY_TABLE:
    print(f"  {r['group']} {r['measure']}: {r['value']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] n_num={r['n_num']} n_rows={r['n_rows']} matches={r['n_matches']}")
print("WEAPON TABLE")
for w in WEAPON_TABLE:
    print(f"  def {w['def_index']} {w['weapon_name_in_drops']}: drops={w['n_drops']} picked={w['n_picked']} share={w['picked_share']:.3f} death={w['n_death_drops']} ground={w['n_ground_pickups']}")
