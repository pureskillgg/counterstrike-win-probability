"""E1: economy proxy from item events (G: tomes only).

Question: does a per-round, per-side count of item events before round_freeze_end give a usable
pre-round signal for CT win? Univariate AUC of (CT count minus T count) for ct_win, with a
match-cluster bootstrap 95% CI.

Universe: 2026 canonical tome (days 01 to 08), header is_wingman False (24-round format),
identical-match duplicates removed (keep lowest key, same signature rule as bakeoff.py: round_end
and player_info), rounds with a round_freeze_end, a round_start, and exactly 5 T and 5 CT in
player_info (basic_player_info joined on match_key, round, player_id gives the side).

Windows (per round, tick based):
  pre  = round_start tick <= tick < round_freeze_end tick
  post = round_freeze_end tick <= tick <= buytime_ended tick (sensitivity: purchases after freeze end)

Buy-like rows (equip, pickup): not knife, not c4, not the side's default pistol
  (T glock def 4, CT hkp2000 def 61).

Reads: logs\\swarm\\E1\\cache (built from G: tomes by E1_cache.py). No E: reads.
Writes (logs\\swarm\\E1, data only): e1_results.json, e1_per_round.csv, e1_per_match.csv

Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\E1_economy_proxy.py
"""

import collections
import json
import os
import re
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = r"G:\pureskill\csds-cs2"
OUT = os.path.join(ROOT, "logs", "swarm", "E1")
CACHE = os.path.join(OUT, "cache")
T, CT = 2, 3
DEFAULT_DEF = {T: 4, CT: 61}  # glock, hkp2000 (USP-S)
DEFAULT_ITEM = {T: "glock", CT: "hkp2000"}
N_BOOT = 1000
SEED = 0
KEY_RE = re.compile(r"^csds/2026/10/(\d{2})/([0-9a-f-]{36})/csds$")

t_start = time.perf_counter()


def c(name):
    return pl.read_parquet(os.path.join(CACHE, name + ".parquet"))


def log(*a):
    print("[%6.1fs]" % (time.perf_counter() - t_start), *a, flush=True)


# ---------------- universe ----------------
hdr = c("header").select(["key", "is_wingman", "max_rounds", "match_type", "platform", "match_date"])
hdr = hdr.with_columns(pl.col("key").str.extract(r"csds/2026/10/(\d{2})/", 1).alias("day"))
re_all = c("round_end")
pi_all = c("player_info")
rs_all = c("round_state")
rst_all = c("round_start")
n_round_end_all = re_all.height
log("header", hdr.height, "round_end rows", n_round_end_all)

# signature per match (round_end + player_info), keep lowest key among identical signatures
sig_re = collections.defaultdict(list)
for mk, rd, w, tk in re_all.select(["match_key", "round", "winner_team_code", "tick"]).iter_rows():
    sig_re[mk].append((rd, w, tk))
sig_pi = collections.defaultdict(list)
for mk, rd, pid, tm in pi_all.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    sig_pi[mk].append((rd, pid, tm))
groups = collections.defaultdict(list)
for mk in hdr["key"].to_list():
    sig = (tuple(sorted(sig_re.get(mk, []))), tuple(sorted(sig_pi.get(mk, []))))
    groups[sig].append(mk)
dup_groups = [sorted(g) for g in groups.values() if len(g) > 1]
dup_drop = set()
for g in dup_groups:
    dup_drop.update(g[1:])
n_dup_matches_in_groups = sum(len(g) for g in dup_groups)
log("duplicate groups", len(dup_groups), "matches in groups", n_dup_matches_in_groups,
    "dropped copies", len(dup_drop))

hdr_ok = hdr.filter(~pl.col("key").is_in(list(dup_drop)))
n_match_all = hdr.height
n_match_after_dedupe = hdr_ok.height
hdr_24 = hdr_ok.filter(~pl.col("is_wingman"))
universe_matches = set(hdr_24["key"].to_list())
log("matches all", n_match_all, "after dedupe", n_match_after_dedupe, "24-round non-wingman", len(universe_matches))

# rounds: round_end restricted to universe matches
re_u = re_all.filter(pl.col("match_key").is_in(list(universe_matches)))
n_rounds_u0 = re_u.height
fe = rs_all.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("fe_tick"))
bt = rs_all.filter(pl.col("event_type") == "buytime_ended").group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("bt_tick"))
rst = rst_all.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("rs_tick"))
grid = re_u.select(["match_key", "round", "winner_team_code", "tick", "player_count"]).rename({"tick": "re_tick"})
grid = grid.join(fe, on=["match_key", "round"], how="left").join(bt, on=["match_key", "round"], how="left")
grid = grid.join(rst, on=["match_key", "round"], how="left")
n_has_fe = grid.filter(pl.col("fe_tick").is_not_null()).height
n_has_rs = grid.filter(pl.col("fe_tick").is_not_null() & pl.col("rs_tick").is_not_null()).height
grid = grid.filter(pl.col("fe_tick").is_not_null() & pl.col("rs_tick").is_not_null())

pi_u = pi_all.filter(pl.col("match_key").is_in(list(universe_matches)))
per_round_side = pi_u.group_by(["match_key", "round"]).agg(
    pl.len().alias("n_players"),
    (pl.col("team_code") == T).sum().alias("nT"),
    (pl.col("team_code") == CT).sum().alias("nCT"),
)
grid = grid.join(per_round_side, on=["match_key", "round"], how="left")
grid = grid.filter((pl.col("nT") == 5) & (pl.col("nCT") == 5) & pl.col("winner_team_code").is_in([T, CT]))
grid = grid.with_columns(
    (pl.col("winner_team_code") == CT).cast(pl.Int8).alias("ct_win"),
    pl.col("bt_tick").fill_null(pl.col("fe_tick") + 1281).alias("bt_tick"),
)
n_rounds_final = grid.height
n_matches_final = grid["match_key"].n_unique()
log("rounds: round_end in universe", n_rounds_u0, "with freeze_end+round_start", n_has_rs,
    "final 5v5 rounds", n_rounds_final, "matches", n_matches_final)
grid = grid.join(hdr_24.select([pl.col("key").alias("match_key"), "day"]), on="match_key", how="left")

# side mapping table (player_info restricted to final rounds)
pi_map = pi_u.select(["match_key", "round", "player_id", "team_code"])

# ---------------- item events ----------------
def per_channel(name, equip):
    raw = c(name)
    prof = {"rows_all": raw.height}
    raw = raw.join(grid.select(["match_key", "round"]), on=["match_key", "round"], how="inner")
    prof["rows_in_final_rounds"] = raw.height
    j = raw.join(pi_map, on=["match_key", "round", "player_id"], how="left")
    prof["unmapped_side"] = j.filter(pl.col("team_code").is_null()).height
    j = j.filter(pl.col("team_code").is_not_null())
    prof["mapped_rows"] = j.height
    j = j.join(grid.select(["match_key", "round", "fe_tick", "rs_tick", "bt_tick"]),
               on=["match_key", "round"], how="inner")
    j = j.with_columns(
        ((pl.col("tick") >= pl.col("rs_tick")) & (pl.col("tick") < pl.col("fe_tick"))).alias("pre"),
        ((pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("bt_tick"))).alias("post"),
        (pl.col("tick") < pl.col("rs_tick")).alias("before_rs"),
    )
    if equip:
        knife = pl.col("weapon_type_code") == 0
        c4 = pl.col("item") == "c4"
        default = ((pl.col("team_code") == T) & (pl.col("item") == DEFAULT_ITEM[T])) | (
            (pl.col("team_code") == CT) & (pl.col("item") == DEFAULT_ITEM[CT]))
    else:
        knife = pl.col("def_index") >= 500
        c4 = pl.col("def_index") == 49
        default = ((pl.col("team_code") == T) & (pl.col("def_index") == DEFAULT_DEF[T])) | (
            (pl.col("team_code") == CT) & (pl.col("def_index") == DEFAULT_DEF[CT]))
    j = j.with_columns(
        knife.alias("knife"), c4.alias("c4"), default.alias("default_pistol"),
    ).with_columns(
        (~pl.col("knife") & ~pl.col("c4") & ~pl.col("default_pistol")).alias("buyish"),
    )
    prof["rows_pre"] = j.filter(pl.col("pre")).height
    prof["rows_post_buy_window"] = j.filter(pl.col("post")).height
    prof["rows_before_round_start"] = j.filter(pl.col("before_rs")).height
    prof["knife_rows"] = j.filter(pl.col("knife")).height
    prof["default_pistol_rows"] = j.filter(pl.col("default_pistol")).height
    prof["buyish_rows"] = j.filter(pl.col("buyish")).height
    prof["top_items_buyish_pre"] = (j.filter(pl.col("pre") & pl.col("buyish"))
                                    .group_by("item").len().sort("len", descending=True).head(15)
                                    .to_dicts())
    return j, prof


items = {}
profiles = {}
for name, equip in [("item_equip", True), ("item_pickup", False), ("item_dropped", False), ("item_refund", False)]:
    j, prof = per_channel(name, equip)
    items[name] = j
    profiles[name] = prof
    log(name, {k: v for k, v in prof.items() if not k.startswith("top_")})

# ---------------- per-round side counts ----------------
rounds = grid.select(["match_key", "round", "day", "ct_win", "fe_tick", "rs_tick", "bt_tick", "winner_team_code"])


def side_count(df, mask, name, distinct=False):
    sub = df.filter(mask)
    if distinct:
        sub = sub.select(["match_key", "round", "team_code", "player_id", "item"]).unique()
    g = sub.group_by(["match_key", "round", "team_code"]).len().rename({"len": name})
    out = rounds.select(["match_key", "round"])
    for tm, pref in [(T, "T_"), (CT, "CT_")]:
        x = g.filter(pl.col("team_code") == tm).select(["match_key", "round", name]).rename({name: pref + name})
        out = out.join(x, on=["match_key", "round"], how="left")
    # cast to Int64: polars len() is u32, and CT minus T would wrap around when T > CT
    return out.with_columns(pl.col("T_" + name).fill_null(0).cast(pl.Int64),
                            pl.col("CT_" + name).fill_null(0).cast(pl.Int64))


eq = items["item_equip"]
pk = items["item_pickup"]
dr = items["item_dropped"]
rf = items["item_refund"]

# union of distinct buy-like (player, item) pairs from equip and pickup, before freeze end
un = pl.concat([
    eq.filter(pl.col("pre") & pl.col("buyish")).select(["match_key", "round", "team_code", "player_id", "item"]),
    pk.filter(pl.col("pre") & pl.col("buyish")).select(["match_key", "round", "team_code", "player_id", "item"]),
]).unique()
parts = []
spec = [
    ("eq_all_pre", eq, pl.col("pre"), False),
    ("eq_nonknife_pre", eq, pl.col("pre") & ~pl.col("knife"), False),
    ("eq_buyish_pre_raw", eq, pl.col("pre") & pl.col("buyish"), False),
    ("eq_buyish_pre_distinct", eq, pl.col("pre") & pl.col("buyish"), True),
    ("eq_buyish_post_distinct", eq, pl.col("post") & pl.col("buyish"), True),
    ("eq_buyish_pre_before_rs", eq, pl.col("before_rs"), False),
    ("pk_all_pre", pk, pl.col("pre"), False),
    ("pk_buyish_pre_distinct", pk, pl.col("pre") & pl.col("buyish"), True),
    ("pk_buyish_post_distinct", pk, pl.col("post") & pl.col("buyish"), True),
    ("dr_all_pre", dr, pl.col("pre"), False),
    ("rf_all_pre", rf, pl.col("pre"), False),
    ("rf_all_post", rf, pl.col("post"), False),
    ("eq_pk_union_buyish_pre", un, pl.lit(True), True),
    ("eq_all_round", eq, pl.lit(True), False),
    ("pk_all_round", pk, pl.lit(True), False),
    ("dr_all_round", dr, pl.lit(True), False),
    ("rf_all_round", rf, pl.lit(True), False),
]
for nm, df, mask, distinct in spec:
    x = side_count(df, mask, nm, distinct)
    parts.append(x.select([c_ for c_ in x.columns if c_ not in ("match_key", "round")]))
per_round = rounds.hstack(pl.concat(parts, how="horizontal"))
assert per_round.height == rounds.height
log("per_round rows", per_round.height)

# zero-event shares
def zero_share(col_sum):
    return float((per_round[col_sum] == 0).mean())

pr = per_round.with_columns(
    (pl.col("T_eq_all_round") + pl.col("CT_eq_all_round") + pl.col("T_pk_all_round") + pl.col("CT_pk_all_round")
     + pl.col("T_dr_all_round") + pl.col("CT_dr_all_round") + pl.col("T_rf_all_round") + pl.col("CT_rf_all_round"))
    .alias("all_items_round"),
    (pl.col("T_eq_all_pre") + pl.col("CT_eq_all_pre")).alias("eq_all_pre_round"),
    (pl.col("T_eq_buyish_pre_distinct") + pl.col("CT_eq_buyish_pre_distinct")).alias("eq_buyish_pre_distinct_round"),
    (pl.col("T_eq_buyish_pre_raw") + pl.col("CT_eq_buyish_pre_raw")).alias("eq_buyish_pre_raw_round"),
)
per_round = pr

zero = {
    "rounds": per_round.height,
    "zero_item_events_whole_round_all_channels": int((per_round["all_items_round"] == 0).sum()),
    "zero_item_events_whole_round_all_channels_share": float((per_round["all_items_round"] == 0).mean()),
    "zero_item_equip_pre_freeze_round": int((per_round["eq_all_pre_round"] == 0).sum()),
    "zero_item_equip_pre_freeze_round_share": float((per_round["eq_all_pre_round"] == 0).mean()),
    "zero_buyish_equip_pre_distinct_round": int((per_round["eq_buyish_pre_distinct_round"] == 0).sum()),
    "zero_buyish_equip_pre_distinct_round_share": float((per_round["eq_buyish_pre_distinct_round"] == 0).mean()),
    "zero_T_buyish_pre_distinct_share": float((per_round["T_eq_buyish_pre_distinct"] == 0).mean()),
    "zero_CT_buyish_pre_distinct_share": float((per_round["CT_eq_buyish_pre_distinct"] == 0).mean()),
}
log("zero shares", zero)

# ---------------- AUC with match-cluster bootstrap ----------------
match_codes, match_inv = np.unique(per_round["match_key"].to_numpy(), return_inverse=True)
order = np.argsort(match_inv, kind="stable")
counts = np.bincount(match_inv, minlength=len(match_codes))
starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
y_all = per_round["ct_win"].to_numpy().astype(int)


def auc_pt(x, y):
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, x))


def auc_boot(x, y, rng):
    n = len(match_codes)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([order[starts[p]:starts[p] + counts[p]] for p in pick])
        vals.append(auc_pt(x[idx], y[idx]))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mean_by_side(name):
    return float(per_round["CT_" + name].mean()), float(per_round["T_" + name].mean())


feats = {
    "diff_eq_all_pre (literal: item_equip events before round_freeze_end, CT minus T)": (
        per_round["CT_eq_all_pre"] - per_round["T_eq_all_pre"]),
    "diff_eq_nonknife_pre (item_equip non-knife events before freeze end, CT minus T)": (
        per_round["CT_eq_nonknife_pre"] - per_round["T_eq_nonknife_pre"]),
    "diff_eq_buyish_pre_raw (buy-like equip events, CT minus T)": (
        per_round["CT_eq_buyish_pre_raw"] - per_round["T_eq_buyish_pre_raw"]),
    "diff_eq_buyish_pre_distinct (distinct buy-like items per side, CT minus T) [primary buy proxy]": (
        per_round["CT_eq_buyish_pre_distinct"] - per_round["T_eq_buyish_pre_distinct"]),
    "diff_eq_buyish_post_distinct (buy-like items after freeze end to buytime end, CT minus T) [sensitivity]": (
        per_round["CT_eq_buyish_post_distinct"] - per_round["T_eq_buyish_post_distinct"]),
    "diff_pk_buyish_pre_distinct (buy-like pickups before freeze end, CT minus T) [sensitivity]": (
        per_round["CT_pk_buyish_pre_distinct"] - per_round["T_pk_buyish_pre_distinct"]),
    "diff_eq_buyish_pre_distinct_plus_pk (equip+pickup buy-like distinct, CT minus T) [sensitivity]": (
        (per_round["CT_eq_buyish_pre_distinct"] + per_round["CT_pk_buyish_pre_distinct"])
        - (per_round["T_eq_buyish_pre_distinct"] + per_round["T_pk_buyish_pre_distinct"])),
    "diff_eq_pk_union_buyish_pre (distinct buy-like player-items in equip or pickup, CT minus T) [sensitivity]": (
        per_round["CT_eq_pk_union_buyish_pre"] - per_round["T_eq_pk_union_buyish_pre"]),
}
rng = np.random.default_rng(SEED)
auc_rows = []
for nm, s in feats.items():
    x = s.to_numpy().astype(float)
    pt = auc_pt(x, y_all)
    lo, hi = auc_boot(x, y_all, rng)
    auc_rows.append({"feature": nm, "auc": round(pt, 4), "ci95_match_cluster_boot": [round(lo, 4), round(hi, 4)],
                     "n_rounds": int(len(x)), "n_matches": int(len(match_codes)),
                     "mean_diff": round(float(np.mean(x)), 4), "share_nonzero_diff": round(float(np.mean(x != 0)), 4)})
    log("AUC", nm[:60], round(pt, 4), (round(lo, 4), round(hi, 4)))

# side-only AUCs for the buy-distinct proxy (direction check)
side_only = {}
for nm, col in [("CT_eq_buyish_pre_distinct_alone", "CT_eq_buyish_pre_distinct"),
                ("T_eq_buyish_pre_distinct_alone", "T_eq_buyish_pre_distinct"),
                ("CT_eq_all_pre_alone", "CT_eq_all_pre"), ("T_eq_all_pre_alone", "T_eq_all_pre")]:
    side_only[nm] = round(auc_pt(per_round[col].to_numpy().astype(float), y_all), 4)

# subsets: days 01 to 07 vs 08 (batch effect), by round position
subsets = {}
for lab, mask in [("days_01_07", per_round["day"] != "08"), ("day_08", per_round["day"] == "08")]:
    m = mask.to_numpy()
    x = feats["diff_eq_buyish_pre_distinct (distinct buy-like items per side, CT minus T) [primary buy proxy]"].to_numpy()[m].astype(float)
    yy = y_all[m]
    subsets[lab] = {"n_rounds": int(m.sum()), "auc_primary_buy_proxy": round(auc_pt(x, yy), 4),
                    "ct_win_rate": round(float(yy.mean()), 4)}
x_lit = feats["diff_eq_all_pre (literal: item_equip events before round_freeze_end, CT minus T)"].to_numpy().astype(float)
half = per_round["fe_tick"].to_numpy()
# first half (rounds 1 to 12) vs second half (13 to 24)
rnum = per_round["round"].to_numpy()
for lab, m in [("rounds_1_12", rnum <= 12), ("rounds_13_24", rnum > 12)]:
    subsets[lab] = {"n_rounds": int(m.sum()),
                    "auc_literal_eq_all_pre": round(auc_pt(x_lit[m], y_all[m]), 4),
                    "auc_primary_buy_proxy": round(auc_pt(feats[list(feats)[3]].to_numpy().astype(float)[m], y_all[m]), 4)}

# distribution of per-round counts (for profiling)
def q(s):
    s = s.to_numpy().astype(float)
    return {"mean": round(float(s.mean()), 3), "p50": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)), "p99": float(np.percentile(s, 99)), "max": float(s.max())}

dist = {k: q(per_round[k]) for k in ["eq_all_pre_round", "eq_buyish_pre_distinct_round", "eq_buyish_pre_raw_round",
                                    "all_items_round"]}

# per match totals (events per match and per side)
per_match = (per_round.group_by("match_key").agg(
    pl.len().alias("n_rounds"),
    pl.col("ct_win").mean().alias("ct_win_rate"),
    pl.col("eq_all_pre_round").sum().alias("eq_all_pre"),
    pl.col("all_items_round").sum().alias("all_items"),
    pl.col("T_eq_all_round").sum().alias("T_eq_all_round"),
    pl.col("CT_eq_all_round").sum().alias("CT_eq_all_round"),
    pl.col("eq_buyish_pre_distinct_round").sum().alias("buyish_pre_distinct"),
    pl.col("T_pk_all_round").sum().alias("T_pk_all"),
    pl.col("CT_pk_all_round").sum().alias("CT_pk_all"),
    pl.col("T_dr_all_round").sum().alias("T_dr_all"),
    pl.col("CT_dr_all_round").sum().alias("CT_dr_all"),
    pl.col("T_rf_all_round").sum().alias("T_rf_all"),
    pl.col("CT_rf_all_round").sum().alias("CT_rf_all"),
))
per_match_stats = {}
for col in ["n_rounds", "all_items", "eq_all_pre", "T_eq_all_round", "CT_eq_all_round", "buyish_pre_distinct",
            "T_pk_all", "CT_pk_all", "T_dr_all", "CT_dr_all", "T_rf_all", "CT_rf_all"]:
    s = per_match[col].to_numpy().astype(float)
    per_match_stats[col] = {"mean": round(float(s.mean()), 2), "p50": float(np.percentile(s, 50)),
                            "p05": float(np.percentile(s, 5)), "p95": float(np.percentile(s, 95))}

# ---------------- write outputs ----------------
os.makedirs(OUT, exist_ok=True)
per_round.write_csv(os.path.join(OUT, "e1_per_round.csv"))
per_match.write_csv(os.path.join(OUT, "e1_per_match.csv"))

result = {
    "universe": {
        "header_matches_all": n_match_all,
        "duplicate_groups": len(dup_groups),
        "matches_in_duplicate_groups": n_dup_matches_in_groups,
        "duplicate_copies_dropped": len(dup_drop),
        "matches_after_dedupe": n_match_after_dedupe,
        "matches_24_round_non_wingman": len(universe_matches),
        "round_end_rows_in_universe": n_rounds_u0,
        "rounds_with_freeze_end_and_round_start": n_has_rs,
        "final_5v5_rounds": n_rounds_final,
        "final_matches": n_matches_final,
        "ct_win_rate": round(float(y_all.mean()), 4),
        "days_in_final": per_round["day"].value_counts().sort("day").to_dicts(),
    },
    "channel_profiles": profiles,
    "zero_shares": zero,
    "per_round_distribution": dist,
    "per_match_distribution": per_match_stats,
    "auc_univariate_ct_win": auc_rows,
    "auc_side_only": side_only,
    "subsets": subsets,
    "bootstrap": {"n_boot": N_BOOT, "cluster": "match_key", "seed": SEED},
    "mean_per_round_by_side": {
        "eq_all_pre_CT": round(float(per_round["CT_eq_all_pre"].mean()), 4),
        "eq_all_pre_T": round(float(per_round["T_eq_all_pre"].mean()), 4),
        "eq_buyish_pre_distinct_CT": round(float(per_round["CT_eq_buyish_pre_distinct"].mean()), 4),
        "eq_buyish_pre_distinct_T": round(float(per_round["T_eq_buyish_pre_distinct"].mean()), 4),
        "eq_buyish_post_distinct_CT": round(float(per_round["CT_eq_buyish_post_distinct"].mean()), 4),
        "eq_buyish_post_distinct_T": round(float(per_round["T_eq_buyish_post_distinct"].mean()), 4),
    },
    "runtime_s": round(time.perf_counter() - t_start, 1),
}
with open(os.path.join(OUT, "e1_results.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, default=str)
log("wrote", os.path.join(OUT, "e1_results.json"))
print(json.dumps(result, indent=1, default=str)[:12000])
