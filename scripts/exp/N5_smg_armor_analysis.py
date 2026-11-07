"""N5_smg_armor: armor and helmet purchase by primary-weapon class at the first live second.

Unit: player x round at seconds_since_live == 0 (post-buy state), clean 5v5 rounds only.
Primary class: inv_primary (carried primary item code), mapped to a name with the 2026
weapon_fire tome (dominant name per code) and to a class with the 2026 item_equip
weapon_type_code (game class enum, deterministic per item in the data).
Interval: match-cluster bootstrap, 500 resamples of matches with replacement,
numpy default_rng(0), 95 percent percentile intervals; the same resample draws are used
for every statistic so differences are computed on the same bootstrap samples.
"""
import json
from collections import defaultdict

import numpy as np
import polars as pl

BASE = "G:/pureskill/csds-cs2"
P = f"{BASE}/data/state_sample/players_1hz.parquet"
BUY = f"{BASE}/data/state_sample/round_buy_types.parquet"
TOME = f"{BASE}/tomes/combined_test/tome/csds"
OUT = f"{BASE}/data/viz/N5_smg_armor_results.json"
N_BOOT = 500
SEED = 0
PISTOL_ROUNDS = [1, 13]
CLASSES = ["rifle", "SMG", "sniper", "shotgun/MG", "none"]
SIDES = ["T", "CT", "ALL"]
TYPE_NAME = {0: "knife", 1: "pistol", 2: "SMG", 3: "rifle", 4: "shotgun", 5: "sniper",
             6: "MG", 7: "c4", 8: "taser", 9: "grenade"}
TYPE_TO_CLASS = {2: "SMG", 3: "rifle", 4: "shotgun/MG", 5: "sniper", 6: "shotgun/MG"}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 1. first live second
sec0_all = (pl.scan_parquet(P)
            .filter(pl.col("seconds_since_live") == 0)
            .select(["match_key", "round", "player_id", "team_code", "health", "armor",
                     "has_helmet", "money", "inv_primary"])
            .collect())
log("sec0 player rows:", sec0_all.height,
    "| matches:", sec0_all["match_key"].n_unique(),
    "| rounds:", sec0_all.select(["match_key", "round"]).unique().height)

# ---------------------------------------------------------------- 2. clean 5v5 filter
rstat = (sec0_all.group_by(["match_key", "round"])
         .agg(pl.len().alias("n_players"),
              (pl.col("team_code") == 2).sum().alias("n_t"),
              (pl.col("team_code") == 3).sum().alias("n_ct")))
clean_expr = (pl.col("n_players") == 10) & (pl.col("n_t") == 5) & (pl.col("n_ct") == 5)
bad_rounds = rstat.filter(~clean_expr).sort(["match_key", "round"])
log("rounds not clean 5v5 at sec0:", bad_rounds.height)
for r in bad_rounds.iter_rows(named=True):
    log("   excluded:", r["match_key"], "round", r["round"],
        "players", r["n_players"], "T", r["n_t"], "CT", r["n_ct"])
clean_keys = rstat.filter(clean_expr).select(["match_key", "round"])
rows = sec0_all.join(clean_keys, on=["match_key", "round"], how="inner")
n_rounds_total = rstat.height
n_rounds_clean = clean_keys.height
log("clean rounds:", n_rounds_clean, "of", n_rounds_total, "| clean player-rounds:", rows.height)

rows = rows.with_columns(
    pl.col("round").is_in(PISTOL_ROUNDS).alias("is_pistol_round"),
    pl.when(pl.col("team_code") == 2).then(pl.lit("T")).otherwise(pl.lit("CT")).alias("side"),
)

# cross-check the pistol flag against round_buy_types.is_pistol
buy = pl.read_parquet(BUY).select(["match_key", "round", "is_pistol"])
chk = (rows.select(["match_key", "round", "is_pistol_round"]).unique()
       .join(buy, on=["match_key", "round"], how="left"))
chk_known = chk.filter(pl.col("is_pistol").is_not_null())
pistol_agree = int((chk_known["is_pistol"] == chk_known["is_pistol_round"]).sum())
pistol_known = chk_known.height
pistol_missing_buy = chk.filter(pl.col("is_pistol").is_null()).height
log("pistol flag check (round in {1,13} vs round_buy_types.is_pistol):",
    pistol_agree, "agree of", pistol_known, "| rounds missing from round_buy_types:", pistol_missing_buy)

# ---------------------------------------------------------------- 3. primary code -> name -> class
codes_present = sorted(int(c) for c in rows["inv_primary"].unique().to_list() if int(c) != 0)
log("nonzero inv_primary codes at sec0:", codes_present)

wf = (pl.scan_parquet(f"{TOME}/basic_weapon_fire.COMBINED/dataframe_*")
      .select(["match_key", "player_weapon_code", "weapon_name"])
      .filter(pl.col("match_key").str.starts_with("csds/2026/"))
      .filter(pl.col("player_weapon_code").is_in(codes_present))
      .group_by(["player_weapon_code", "weapon_name"]).agg(pl.len().alias("rows"))
      .collect())
tally = defaultdict(dict)
for code, name, n in wf.iter_rows():
    tally[int(code)][str(name)] = int(n)

ie = (pl.scan_parquet(f"{TOME}/basic_item_equip.COMBINED/dataframe_*")
      .select(["match_key", "item", "weapon_type_code"])
      .filter(pl.col("match_key").str.starts_with("csds/2026/"))
      .group_by(["item", "weapon_type_code"]).agg(pl.len().alias("rows"))
      .collect())
item_type = defaultdict(set)
for it, wt, _n in ie.iter_rows():
    item_type[str(it)].add(int(wt))

code_info = {}
for code in codes_present:
    d = tally.get(code, {})
    if not d:
        code_info[code] = {"name": None, "purity": None, "n_fire": 0, "cls": "unmapped",
                           "source": "no weapon_fire rows"}
        continue
    name, cnt = max(d.items(), key=lambda kv: kv[1])
    tot = sum(d.values())
    base = name.replace("_silencer", "")
    types = item_type.get(base, set())
    cls, src = None, ""
    if len(types) == 1:
        t = next(iter(types))
        cls = TYPE_TO_CLASS.get(t)
        src = f"item_equip weapon_type_code={t} ({TYPE_NAME.get(t, '?')}) for item '{base}'"
    elif len(types) > 1:
        src = f"item_equip item '{base}' has several class codes {sorted(types)}"
    if cls is None and base == "mp5sd" and not types:
        cls = "SMG"
        src = "ASSUMPTION: 'mp5sd' absent from 2026 item_equip; treated as SMG (game category)"
    if cls is None:
        cls = "unmapped"
    code_info[code] = {"name": name, "purity": round(cnt / tot, 4), "n_fire": tot,
                       "cls": cls, "source": src}

log("code map (inv_primary code -> weapon_fire name, purity, class, source):")
for code in codes_present:
    ci = code_info[code]
    log(f"   {code:>3} {str(ci['name']):<16} purity={ci['purity']} n_fire={ci['n_fire']} "
        f"class={ci['cls']} | {ci['source']}")

code_map = pl.DataFrame(
    {"inv_primary": [0] + codes_present,
     "pclass": ["none"] + [code_info[c]["cls"] for c in codes_present]},
    schema={"inv_primary": pl.Int16, "pclass": pl.String})
rows = rows.join(code_map, on="inv_primary", how="left").with_columns(
    pl.col("pclass").fill_null("unmapped"))
log("primary class counts (all clean player-rounds at sec0):")
log(rows.group_by("pclass").agg(pl.len().alias("n")).sort("pclass"))

# ---------------------------------------------------------------- 4. arrays for the bootstrap
match_list = sorted(rows["match_key"].unique().to_list())
midx = {m: i for i, m in enumerate(match_list)}
M = len(match_list)
mi = np.array([midx[m] for m in rows["match_key"].to_list()], dtype=np.int64)
side = np.array(rows["side"].to_list())
pcl = np.array(rows["pclass"].to_list())
rnd = rows["round"].to_numpy()
armor = rows["armor"].cast(pl.Float64).to_numpy()
ge1 = (armor >= 1).astype(np.float64)
hel = rows["has_helmet"].cast(pl.Float64).to_numpy()
money = rows["money"].cast(pl.Float64).to_numpy()
inv = rows["inv_primary"].to_numpy()
is_pist = rows["is_pistol_round"].to_numpy()
N = rows.height
log("matches (clusters):", M, "| player-rounds:", N)

rng = np.random.default_rng(SEED)
draw = rng.integers(0, M, size=(N_BOOT, M))
W = np.zeros((N_BOOT, M), dtype=np.int64)
np.add.at(W, (np.repeat(np.arange(N_BOOT), M), draw.ravel()), 1)


def per_match(mask, values=None):
    if values is None:
        return np.bincount(mi[mask], minlength=M).astype(np.int64)
    return np.bincount(mi[mask], weights=values[mask], minlength=M)


def share_draws(mask):
    nm = per_match(mask)
    nb = W @ nm
    ok = nb > 0
    a = np.full(N_BOOT, np.nan)
    h = np.full(N_BOOT, np.nan)
    a[ok] = (W @ per_match(mask, ge1))[ok] / nb[ok]
    h[ok] = (W @ per_match(mask, hel))[ok] / nb[ok]
    return a, h


def cell(mask):
    n = int(mask.sum())
    if n == 0:
        return {"n_player_rounds": 0, "n_rounds": 0, "n_matches": 0}
    a, h = share_draws(mask)
    keys = set(zip(mi[mask].tolist(), rnd[mask].tolist()))
    return {
        "n_player_rounds": n,
        "n_rounds": len(keys),
        "n_matches": int((per_match(mask) > 0).sum()),
        "armor_ge1_share": float(ge1[mask].mean()),
        "armor_ge1_ci95": [float(x) for x in np.nanpercentile(a, [2.5, 97.5])],
        "helmet_share": float(hel[mask].mean()),
        "helmet_ci95": [float(x) for x in np.nanpercentile(h, [2.5, 97.5])],
        "mean_armor": float(armor[mask].mean()),
        "mean_money_left": float(money[mask].mean()),
        "boot_draws_with_data": int(np.isfinite(a).sum()),
    }


def diff(ma, mb):
    if ma.sum() == 0 or mb.sum() == 0:
        return None
    aa, ha = share_draws(ma)
    ab, hb = share_draws(mb)
    point_a = float(ge1[ma].mean())
    point_b = float(ge1[mb].mean())
    d_arm = aa - ab
    d_hel = ha - hb
    return {
        "n_a": int(ma.sum()), "n_b": int(mb.sum()),
        "armor_a": point_a, "armor_b": point_b,
        "armor_diff_a_minus_b": point_a - point_b,
        "armor_diff_ci95": [float(x) for x in np.nanpercentile(d_arm, [2.5, 97.5])],
        "helmet_diff_a_minus_b": float(hel[ma].mean() - hel[mb].mean()),
        "helmet_diff_ci95": [float(x) for x in np.nanpercentile(d_hel, [2.5, 97.5])],
        "boot_draws_with_data": int(np.isfinite(d_arm).sum()),
    }


SCOPES = {"all_rounds": np.ones(N, dtype=bool), "excl_pistol_rounds": ~is_pist}
SIDE_MASK = {"T": side == "T", "CT": side == "CT", "ALL": np.ones(N, dtype=bool)}

# ---------------------------------------------------------------- 5. tables
tables = {}
for sname, smask in SCOPES.items():
    rows_out = []
    for sd in SIDES:
        for cls in CLASSES + ["unmapped"]:
            m = smask & SIDE_MASK[sd] & (pcl == cls)
            if m.sum() == 0 and cls == "unmapped":
                continue
            c = cell(m)
            c.update({"scope": sname, "side": sd, "class": cls})
            rows_out.append(c)
    tables[sname] = rows_out
    log(f"\n=== table: {sname} ===")
    log(f"{'side':<5}{'class':<12}{'n_pr':>7}{'n_rd':>6}{'n_m':>5}{'armor>=1':>10}"
        f"{'  CI95':>22}{'helmet':>9}{'  CI95':>22}{'m_armor':>9}{'m_money':>9}")
    for c in rows_out:
        if c["n_player_rounds"] == 0:
            log(f"{c['side']:<5}{c['class']:<12}{0:>7}")
            continue
        log(f"{c['side']:<5}{c['class']:<12}{c['n_player_rounds']:>7}{c['n_rounds']:>6}"
            f"{c['n_matches']:>5}{c['armor_ge1_share']:>10.4f}"
            f"  [{c['armor_ge1_ci95'][0]:.4f},{c['armor_ge1_ci95'][1]:.4f}]"
            f"{c['helmet_share']:>9.4f}"
            f"  [{c['helmet_ci95'][0]:.4f},{c['helmet_ci95'][1]:.4f}]"
            f"{c['mean_armor']:>9.2f}{c['mean_money_left']:>9.1f}")

# ---------------------------------------------------------------- 6. SMG vs rifle question
question = {}
for sname, smask in SCOPES.items():
    question[sname] = {}
    for sd in SIDES:
        d = diff(smask & SIDE_MASK[sd] & (pcl == "SMG"), smask & SIDE_MASK[sd] & (pcl == "rifle"))
        question[sname][sd] = d
        if d is None:
            log(f"{sname} {sd}: SMG vs rifle not computable (empty group)")
            continue
        log(f"{sname} {sd}: SMG n={d['n_a']} armor={d['armor_a']:.4f} | rifle n={d['n_b']} "
            f"armor={d['armor_b']:.4f} | diff(SMG-rifle)={d['armor_diff_a_minus_b']:+.4f} "
            f"CI95 [{d['armor_diff_ci95'][0]:+.4f},{d['armor_diff_ci95'][1]:+.4f}] "
            f"| helmet diff={d['helmet_diff_a_minus_b']:+.4f} "
            f"CI95 [{d['helmet_diff_ci95'][0]:+.4f},{d['helmet_diff_ci95'][1]:+.4f}]")

# sensitivity: SMG without mp5sd (inv_primary 23), all rounds, all sides
sens_mask = (pcl == "SMG") & (inv != 23)
sens = diff(sens_mask, pcl == "rifle")
log("sensitivity (all rounds, ALL sides) SMG without mp5sd vs rifle:",
    None if sens is None else
    f"n={sens['n_a']} armor={sens['armor_a']:.4f} diff={sens['armor_diff_a_minus_b']:+.4f} "
    f"CI95 [{sens['armor_diff_ci95'][0]:+.4f},{sens['armor_diff_ci95'][1]:+.4f}]")

# ---------------------------------------------------------------- 7. pistol rounds and overtime check
pist_player_rounds = int(is_pist.sum())
pist_rounds = len(set(zip(mi[is_pist].tolist(), rnd[is_pist].tolist())))
log(f"\npistol rounds (round 1 and 13) in clean set: {pist_rounds} rounds, {pist_player_rounds} player-rounds")
ot = (rows.filter(pl.col("round").is_in([1, 2, 12, 13, 14, 24, 25, 26, 27, 28, 29]))
      .group_by("round")
      .agg(pl.len().alias("player_rounds"),
           pl.col("money").cast(pl.Float64).mean().alias("mean_money_left"),
           (pl.col("armor") >= 1).cast(pl.Float64).mean().alias("armor_ge1_share"))
      .sort("round"))
log("per-round money and armor at sec0 (rounds 1,2,12-14,24-29):")
log(ot)
overtime_check = [{k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v)
                   for k, v in r.items()} for r in ot.iter_rows(named=True)]

# ---------------------------------------------------------------- 8. sanity
sanity = {
    "health_le_0_rows": int(rows.filter(pl.col("health") <= 0).height),
    "helmet_without_armor_rows": int(rows.filter(pl.col("has_helmet") & (pl.col("armor") == 0)).height),
    "money_min": int(rows["money"].min()),
    "money_max": int(rows["money"].max()),
    "armor_min": int(rows["armor"].min()),
    "armor_max": int(rows["armor"].max()),
    "player_rounds_per_side": {k: int((side == k).sum()) for k in ["T", "CT"]},
}
log("\nsanity:", sanity)

mapping_out = []
for code in codes_present:
    ci = code_info[code]
    mapping_out.append({"inv_primary_code": code, "weapon_fire_name": ci["name"],
                        "purity": ci["purity"], "n_fire": ci["n_fire"],
                        "class": ci["cls"], "source": ci["source"]})
mapping_out.insert(0, {"inv_primary_code": 0, "weapon_fire_name": None, "purity": None,
                       "n_fire": None, "class": "none", "source": "inv_primary == 0 (no primary carried)"})

class_counts = {k: int(v) for k, v in rows.group_by("pclass").agg(pl.len().alias("n"))
                .iter_rows()}

result = {
    "id": "N5_smg_armor",
    "script": f"{BASE}/scripts/exp/N5_smg_armor_analysis.py",
    "output_json": OUT,
    "settings": {
        "unit": "player x round at seconds_since_live == 0 (post-buy)",
        "cluster": "match_key",
        "bootstrap": {"method": "match-cluster resampling with replacement", "n": N_BOOT,
                      "seed": SEED, "rng": "numpy default_rng(0)", "interval": "95% percentile"},
        "pistol_rounds": PISTOL_ROUNDS,
        "primary_column": "inv_primary",
    },
    "sample": {
        "matches": M, "rounds_in_state_file": n_rounds_total,
        "rounds_clean_5v5": n_rounds_clean,
        "rounds_excluded": bad_rounds.height,
        "player_rounds_clean": N,
        "pistol_rounds": pist_rounds, "pistol_player_rounds": pist_player_rounds,
        "pistol_flag_agree": pistol_agree, "pistol_flag_checked": pistol_known,
        "pistol_rounds_missing_from_round_buy_types": pistol_missing_buy,
        "class_counts_player_rounds": class_counts,
    },
    "mapping": mapping_out,
    "tables": tables,
    "smg_vs_rifle": question,
    "sensitivity_smg_without_mp5sd": sens,
    "overtime_check": overtime_check,
    "sanity": sanity,
    "assumptions": [
        "ASSUMPTION: unit is the player-round at seconds_since_live == 0; money is post-buy money left at that second.",
        "ASSUMPTION: primary class uses inv_primary (carried primary). weapon_code is the active weapon and can be a knife or pistol, so it is not used.",
        "ASSUMPTION: a round is clean 5v5 if it has 10 players at sec0 with 5 T and 5 CT; 1 round in the state file has 9 players and is excluded.",
        "ASSUMPTION: class per item comes from 2026 item_equip weapon_type_code (2 SMG, 3 rifle, 4 shotgun, 5 sniper, 6 MG). Each item has one code in the data. m4a1_silencer maps to item m4a1 (rifle).",
        "ASSUMPTION: mp5sd is absent from 2026 item_equip; it is treated as SMG (game category). A sensitivity run excludes it.",
        "ASSUMPTION: pistol rounds are round 1 and round 13, as specified. Overtime rounds 25-29 exist in the data and are not flagged as pistol rounds in round_buy_types; they stay in both copies.",
        "ASSUMPTION: side is team_code at sec0 (2 = T, 3 = CT).",
        "ASSUMPTION: bootstrap resamples matches (not rounds or players) with replacement, 500 draws, numpy default_rng(0), 95% percentile intervals. The same draws are reused for every statistic.",
        "ASSUMPTION: all 150 matches in players_1hz are used (train and test splits pooled; the split column is not used here).",
    ],
    "caveats": [
        "Descriptive only: no adjustment for pre-buy budget or round economy. Money at sec0 is after the buy, and the pre-buy amount is not in the file.",
        "The SMG vs rifle comparison is an unadjusted difference in shares. A gap can reflect economy or side, not only preference.",
        "Armor and helmet shares are for the purchase made in the same round, at the first live second.",
    ],
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, default=float)
log("\nwrote", OUT)
