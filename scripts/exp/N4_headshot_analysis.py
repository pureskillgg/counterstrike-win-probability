"""N4_headshot analysis: AK-47 / M4 headshots in the 2026 tome, helmet split, distance terciles, full-buy helmet rates.

Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 G:/pureskill/csds-cs2/scripts/exp/N4_headshot_analysis.py
Writes: G:/pureskill/csds-cs2/data/viz/N4_headshot_results.json
"""
import glob
import json
import os

import numpy as np
import polars as pl

TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
STATE = "G:/pureskill/csds-cs2/data/state_sample"
SCRIPT_PATH = "G:/pureskill/csds-cs2/scripts/exp/N4_headshot_analysis.py"
OUT_PATH = "G:/pureskill/csds-cs2/data/viz/N4_headshot_results.json"

B = 500
SEED = 0
MIN_HITS = 30
MIN_MATCHES = 5
FULL_BUY = 20000
SIDE_T, SIDE_CT = 2, 3
HEAD_EXPECTED = 1

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(230)

OUT = {
    "id": "N4_headshot",
    "script": SCRIPT_PATH,
    "output_json": OUT_PATH,
    "numbers": [],
    "table": [],
    "checks": {},
    "validation_rows": [],
    "assumptions": [],
    "caveats": [],
}


def pages(channel):
    folder = os.path.join(TOME, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(folder, "dataframe_*")))


def load26(channel, cols):
    return (
        pl.scan_parquet(pages(channel), missing_columns="insert")
        .filter(pl.col("match_key").str.starts_with("csds/2026/"))
        .select(cols)
        .collect()
    )


def boot_ci(num, den, idx):
    """Match-cluster bootstrap CI (2.5, 97.5 pct) of sum(num)/sum(den); idx has shape (B, M)."""
    nb = num[idx].sum(axis=1)
    db = den[idx].sum(axis=1)
    ratio = np.full(nb.shape, np.nan)
    ok = db > 0
    ratio[ok] = nb[ok] / db[ok]
    lo, hi = np.nanpercentile(ratio, [2.5, 97.5])
    return float(lo), float(hi)


def per_match(df, univ, exprs):
    agg = df.group_by("match_key").agg([e.alias(k) for k, e in exprs.items()])
    out = univ.join(agg, on="match_key", how="left").fill_null(0)
    return {k: out[k].to_numpy().astype(np.float64) for k in exprs}


HIT_EXPRS = {
    "n": pl.len(),
    "hr": pl.col("health_removed").cast(pl.Float64).sum(),
    "capped": pl.col("capped_dmg").cast(pl.Float64).sum(),
    "one": pl.col("one_shot").cast(pl.Int64).sum(),
    "lethal": pl.col("lethal").cast(pl.Int64).sum(),
    "full": pl.col("full_hp").cast(pl.Int64).sum(),
}


def summarize(part, label, df, univ, idx):
    n_hits = df.height
    if n_hits == 0:
        row = {"part": part, "cell": label, "n_hits": 0, "n_matches_with_hits": 0, "too_small": True}
        OUT["table"].append(row)
        return row
    a = per_match(df, univ, HIT_EXPRS)
    n_tot = a["n"].sum()
    mw = int((a["n"] > 0).sum())

    def share(key):
        p = a[key].sum() / n_tot
        lo, hi = boot_ci(a[key], a["n"], idx)
        return [round(float(p), 4), round(lo, 4), round(hi, 4)]

    def cond_share(num, den):
        if a[den].sum() == 0:
            return [None, None, None]
        p = a[num].sum() / a[den].sum()
        lo, hi = boot_ci(a[num], a[den], idx)
        return [round(float(p), 4), round(lo, 4), round(hi, 4)]

    def mean(key):
        p = a[key].sum() / n_tot
        lo, hi = boot_ci(a[key], a["n"], idx)
        return [round(float(p), 2), round(lo, 2), round(hi, 2)]

    row = {
        "part": part,
        "cell": label,
        "n_hits": int(n_tot),
        "n_matches_with_hits": mw,
        "mean_health_removed_ci95": mean("hr"),
        "mean_capped_damage_ci95": mean("capped"),
        "one_shot_strict_share_ci95": share("one"),
        "lethal_share_ci95": share("lethal"),
        "full_hp_share_ci95": share("full"),
        "one_shot_given_full_hp_ci95": cond_share("one", "full"),
        "n_full_hp": int(a["full"].sum()),
        "n_one_shot": int(a["one"].sum()),
        "too_small": bool(n_tot < MIN_HITS or mw < MIN_MATCHES or a["full"].sum() < MIN_HITS),
    }
    OUT["table"].append(row)
    print(f"[{part}] {label}: n={row['n_hits']} matches={mw} one_shot={row['one_shot_strict_share_ci95']} "
          f"lethal={row['lethal_share_ci95']} full_hp={row['full_hp_share_ci95']} "
          f"mean_hr={row['mean_health_removed_ci95']} capped={row['mean_capped_damage_ci95']} small={row['too_small']}")
    return row


def add_number(name, row, key):
    OUT["numbers"].append({
        "name": name,
        "value": row[key],
        "n_rows": row.get("n_hits", row.get("n_players")),
        "n_matches": row.get("n_matches_with_hits", row.get("n_matches_with_players")),
    })


# ---------------------------------------------------------------- A. clean 5v5 rounds (2026)
re_ = load26("round_end", ["match_key", "round", "player_count"]).with_columns(pl.col("round").cast(pl.Int64))
clean_rounds = re_.filter(pl.col("player_count") == 10).select(["match_key", "round"]).unique()
OUT["checks"]["round_end_2026_rows"] = re_.height
OUT["checks"]["round_end_2026_distinct_rounds"] = re_.select(["match_key", "round"]).unique().height
OUT["checks"]["clean_5v5_rounds"] = clean_rounds.height
UNIV_ALL_LIST = sorted(clean_rounds["match_key"].unique().to_list())
UNIV_ALL = pl.DataFrame({"match_key": UNIV_ALL_LIST})
M_ALL = len(UNIV_ALL_LIST)
IDX_ALL = np.random.default_rng(SEED).integers(0, M_ALL, size=(B, M_ALL))
OUT["checks"]["clean_2026_matches_universe"] = M_ALL
print("clean 5v5 rounds:", clean_rounds.height, "matches:", M_ALL)

# ---------------------------------------------------------------- B. hurt rows, weapon strings
HCOLS = ["match_key", "round", "tick", "player_id", "attacker_id", "health", "weapon_name", "health_removed",
         "hit_box_code", "player_team_code", "attacker_team_code", "attacker_weapon_code",
         "player_x_pos", "player_y_pos", "player_z_pos", "attacker_x_pos", "attacker_y_pos", "attacker_z_pos"]
hurt_raw = load26("player_hurt", HCOLS).join(clean_rounds, on=["match_key", "round"], how="semi")
int_cols = ["round", "tick", "player_id", "attacker_id", "health", "health_removed", "hit_box_code",
            "player_team_code", "attacker_team_code", "attacker_weapon_code"]
hurt_raw = hurt_raw.with_columns([pl.col(c).cast(pl.Int64) for c in int_cols])
OUT["checks"]["hurt_rows_clean_all_weapons"] = hurt_raw.height
present = hurt_raw.filter(pl.col("weapon_name").is_in(["ak47", "m4a1", "m4a4", "m4a1_silencer"])) \
    .group_by("weapon_name").agg(pl.len().alias("rows")).sort("weapon_name").to_dicts()
OUT["checks"]["weapon_name_presence_ak_m4"] = present
for nm in ["ak47", "m4a1", "m4a4", "m4a1_silencer"]:
    OUT["checks"][f"hurt_weapon_name_rows[{nm}]"] = int(hurt_raw.filter(pl.col("weapon_name") == nm).height)
print("weapon names (ak/m4 family) present in clean 2026 hurt:", present)

# ---------------------------------------------------------------- C. health chain (previous recorded health per victim per round)
G = ["match_key", "round", "player_id"]
hurt_s = hurt_raw.sort(["match_key", "round", "player_id", "tick", "health"],
                       descending=[False, False, False, False, True])
hurt_s = hurt_s.with_columns(
    pl.col("health").shift(1).over(G).alias("prev_health"),
    (pl.col("health") == 0).alias("lethal"),
)
hurt_s = hurt_s.with_columns(pl.col("lethal").cast(pl.Int64).cum_sum().over(G).alias("cum_lethal"))
hurt_s = hurt_s.with_columns(
    (pl.col("cum_lethal") - pl.col("lethal").cast(pl.Int64)).alias("lethal_before"),
    pl.col("prev_health").fill_null(100).alias("hb_chain"),
    (pl.col("health") + pl.col("health_removed")).alias("hb_spec"),
    pl.col("prev_health").is_null().alias("first_hit"),
)
post = hurt_s.filter(pl.col("lethal_before") > 0)
base = hurt_s.filter(pl.col("lethal_before") == 0)
OUT["checks"]["post_death_rows_dropped"] = post.height
OUT["checks"]["hurt_rows_after_post_death_filter"] = base.height
OUT["checks"]["lethal_rows_all_weapons_clean"] = int(base.filter(pl.col("lethal")).height)

ns = base.filter((~pl.col("lethal")) & (~pl.col("first_hit")))
ns = ns.with_columns((pl.col("health") - (pl.col("prev_health") - pl.col("health_removed"))).alias("delta"))
n_ns = ns.height
OUT["checks"]["chain_nonlethal_with_prev_n"] = n_ns
OUT["checks"]["chain_nonlethal_delta_exact0_share"] = round(ns.filter(pl.col("delta") == 0).height / n_ns, 4)
OUT["checks"]["chain_nonlethal_delta_minus1_share"] = round(ns.filter(pl.col("delta") == -1).height / n_ns, 4)
OUT["checks"]["chain_nonlethal_delta_other_share"] = round(
    ns.filter(~pl.col("delta").is_in([0, -1])).height / n_ns, 4)
fh = base.filter(pl.col("first_hit") & (~pl.col("lethal")))
OUT["checks"]["first_hit_nonlethal_n"] = fh.height
OUT["checks"]["first_hit_spec_hb_eq_100_share"] = round(fh.filter(pl.col("hb_spec") == 100).height / fh.height, 4)
lt_all = base.filter(pl.col("lethal"))
OUT["checks"]["lethal_hr_gt_chain_hb_share"] = round(
    lt_all.filter(pl.col("health_removed") > pl.col("hb_chain")).height / lt_all.height, 4)
OUT["checks"]["lethal_hr_gt_chain_hb_n"] = lt_all.filter(pl.col("health_removed") > pl.col("hb_chain")).height
OUT["checks"]["lethal_all_clean_n"] = lt_all.height
print("post-death rows dropped:", post.height)
print("chain non-lethal delta shares: exact0", OUT["checks"]["chain_nonlethal_delta_exact0_share"],
      "minus1", OUT["checks"]["chain_nonlethal_delta_minus1_share"],
      "other", OUT["checks"]["chain_nonlethal_delta_other_share"])
print("first-hit spec hb==100 share:", OUT["checks"]["first_hit_spec_hb_eq_100_share"])
print("lethal rows with hr > chain hb share:", OUT["checks"]["lethal_hr_gt_chain_hb_share"],
      "n", OUT["checks"]["lethal_hr_gt_chain_hb_n"], "of", lt_all.height)

# ---------------------------------------------------------------- D. enemy hits and head code
enemy = (pl.col("attacker_team_code").is_in([SIDE_T, SIDE_CT])
         & pl.col("player_team_code").is_in([SIDE_T, SIDE_CT])
         & (pl.col("attacker_team_code") != pl.col("player_team_code"))
         & (pl.col("attacker_id") != pl.col("player_id")))
enemy_base = base.filter(enemy)
OUT["checks"]["enemy_hurt_rows_all_weapons"] = enemy_base.height

rif = enemy_base.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"]))
hb_tab = (rif.group_by("hit_box_code")
          .agg(pl.len().alias("rows"), pl.col("health_removed").cast(pl.Float64).mean().alias("mean_health_removed"))
          .sort("hit_box_code"))
print("\nrifle (ak47+m4a1) enemy hits by hit_box_code:")
print(hb_tab)
OUT["checks"]["rifle_hitbox_table"] = [
    {"hit_box_code": int(r["hit_box_code"]), "rows": int(r["rows"]),
     "mean_health_removed": round(float(r["mean_health_removed"]), 3)}
    for r in hb_tab.to_dicts()]
cand = hb_tab.filter(pl.col("rows") >= MIN_HITS).sort("mean_health_removed", descending=True)
head_code = int(cand["hit_box_code"][0])
OUT["checks"]["head_hit_box_code_from_max_mean"] = head_code
assert head_code == HEAD_EXPECTED, f"head code {head_code} != {HEAD_EXPECTED}"

# ---------------------------------------------------------------- E. player_death confirmation
dcols = ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "is_headshot", "attacker_weapon_code"]
deaths = load26("player_death", dcols).join(clean_rounds, on=["match_key", "round"], how="semi")
deaths = deaths.with_columns([pl.col(c).cast(pl.Int64) for c in ["round", "tick", "player_id", "attacker_id", "attacker_weapon_code"]])
deaths_u = deaths.unique(subset=["match_key", "round", "tick", "player_id", "attacker_id"], keep="first")
OUT["checks"]["player_death_clean_rows"] = deaths.height
OUT["checks"]["player_death_unique_key_rows"] = deaths_u.height

lethal_enemy = enemy_base.filter(pl.col("lethal")).select(
    ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "hit_box_code"])
conf = lethal_enemy.join(
    deaths_u.select(["match_key", "round", "tick", "player_id", "attacker_id", "is_headshot"]),
    on=["match_key", "round", "tick", "player_id", "attacker_id"], how="left")
conf = conf.with_columns(pl.col("is_headshot").is_not_null().alias("death_exact_tick_match"))
OUT["checks"]["lethal_enemy_rows"] = conf.height
OUT["checks"]["lethal_enemy_exact_tick_death_match_share"] = round(
    conf.filter(pl.col("death_exact_tick_match")).height / conf.height, 5)
conf_box = (conf.group_by("hit_box_code")
            .agg(pl.len().alias("lethal_rows"),
                 pl.col("death_exact_tick_match").cast(pl.Int64).sum().alias("matched_to_death"),
                 pl.col("is_headshot").cast(pl.Int64).sum().alias("death_is_headshot"))
            .sort("hit_box_code"))
print("\nlethal enemy hits vs player_death exact tick by hit box:")
print(conf_box)
OUT["checks"]["lethal_by_hitbox_vs_death"] = [
    {k: (int(v) if v is not None else None) for k, v in r.items()} for r in conf_box.to_dicts()]
hd = conf.filter(pl.col("hit_box_code") == head_code)
OUT["checks"]["lethal_head_matched_to_death"] = hd.filter(pl.col("death_exact_tick_match")).height
OUT["checks"]["lethal_head_n"] = hd.height
OUT["checks"]["lethal_head_death_is_headshot_share"] = round(
    hd.filter(pl.col("is_headshot") == True).height / max(hd.filter(pl.col("death_exact_tick_match")).height, 1), 4)  # noqa: E712

# ---------------------------------------------------------------- E2. 20-row validation of the health-before assumption
lr = enemy_base.filter(pl.col("lethal") & pl.col("weapon_name").is_in(["ak47", "m4a1"])
                       & (pl.col("hit_box_code") == head_code)).sort(["match_key", "round", "tick", "player_id"])
rng20 = np.random.default_rng(SEED)
pick = np.sort(rng20.choice(lr.height, size=20, replace=False))
samp = (lr.with_row_index("ridx")
        .join(pl.DataFrame({"ridx": pick.astype(np.uint32)}), on="ridx", how="semi")
        .sort("tick"))
dsel = (deaths_u.select(["match_key", "round", "player_id", "tick", "attacker_id", "weapon_name", "is_headshot"])
        .rename({"tick": "d_tick", "attacker_id": "d_attacker", "weapon_name": "d_weapon"}).sort("d_tick"))
val = samp.join_asof(dsel, left_on="tick", right_on="d_tick", by=["match_key", "round", "player_id"],
                     strategy="forward")
val = val.with_columns(
    (pl.col("d_tick") - pl.col("tick")).alias("dt_ticks"),
    (pl.col("d_attacker") == pl.col("attacker_id")).alias("same_attacker"),
    (pl.col("health") + pl.col("health_removed")).alias("hb_spec_formula"),
)
val_rows = []
for r in val.to_dicts():
    val_rows.append({
        "match": r["match_key"].split("/")[3][:8],
        "round": r["round"], "victim": r["player_id"], "attacker": r["attacker_id"],
        "hurt_tick": r["tick"], "death_tick": r["d_tick"], "dt_ticks": r["dt_ticks"],
        "same_attacker": r["same_attacker"], "hurt_weapon": r["weapon_name"], "death_weapon": r["d_weapon"],
        "death_is_headshot": r["is_headshot"], "health_after": r["health"],
        "health_removed": r["health_removed"], "hb_spec_formula": r["hb_spec_formula"],
        "hb_chain": r["hb_chain"],
    })
OUT["validation_rows"] = val_rows
print("\n20-row validation (lethal rifle head hits vs next player_death):")
for r in val_rows:
    print(r)
OUT["checks"]["validation_20_same_attacker"] = int(sum(1 for r in val_rows if r["same_attacker"]))
OUT["checks"]["validation_20_dt_in_0_3"] = int(sum(1 for r in val_rows if r["dt_ticks"] is not None and 0 <= r["dt_ticks"] <= 3))
OUT["checks"]["validation_20_hr_gt_spec_hb"] = int(sum(1 for r in val_rows if r["health_removed"] > r["hb_spec_formula"]))
OUT["checks"]["validation_20_hr_gt_chain_hb"] = int(sum(1 for r in val_rows if r["health_removed"] > r["hb_chain"]))

# ---------------------------------------------------------------- F. head-hit table (rifle head hits, enemy, clean rounds)
H = enemy_base.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"]) & (pl.col("hit_box_code") == head_code)).with_columns(
    pl.when(pl.col("weapon_name") == "ak47").then(pl.lit("AK47")).otherwise(pl.lit("M4")).alias("group"),
    pl.when(pl.col("attacker_weapon_code") == 60).then(pl.lit("M4A1-S"))
      .when(pl.col("attacker_weapon_code") == 16).then(pl.lit("M4A4"))
      .otherwise(pl.lit("other")).alias("variant"),
    pl.col("attacker_team_code").alias("attacker_side"),
    pl.col("player_team_code").alias("victim_side"),
    (pl.col("hb_chain") == 100).alias("full_hp"),
    (pl.col("lethal") & (pl.col("hb_chain") == 100)).alias("one_shot"),
    pl.min_horizontal(pl.col("health_removed"), pl.col("hb_chain")).cast(pl.Float64).alias("capped_dmg"),
    ((pl.col("attacker_x_pos") - pl.col("player_x_pos")).pow(2)
     + (pl.col("attacker_y_pos") - pl.col("player_y_pos")).pow(2)
     + (pl.col("attacker_z_pos") - pl.col("player_z_pos")).pow(2)).sqrt().alias("dist"),
)
OUT["checks"]["head_hits_total"] = H.height
OUT["checks"]["head_hits_group_counts"] = {k: int(v) for k, v in
                                          H.group_by("group").agg(pl.len()).iter_rows()}
OUT["checks"]["head_hits_dist_null"] = int(H.filter(pl.col("dist").is_null()).height)
OUT["checks"]["spec_literal_one_shot_equals_lethal"] = bool(
    H.filter((pl.col("health_removed") >= pl.col("hb_spec")) != pl.col("lethal")).height == 0)
OUT["checks"]["head_hits_attacker_side_counts"] = {str(k): int(v) for k, v in
                                                  H.group_by("attacker_side").agg(pl.len()).iter_rows()}
print("\nhead hits:", H.height, OUT["checks"]["head_hits_group_counts"])

# Part 1: weapon statistics on all clean 2026 matches
print("\n== Part 1: head-hit statistics (all clean 2026 matches)")
p1rows = {}
p1rows["AK47"] = summarize("1", "AK47 (ak47) all", H.filter(pl.col("group") == "AK47"), UNIV_ALL, IDX_ALL)
summarize("1", "AK47 attacker T", H.filter((pl.col("group") == "AK47") & (pl.col("attacker_side") == SIDE_T)), UNIV_ALL, IDX_ALL)
summarize("1", "AK47 attacker CT", H.filter((pl.col("group") == "AK47") & (pl.col("attacker_side") == SIDE_CT)), UNIV_ALL, IDX_ALL)
p1rows["M4"] = summarize("1", "M4 (m4a1 name) all", H.filter(pl.col("group") == "M4"), UNIV_ALL, IDX_ALL)
p1rows["M4A1S"] = summarize("1", "M4A1-S (m4a1 name, attacker_weapon_code 60)",
                            H.filter(pl.col("variant") == "M4A1-S"), UNIV_ALL, IDX_ALL)
p1rows["M4A4"] = summarize("1", "M4A4 (m4a1 name, attacker_weapon_code 16)",
                           H.filter(pl.col("variant") == "M4A4"), UNIV_ALL, IDX_ALL)
add_number("AK47 head one-shot strict share (victim at 100 HP, lethal)", p1rows["AK47"], "one_shot_strict_share_ci95")
add_number("M4 head one-shot strict share (victim at 100 HP, lethal)", p1rows["M4"], "one_shot_strict_share_ci95")
add_number("AK47 head lethal share (spec-literal analogue)", p1rows["AK47"], "lethal_share_ci95")
add_number("M4 head lethal share (spec-literal analogue)", p1rows["M4"], "lethal_share_ci95")
add_number("AK47 head full-HP share (victim at 100 HP before hit)", p1rows["AK47"], "full_hp_share_ci95")
add_number("M4 head full-HP share (victim at 100 HP before hit)", p1rows["M4"], "full_hp_share_ci95")
add_number("AK47 head mean health_removed", p1rows["AK47"], "mean_health_removed_ci95")
add_number("M4 head mean health_removed", p1rows["M4"], "mean_health_removed_ci95")
add_number("AK47 head mean capped damage (min(health_removed, health before))", p1rows["AK47"], "mean_capped_damage_ci95")
add_number("M4 head mean capped damage (min(health_removed, health before))", p1rows["M4"], "mean_capped_damage_ci95")

# attacker-side sanity: AK head hits by attacker side and round half
ak = H.filter(pl.col("group") == "AK47")
OUT["checks"]["ak_head_attacker_victim_side_counts"] = [
    {"attacker_side": int(r[0]), "victim_side": int(r[1]), "rows": int(r[2])}
    for r in ak.group_by(["attacker_side", "victim_side"]).agg(pl.len()).sort(["attacker_side", "victim_side"]).iter_rows()]
ak_ct = ak.filter(pl.col("attacker_side") == SIDE_CT)
OUT["checks"]["ak_head_attacker_ct_rounds_1_12"] = ak_ct.filter(pl.col("round") <= 12).height
OUT["checks"]["ak_head_attacker_ct_rounds_13_24"] = ak_ct.filter(pl.col("round") > 12).height
OUT["checks"]["ak_head_all_rounds_1_12"] = ak.filter(pl.col("round") <= 12).height
OUT["checks"]["ak_head_all_rounds_13_24"] = ak.filter(pl.col("round") > 12).height
print("AK attacker CT by round half:", OUT["checks"]["ak_head_attacker_ct_rounds_1_12"],
      OUT["checks"]["ak_head_attacker_ct_rounds_13_24"], "of", OUT["checks"]["ak_head_all_rounds_1_12"],
      OUT["checks"]["ak_head_all_rounds_13_24"])

# Part 2: helmet split (150-match state sample only)
print("\n== Part 2: helmet split (150-match state sample)")
p1 = pl.read_parquet(os.path.join(STATE, "players_1hz.parquet"),
                     columns=["match_key", "round", "tick", "seconds_since_live", "player_id", "team_code",
                              "has_helmet", "equipment_value_calc"])
p1 = (p1.filter(pl.col("match_key").str.starts_with("csds/2026/") & pl.col("player_id").is_between(0, 9))
      .with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
                    pl.col("team_code").cast(pl.Int64), pl.col("equipment_value_calc").cast(pl.Float64))
      .rename({"tick": "s_tick"}).sort("s_tick"))
OUT["checks"]["players_1hz_rows_player_0_9"] = p1.height
OUT["checks"]["players_1hz_rows_player_id_ge_10_dropped"] = int(
    pl.read_parquet(os.path.join(STATE, "players_1hz.parquet"), columns=["player_id"]).filter(
        pl.col("player_id") >= 10).height)
UNIV_S_LIST = sorted(p1["match_key"].unique().to_list())
UNIV_S = pl.DataFrame({"match_key": UNIV_S_LIST})
M_S = len(UNIV_S_LIST)
IDX_S = np.random.default_rng(SEED).integers(0, M_S, size=(B, M_S))
OUT["checks"]["state_sample_matches"] = M_S

H_s = H.sort("tick")
hs = H_s.join_asof(
    p1.select(["match_key", "round", "player_id", "s_tick", "team_code", "has_helmet"]).sort("s_tick"),
    left_on="tick", right_on="s_tick", by=["match_key", "round", "player_id"], strategy="backward")
hs = hs.with_columns((pl.col("tick") - pl.col("s_tick")).alias("lag_ticks"))
in_sample = H.filter(pl.col("match_key").is_in(UNIV_S_LIST)).height
hs_m = hs.filter(pl.col("has_helmet").is_not_null())
OUT["checks"]["helmet_join_lag_ticks_max_matched"] = int(hs_m["lag_ticks"].max())
OUT["checks"]["helmet_join_matched_lag_gt_64_hits"] = int(hs_m.filter(pl.col("lag_ticks") >= 64).height)
OUT["checks"]["helmet_join_matched_lag_gt_64_share"] = round(
    hs_m.filter(pl.col("lag_ticks") >= 64).height / max(hs_m.height, 1), 4)
hs_s = hs.join(UNIV_S, on="match_key", how="semi")
unm = hs_s.filter(pl.col("has_helmet").is_null())
OUT["checks"]["unmatched_sample_head_hits_no_prior_sample"] = unm.filter(pl.col("s_tick").is_null()).height
OUT["checks"]["unmatched_sample_head_hits_lag_ge_64"] = unm.filter(pl.col("s_tick").is_not_null()).height
OUT["checks"]["unmatched_sample_full_hp_share"] = round(unm.filter(pl.col("full_hp")).height / max(unm.height, 1), 4)
OUT["checks"]["matched_sample_full_hp_share"] = round(hs_m.filter(pl.col("full_hp")).height / max(hs_m.height, 1), 4)
OUT["checks"]["unmatched_sample_lethal_share"] = round(unm.filter(pl.col("lethal")).height / max(unm.height, 1), 4)
OUT["checks"]["matched_sample_lethal_share"] = round(hs_m.filter(pl.col("lethal")).height / max(hs_m.height, 1), 4)
OUT["checks"]["unmatched_sample_group_counts"] = {str(k): int(v) for k, v in
                                                 unm.group_by("group").agg(pl.len()).iter_rows()}
OUT["checks"]["head_hits_in_state_sample_matches"] = in_sample
OUT["checks"]["head_hits_matched_to_helmet_sample"] = hs_m.height
OUT["checks"]["head_hits_unmatched_in_sample"] = in_sample - hs_m.height
OUT["checks"]["victim_team_agrees_sample_team_share"] = round(
    hs_m.filter(pl.col("victim_side") == pl.col("team_code")).height / max(hs_m.height, 1), 5)
print("helmet join: sample head hits", in_sample, "matched", hs_m.height)

for g in ["AK47", "M4"]:
    for hv in [True, False]:
        summarize("2", f"{g} helmet={hv}", hs_m.filter((pl.col("group") == g) & (pl.col("has_helmet") == hv)),
                  UNIV_S, IDX_S)
for vs in [SIDE_T, SIDE_CT]:
    for g in ["AK47", "M4"]:
        for hv in [True, False]:
            summarize("2", f"{g} victim_side={vs} helmet={hv}",
                      hs_m.filter((pl.col("group") == g) & (pl.col("victim_side") == vs)
                                  & (pl.col("has_helmet") == hv)), UNIV_S, IDX_S)
for hv in [True, False]:
    summarize("2", f"AK47+M4 pooled helmet={hv}", hs_m.filter(pl.col("has_helmet") == hv), UNIV_S, IDX_S)

# Part 3: distance terciles (pooled head-hit distances, clean matches)
print("\n== Part 3: distance terciles")
H_d = H.filter(pl.col("dist").is_not_null())
q1, q2 = np.quantile(H_d["dist"].to_numpy(), [1 / 3, 2 / 3])
OUT["checks"]["distance_tercile_cutoffs_game_units"] = [round(float(q1), 2), round(float(q2), 2)]
OUT["checks"]["distance_min_max"] = [round(float(H_d["dist"].min()), 2), round(float(H_d["dist"].max()), 2)]
H_d = H_d.with_columns(
    pl.when(pl.col("dist") <= q1).then(pl.lit(1)).when(pl.col("dist") <= q2).then(pl.lit(2)).otherwise(pl.lit(3))
    .alias("tercile"))
for t in [1, 2, 3]:
    for g in ["AK47", "M4"]:
        summarize("3", f"tercile {t} {g} (clean, all matches)",
                  H_d.filter((pl.col("tercile") == t) & (pl.col("group") == g)), UNIV_ALL, IDX_ALL)
hs_d = hs_m.filter(pl.col("dist").is_not_null())
hs_d = hs_d.with_columns(
    pl.when(pl.col("dist") <= q1).then(pl.lit(1)).when(pl.col("dist") <= q2).then(pl.lit(2)).otherwise(pl.lit(3))
    .alias("tercile"))
for t in [1, 2, 3]:
    for hv in [True, False]:
        summarize("3", f"tercile {t} helmet={hv} (AK+M4 pooled, state sample)",
                  hs_d.filter((pl.col("tercile") == t) & (pl.col("has_helmet") == hv)), UNIV_S, IDX_S)
    for g in ["AK47", "M4"]:
        for hv in [True, False]:
            summarize("3", f"tercile {t} {g} helmet={hv} (state sample)",
                      hs_d.filter((pl.col("tercile") == t) & (pl.col("group") == g) & (pl.col("has_helmet") == hv)),
                      UNIV_S, IDX_S)

# Part 4: first live second helmet rates on full-buy rounds (150-match state sample)
print("\n== Part 4: first-live-second helmet rates")
fl = p1.filter(pl.col("seconds_since_live") == 0)
team = fl.group_by(["match_key", "round", "team_code"]).agg(
    pl.len().alias("n_team"), pl.col("equipment_value_calc").sum().alias("team_equip"))
ok_rounds = (team.filter(pl.col("n_team") == 5).group_by(["match_key", "round"]).agg(pl.len().alias("k"))
             .filter(pl.col("k") == 2).select(["match_key", "round"]))
OUT["checks"]["first_live_rounds_total"] = team.select(["match_key", "round"]).unique().height
OUT["checks"]["first_live_rounds_with_5v5_players"] = ok_rounds.height
fl = fl.join(ok_rounds, on=["match_key", "round"], how="semi")
fl = fl.join(team.select(["match_key", "round", "team_code", "team_equip"]), on=["match_key", "round", "team_code"], how="left")
fl = fl.with_columns((pl.col("team_equip") >= FULL_BUY).alias("full_buy_team"))

rbt = pl.read_parquet(os.path.join(STATE, "round_buy_types.parquet"),
                      columns=["match_key", "round", "ct_equipment", "t_equipment"]) \
    .filter(pl.col("match_key").str.starts_with("csds/2026/")).with_columns(pl.col("round").cast(pl.Int64))
ct_sum = team.filter(pl.col("team_code") == SIDE_CT).select(["match_key", "round", pl.col("team_equip").alias("ct_sum")])
t_sum = team.filter(pl.col("team_code") == SIDE_T).select(["match_key", "round", pl.col("team_equip").alias("t_sum")])
cmp = rbt.join(ct_sum, on=["match_key", "round"], how="inner").join(t_sum, on=["match_key", "round"], how="inner")
OUT["checks"]["round_buy_types_crosscheck_rounds"] = cmp.height
OUT["checks"]["round_buy_types_ct_equipment_exact_match_share"] = round(
    cmp.filter(pl.col("ct_equipment").cast(pl.Float64) == pl.col("ct_sum")).height / max(cmp.height, 1), 4)
OUT["checks"]["round_buy_types_t_equipment_exact_match_share"] = round(
    cmp.filter(pl.col("t_equipment").cast(pl.Float64) == pl.col("t_sum")).height / max(cmp.height, 1), 4)
OUT["checks"]["round_buy_types_full_buy_agree_share"] = round(
    cmp.filter(((pl.col("ct_equipment") >= FULL_BUY) == (pl.col("ct_sum") >= FULL_BUY))
               & ((pl.col("t_equipment") >= FULL_BUY) == (pl.col("t_sum") >= FULL_BUY))).height / max(cmp.height, 1), 4)

P4_EXPRS = {"n": pl.len(), "h": pl.col("has_helmet").cast(pl.Int64).sum()}


def side_arrays(side, full_only):
    d = fl.filter(pl.col("team_code") == side)
    if full_only:
        d = d.filter(pl.col("full_buy_team"))
    a = per_match(d, UNIV_S, P4_EXPRS)
    return a, d


def p4_row(label, side, full_only):
    a, d = side_arrays(side, full_only)
    n_tot = a["n"].sum()
    if n_tot == 0:
        row = {"part": "4", "cell": label, "n_players": 0, "too_small": True}
        OUT["table"].append(row)
        return row
    p = a["h"].sum() / n_tot
    lo, hi = boot_ci(a["h"], a["n"], IDX_S)
    n_rounds = d.select(["match_key", "round"]).unique().height
    row = {"part": "4", "cell": label, "n_players": int(n_tot),
           "n_rounds": int(n_rounds), "n_matches_with_players": int((a["n"] > 0).sum()),
           "helmet_share_ci95": [round(float(p), 4), round(lo, 4), round(hi, 4)],
           "too_small": bool(n_tot < MIN_HITS or n_rounds < MIN_MATCHES)}
    OUT["table"].append(row)
    print(f"[4] {label}: players={row['n_players']} rounds={row['n_rounds']} helmet_share={row['helmet_share_ci95']} small={row['too_small']}")
    return row


ct_full = p4_row("CT players, CT team equip >= 20000 (full buy)", SIDE_CT, True)
t_full = p4_row("T players, T team equip >= 20000 (full buy)", SIDE_T, True)
ct_all = p4_row("CT players, all complete rounds", SIDE_CT, False)
t_all = p4_row("T players, all complete rounds", SIDE_T, False)

# CT minus T difference on full buys with match-cluster bootstrap
a_ct, _ = side_arrays(SIDE_CT, True)
a_t, _ = side_arrays(SIDE_T, True)
ct_draw = a_ct["h"][IDX_S].sum(axis=1) / np.maximum(a_ct["n"][IDX_S].sum(axis=1), 1)
t_draw = a_t["h"][IDX_S].sum(axis=1) / np.maximum(a_t["n"][IDX_S].sum(axis=1), 1)
diff_point = a_ct["h"].sum() / a_ct["n"].sum() - a_t["h"].sum() / a_t["n"].sum()
dlo, dhi = np.percentile(ct_draw - t_draw, [2.5, 97.5])
OUT["checks"]["full_buy_helmet_CT_minus_T_point"] = round(float(diff_point), 4)
OUT["checks"]["full_buy_helmet_CT_minus_T_ci95"] = [round(float(dlo), 4), round(float(dhi), 4)]
print("full-buy helmet CT-T difference:", round(float(diff_point), 4), round(float(dlo), 4), round(float(dhi), 4))
add_number("Full-buy helmet share, CT players (first live second)", ct_full, "helmet_share_ci95")
add_number("Full-buy helmet share, T players (first live second)", t_full, "helmet_share_ci95")
add_number("All complete rounds helmet share, CT players", ct_all, "helmet_share_ci95")
add_number("All complete rounds helmet share, T players", t_all, "helmet_share_ci95")

# has_helmet vs armor at first live second (validation of the helmet flag)
fl_armor = pl.read_parquet(os.path.join(STATE, "players_1hz.parquet"),
                           columns=["match_key", "round", "seconds_since_live", "player_id", "armor", "has_helmet"]) \
    .filter(pl.col("match_key").str.starts_with("csds/2026/") & (pl.col("seconds_since_live") == 0)
            & pl.col("player_id").is_between(0, 9))
OUT["checks"]["first_live_has_helmet_by_armor"] = [
    {"armor_gt_0": bool(r[0]), "has_helmet": bool(r[1]), "players": int(r[2])}
    for r in fl_armor.with_columns((pl.col("armor") > 0).alias("armor_gt_0"))
    .group_by(["armor_gt_0", "has_helmet"]).agg(pl.len()).sort(["armor_gt_0", "has_helmet"]).iter_rows()]
OUT["checks"]["first_live_armor_values"] = [
    {"armor": int(r[0]), "players": int(r[1]), "helmet_true": int(r[2])}
    for r in fl_armor.group_by("armor").agg(pl.len(), pl.col("has_helmet").cast(pl.Int64).sum())
    .sort("armor").iter_rows()]
print("first-live has_helmet by armor>0:", OUT["checks"]["first_live_has_helmet_by_armor"])
print("first-live armor values:", OUT["checks"]["first_live_armor_values"][:8])

# ---------------------------------------------------------------- F2. key numbers from the table
def row_by_cell(name):
    return next(r for r in OUT["table"] if r.get("cell") == name)


add_number("AK47 one-shot given victim at full HP (all clean)", p1rows["AK47"], "one_shot_given_full_hp_ci95")
add_number("M4 one-shot given victim at full HP (all clean)", p1rows["M4"], "one_shot_given_full_hp_ci95")
for g in ["AK47", "M4"]:
    for hv in [True, False]:
        rr = row_by_cell(f"{g} helmet={hv}")
        add_number(f"{g} head hits, victim helmet={hv}: one-shot given full HP (state sample)", rr,
                   "one_shot_given_full_hp_ci95")
        add_number(f"{g} head hits, victim helmet={hv}: one-shot share of all head hits (state sample)", rr,
                   "one_shot_strict_share_ci95")

# ---------------------------------------------------------------- G. assumptions and caveats
OUT["assumptions"] = [
    "ASSUMPTION: clean 5v5 round = round_end.player_count == 10 (2026), per the task note; wingman/partial rounds dropped.",
    "ASSUMPTION: a hit is an enemy hit (attacker and victim on opposite sides, team codes 2/3, not self); friendly fire and world damage excluded from head-hit statistics.",
    "ASSUMPTION: head = hit_box_code 1, taken as the code with the highest mean health_removed among rifle hits (table in checks); it also matches player_death.is_headshot on lethal head hits.",
    "ASSUMPTION: health before a hit = health column of the previous recorded hurt row for the same victim in the same round (sorted by tick, then higher health first); 100 when there is no earlier hurt row in that round.",
    "ASSUMPTION: the spec formula health_before = health + health_removed is NOT used for lethal hits. On lethal hits health is clamped to 0 while health_removed is uncapped, so the formula returns health_before = health_removed and makes every kill look one-shot (checks: lethal_hr_gt_chain_hb_share, validation_20_hr_gt_chain_hb).",
    "ASSUMPTION: rows after a victim's first lethal row in a round are post-death damage and are dropped.",
    "ASSUMPTION: one-shot kill (strict) = lethal hit (health after = 0) on a victim at 100 HP before the hit. The spec's literal test (health_removed >= health before) reduces to 'lethal' for every lethal hit, so it cannot separate one-shot kills from finishing shots; the lethal share is reported as the spec-literal analogue.",
    "ASSUMPTION: M4 group = weapon_name 'm4a1' (hurt table has no 'm4a4' or 'm4a1_silencer' strings). Sub-split by attacker_weapon_code 60 (M4A1-S) and 16 (M4A4); the mapping is consistent with player_death where weapon_name m4a1_silencer has code 60 and m4a1 has code 16 (checks).",
    "ASSUMPTION: helmet = players_1hz.has_helmet from the nearest available 1 Hz sample at or before the hit tick in the same round (no maximum gap; gaps above 64 ticks are counted in checks). Verified: sample ticks = live_start_tick + 64*seconds_since_live exactly.",
    "ASSUMPTION: players_1hz player_id 0..9 equals player_hurt player_id (raw index); player_id 11 (164 rows, one match-round) is excluded.",
    "ASSUMPTION: distance = 3D Euclidean distance between victim and attacker x/y/z on the hurt row, in game units; terciles are pooled over clean head hits and are fixed across bootstrap resamples.",
    "ASSUMPTION: team equipment value = sum of equipment_value_calc over that team's 5 sampled players at the first live second (seconds_since_live == 0); full buy is team-specific (>= 20000). Cross-checked against round_buy_types.",
    "ASSUMPTION: the first recorded hurt in a round starts at 100 HP (no untracked earlier damage).",
    "ASSUMPTION: bootstrap = match-cluster resampling with replacement, 500 draws, numpy default_rng(0); 95% CI = 2.5 and 97.5 percentiles of the draws.",
]
OUT["caveats"] = [
    "health_removed and the health column disagree by exactly 1 HP in about half of hits after a victim's first hit in a round (checks chain_nonlethal_delta_minus1_share); first hits satisfy health + health_removed = 100 exactly (checks first_hit_spec_hb_eq_100_share). The health column is used for health-before; the cause of the 1 HP offset is not established.",
    "Lethal hits have uncapped health_removed (above the health before the hit in about 97% of cases), so the mean health_removed is inflated for kills; mean_capped_damage (min of health_removed and health before) is the comparable quantity.",
    "Universe is the 1592 clean-5v5 matches of the 2026 tome (about 160 of the 1751 matches with hurt rows have no complete 5v5 round and are excluded). The helmet cells use only the 150-match state sample, so helmet-by-tercile and helmet-by-side cells are small; cells flagged too_small are not interpretable.",
    "Of 5648 state-sample head hits, the 324 that happened before the victim's first 1 Hz sample in that round are excluded (checks); their full-HP share (31.8%) and lethal share (85.2%) are close to the matched hits (32.8%, 83.6%).",
    "AK head hits by CT attackers are 23% of AK head hits (10041 of 44064) and their share is similar across round halves (24% in rounds 1-12, 22% in 13-24), so this is not a halftime team-code artifact. weapon_name is taken as the attacker's weapon at the hit.",
    "About 6% of AK head hits on a full-HP victim were not lethal (one-shot given full HP is about 92%); helmet and distance cuts do not explain the gap in this data, and the cause is not identified.",
    "Positions are asof-merged from nearby ticks in the tome, so distance is approximate.",
    "Only the 2026 tome era is used; the 2024 rows are excluded. The per-tick grid and prediction files (per_tick_states.parquet, per_tick_test_preds.parquet) are not needed for these four tasks and are not used.",
    "Results describe this 2026-10-01..08 batch and do not establish a causal effect of helmets or weapon choice; the helmet motive in the note (CTs forgo helmets because Ts' rifles kill through them) is not testable from these data.",
]

with open(OUT_PATH, "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\nwrote", OUT_PATH)
print("n table rows:", len(OUT["table"]), " numbers:", len(OUT["numbers"]))
