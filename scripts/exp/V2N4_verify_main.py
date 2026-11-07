import glob, os, json
import numpy as np
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample"
OUT = r"G:\pureskill\csds-cs2\data\viz\V2N4_verify_results.json"
B = 500
SEED = 0

R = {"assumptions": [
    "ASSUMPTION A1: clean 5v5 round = 2026 round_end.player_count == 10 (per task text); clean match = match with >=1 clean round (matches the claimed 1,592).",
    "ASSUMPTION A2: head = hit_box_code == 1 (verified by mean health_removed table, see head_code_table).",
    "ASSUMPTION A3: hp before a hit computed two ways: (a) analyst-style health + health_removed; (b) sequential: previous player_hurt.health for same match/round/victim, or 100 for the first hurt row of the victim in the round (sort by tick, then health desc).",
    "ASSUMPTION A4: lethal = health == 0 after hit (health is clamped at 0 while health_removed is uncapped).",
    "ASSUMPTION A5: match-cluster bootstrap over clean matches (universe = 1592 clean matches) with B=500, numpy default_rng(0); 2.5/97.5 percentile CI.",
    "ASSUMPTION A6: 3D distance from victim (player_x/y/z_pos) and attacker (attacker_x/y/z_pos) of player_hurt; null positions dropped.",
    "ASSUMPTION A7: helmet = players_1hz.has_helmet at latest state tick <= hit tick for same match/round/victim player_id (join by raw player_id).",
    "ASSUMPTION A8: full buy = sum of equipment_value_calc over a team at seconds_since_live == 0 >= 20000; team_code 2 = T, 3 = CT.",
]}

def pages(ch):
    d = os.path.join(BASE, f"basic_{ch}.COMBINED")
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

def load(ch):
    return pl.concat([pl.read_parquet(p) for p in pages(ch)], how="diagonal_relaxed")

def boot_ratio(num, den, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    M = len(num)
    idx = rng.integers(0, M, size=(B, M))
    nb = num[idx].sum(axis=1)
    db = den[idx].sum(axis=1)
    ok = db > 0
    r = nb[ok] / db[ok]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]

def point_and_ci(df, num_col, den_col, universe_df):
    g = universe_df.join(
        df.group_by("match_key").agg(pl.col(num_col).cast(pl.Float64).sum().alias("num"),
                                     pl.col(den_col).cast(pl.Float64).sum().alias("den")),
        on="match_key", how="left").fill_null(0.0)
    num = g["num"].to_numpy()
    den = g["den"].to_numpy()
    est = float(num.sum() / den.sum()) if den.sum() > 0 else None
    ci = boot_ratio(num, den) if den.sum() > 0 else [None, None]
    return est, ci

# ---------------------------------------------------------------- 1. clean rounds
re_ = load("round_end")
re26 = re_.filter(pl.col("match_key").str.starts_with("csds/2026/"))
clean = re26.filter(pl.col("player_count") == 10).select(["match_key", "round", "winner_team_code"]).unique(subset=["match_key", "round"])
clean_keys = clean.select(["match_key", "round"])
universe = pl.DataFrame({"match_key": sorted(clean["match_key"].unique().to_list())})
R["n_clean_rounds"] = clean.height
R["n_clean_matches"] = universe.height

# ---------------------------------------------------------------- 2. player_hurt
ph = load("player_hurt").filter(pl.col("match_key").str.starts_with("csds/2026/"))
R["n_player_hurt_2026_all"] = ph.height
ph = ph.join(clean_keys, on=["match_key", "round"], how="inner")
R["n_player_hurt_2026_clean"] = ph.height
ph = ph.with_columns(
    pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64),
    pl.col("player_id").cast(pl.Int64), pl.col("attacker_id").cast(pl.Int64))
ph = ph.sort(["match_key", "round", "player_id", "tick", "health"],
             descending=[False, False, False, False, True])
ph = ph.with_columns(pl.col("health").shift(1).over(["match_key", "round", "player_id"]).alias("prev_health"))
ph = ph.with_columns(
    pl.when(pl.col("prev_health").is_null()).then(pl.lit(100)).otherwise(pl.col("prev_health")).alias("hp_seq"),
    (pl.col("health") + pl.col("health_removed")).alias("hp_a"),
    pl.col("prev_health").is_null().alias("is_first"),
)

# chain checks
mid = ph.filter((~pl.col("is_first")) & (pl.col("prev_health") > 0) & (pl.col("health") > 0))
R["chain_check_nonfirst_nonlethal_n"] = mid.height
R["chain_check_nonfirst_nonlethal_hp_a_equals_prev_health"] = float((mid["hp_a"] == mid["prev_health"]).mean())
firstnl = ph.filter(pl.col("is_first") & (pl.col("health") > 0))
R["chain_check_first_nonlethal_n"] = firstnl.height
R["chain_check_first_nonlethal_hp_a_equals_100"] = float((firstnl["hp_a"] == 100).mean())
R["post_death_hurt_rows_all_weapons_clean"] = ph.filter(pl.col("prev_health") == 0).height
lethal_all = ph.filter(pl.col("health") == 0)
R["lethal_rows_all_weapons_n"] = lethal_all.height
R["lethal_rows_all_weapons_hr_ge_hp_seq_share"] = float((lethal_all["health_removed"] >= lethal_all["hp_seq"]).mean())

# ---------------------------------------------------------------- 3. head hits
head_code_tbl = (ph.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"]))
                   .group_by("hit_box_code").agg(pl.len().alias("n"), pl.col("health_removed").mean().alias("mean_hr"))
                   .sort("hit_box_code"))
R["head_code_table_rifle_ak_m4a1"] = [{"hit_box_code": int(r["hit_box_code"]), "n": int(r["n"]), "mean_health_removed": round(float(r["mean_hr"]), 3)} for r in head_code_tbl.to_dicts()]
R["weapon_name_strings_2026_hurt_containing_m4_or_ak"] = sorted(ph.filter(pl.col("weapon_name").str.contains("m4|ak")).group_by("weapon_name").len()["weapon_name"].to_list())

head = ph.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"]) & (pl.col("hit_box_code") == 1))
head = head.with_row_index("hid").with_columns(
    (pl.col("health_removed") >= pl.col("hp_seq")).alias("one_seq"),
    (pl.col("hp_seq") == 100).alias("full_seq"),
    (pl.col("hp_a") == 100).alias("full_a"),
    (pl.col("health_removed") >= pl.col("hp_a")).alias("one_a"),
    (pl.col("health") == 0).alias("lethal_h0"),
)
R["head_one_seq_equals_health0_share"] = float((head["one_seq"] == head["lethal_h0"]).mean())
R["head_one_a_equals_health0_share"] = float((head["one_a"] == head["lethal_h0"]).mean())

def group_stats(hw, label):
    out = {"label": label}
    n = hw.height
    out["n_head_hits"] = n
    out["n_matches"] = hw["match_key"].n_unique()
    out["mean_health_removed_uncapped"] = round(float(hw["health_removed"].mean()), 3)
    est, ci = point_and_ci(hw.with_columns(pl.lit(1).alias("one")), "health_removed", "one", universe)
    out["mean_health_removed_uncapped_ci"] = [None if ci[0] is None else round(ci[0] / 1.0, 3), None if ci[1] is None else round(ci[1], 3)]
    hw = hw.with_columns(
        pl.lit(1).alias("one"),
        (pl.col("full_seq") & pl.col("one_seq")).alias("strict_seq"),
        (pl.col("full_a") & pl.col("one_a")).alias("strict_a"),
    )
    out["n_full_hp_seq"] = int(hw["full_seq"].sum())
    out["share_full_hp_seq"] = round(float(hw["full_seq"].mean()), 4)
    out["n_strict_seq"] = int(hw["strict_seq"].sum())
    e, c = point_and_ci(hw, "strict_seq", "one", universe)
    out["strict_share_all_head_seq"] = [round(e, 4), [round(c[0], 4), round(c[1], 4)]]
    e, c = point_and_ci(hw, "one_seq", "one", universe)
    out["one_shot_share_all_head_seq"] = [round(e, 4), [round(c[0], 4), round(c[1], 4)]]
    e, c = point_and_ci(hw, "strict_seq", "full_seq", universe)
    out["one_shot_given_full_hp_seq"] = [round(e, 4), [round(c[0], 4), round(c[1], 4)]] if e is not None else None
    out["n_full_hp_a"] = int(hw["full_a"].sum())
    out["n_strict_a"] = int(hw["strict_a"].sum())
    out["strict_share_all_head_analyst"] = round(float(hw["strict_a"].mean()), 4)
    out["one_shot_given_full_hp_analyst"] = round(float(hw["strict_a"].sum() / max(hw["full_a"].sum(), 1)), 4)
    out["share_health0_all_head"] = round(float(hw["lethal_h0"].mean()), 4)
    return out

ak = head.filter(pl.col("weapon_name") == "ak47")
m4 = head.filter(pl.col("weapon_name") == "m4a1")
R["stats_ak47_head"] = group_stats(ak, "ak47 head (hit_box 1)")
R["stats_m4a1_head"] = group_stats(m4, "m4a1 head (hit_box 1)")

# ---------------------------------------------------------------- 4. distance terciles
def add_dist(df):
    return df.with_columns(
        ((pl.col("player_x_pos") - pl.col("attacker_x_pos")) ** 2
         + (pl.col("player_y_pos") - pl.col("attacker_y_pos")) ** 2
         + (pl.col("player_z_pos") - pl.col("attacker_z_pos")) ** 2).sqrt().alias("dist3"))

def tercile_table(hw, label):
    hw = add_dist(hw).filter(pl.col("dist3").is_not_null())
    q1, q2 = hw["dist3"].quantile(1 / 3), hw["dist3"].quantile(2 / 3)
    hw = hw.with_columns(
        pl.when(pl.col("dist3") <= q1).then(pl.lit("T1_near")).when(pl.col("dist3") <= q2).then(pl.lit("T2_mid")).otherwise(pl.lit("T3_far")).alias("tercile"))
    rows = []
    for t in ["T1_near", "T2_mid", "T3_far"]:
        sub = hw.filter(pl.col("tercile") == t)
        full = sub.filter(pl.col("full_seq"))
        rows.append({
            "tercile": t,
            "n_head": sub.height,
            "n_full_hp": full.height,
            "one_shot_given_full_hp": round(float(full["one_seq"].mean()), 4) if full.height else None,
            "one_shot_all_head": round(float(sub["one_seq"].mean()), 4) if sub.height else None,
        })
    return {"label": label, "cut_points_q1_q2": [round(float(q1), 2), round(float(q2), 2)], "rows": rows,
            "n_with_distance": hw.height}

R["distance_terciles_ak47"] = tercile_table(ak, "ak47 head, terciles within AK")
R["distance_terciles_m4a1"] = tercile_table(m4, "m4a1 head, terciles within M4")

# ---------------------------------------------------------------- 5. helmet join (state sample)
state = pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet"))
rounds_s = pl.read_parquet(os.path.join(SAMPLE, "rounds.parquet"))
R["state_sample_matches"] = state["match_key"].n_unique()
R["state_sample_matches_in_clean_universe"] = len(set(state["match_key"].unique().to_list()) & set(universe["match_key"].to_list()))
R["state_sample_match_prefixes"] = sorted(set(k[:9] for k in state["match_key"].unique().to_list()))
state = state.with_columns(
    pl.col("tick").cast(pl.Int64).alias("state_tick"),
    pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
st_sorted = state.select(["match_key", "round", "player_id", "state_tick", "has_helmet", "health", "team_code", "x_pos", "y_pos", "z_pos", "seconds_since_live"]).sort("state_tick")

# restrict head hits to sample matches and check time bounds
rnd = rounds_s.select(["match_key", "round", "live_start_tick", "round_end_tick"]).with_columns(pl.col("round").cast(pl.Int64))
hw_all = head.join(rnd, on=["match_key", "round"], how="left")
R["head_hits_outside_round_live_to_end_window"] = int(hw_all.filter((pl.col("tick") < pl.col("live_start_tick")) | (pl.col("tick") > pl.col("round_end_tick"))).height)
R["head_hits_total_for_window_check"] = hw_all.height
R["head_hits_missing_rounds_row"] = int(hw_all.filter(pl.col("live_start_tick").is_null()).height)

hits = hw_all.filter(pl.col("match_key").is_in(state["match_key"].unique().to_list())).with_columns(pl.col("tick").alias("hit_tick")).sort("hit_tick")
R["head_hits_in_state_sample_matches"] = hits.height
joined = hits.join_asof(st_sorted, left_on="hit_tick", right_on="state_tick",
                        by=["match_key", "round", "player_id"], strategy="backward")
R["head_hits_in_sample_with_state_match"] = int(joined.filter(pl.col("has_helmet").is_not_null()).height)
R["head_hits_in_sample_without_prior_state"] = int(joined.filter(pl.col("has_helmet").is_null()).height)
jj = joined.filter(pl.col("has_helmet").is_not_null())
gap = (jj["hit_tick"] - jj["state_tick"])
R["asof_gap_ticks_max"] = int(gap.max()); R["asof_gap_ticks_median"] = float(gap.median())
# position consistency between tome victim position and state position
dpos = ((jj["player_x_pos"] - jj["x_pos"]) ** 2 + (jj["player_y_pos"] - jj["y_pos"]) ** 2).sqrt()
R["victim_pos_vs_state_pos_dist_median"] = round(float(dpos.drop_nulls().median()), 2)
R["victim_pos_vs_state_pos_dist_p95"] = round(float(dpos.drop_nulls().quantile(0.95)), 2)
R["victim_team_vs_state_team_mismatch"] = int(jj.filter(pl.col("player_team_code") != pl.col("team_code")).height) if "player_team_code" in jj.columns else "n/a"

def helmet_stats(df, label):
    out = {"label": label}
    univ = st_sorted.select("match_key").unique()
    for w, nm in [("ak47", "ak47"), ("m4a1", "m4a1")]:
        sub = df.filter(pl.col("weapon_name") == w).with_columns(pl.lit(1).alias("one"), pl.col("has_helmet").cast(pl.Int64).alias("hel"))
        for hv in [True, False]:
            s2 = sub.filter(pl.col("has_helmet") == hv)
            f2 = s2.filter(pl.col("full_seq"))
            key = f"{nm}_helmet_{'yes' if hv else 'no'}"
            out[key] = {
                "n_head": s2.height,
                "n_full_hp": f2.height,
                "one_shot_given_full_hp": round(float(f2["one_seq"].mean()), 4) if f2.height else None,
                "n_full_hp_one_shot": int(f2["one_seq"].sum()) if f2.height else 0,
                "share_full_hp_of_head": round(float(s2["full_seq"].mean()), 4) if s2.height else None,
                "one_shot_all_head": round(float(s2["one_seq"].mean()), 4) if s2.height else None,
            }
            if f2.height:
                ugrid = univ.join(f2.with_columns(pl.col("one_seq").cast(pl.Float64).alias("num"), pl.lit(1.0).alias("den"))
                                  .group_by("match_key").agg(pl.col("num").sum(), pl.col("den").sum()), on="match_key", how="left").fill_null(0.0)
                numv, denv = ugrid["num"].to_numpy(), ugrid["den"].to_numpy()
                out[key]["one_shot_given_full_hp_ci"] = [round(c, 4) for c in boot_ratio(numv, denv)]
    return out

R["helmet_split_head_hits"] = helmet_stats(jj, "head hits in 150-match state sample, clean rounds, state-joined")

# ---------------------------------------------------------------- 6. death check (20-row ASSUMPTION check)
pdth = load("player_death").filter(pl.col("match_key").str.starts_with("csds/2026/"))
pdth = pdth.join(clean_keys, on=["match_key", "round"], how="inner").with_columns(
    pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("attacker_id").cast(pl.Int64))
R["n_player_death_2026_clean"] = pdth.height
R["player_death_ak47_headshot_share"] = round(float(pdth.filter(pl.col("weapon_name") == "ak47")["is_headshot"].mean()), 4)
R["player_death_m4a1_headshot_share"] = round(float(pdth.filter(pl.col("weapon_name") == "m4a1")["is_headshot"].mean()), 4)

lethal_head = head.filter(pl.col("lethal_h0")).select(["hid", "match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "health_removed", "hp_seq", "hp_a"])
mj = lethal_head.join(pdth.select(["match_key", "round", "player_id", "attacker_id", "weapon_name", "tick", "is_headshot"]).rename({"tick": "death_tick"}),
                      on=["match_key", "round", "player_id", "attacker_id", "weapon_name"], how="left")
mj = mj.with_columns((pl.col("death_tick") - pl.col("tick")).alias("dt")).filter((pl.col("dt") >= 0) & (pl.col("dt") <= 4))
best = mj.sort(["hid", "dt"]).unique(subset=["hid"], keep="first")
R["lethal_head_hits_n"] = lethal_head.height
R["lethal_head_hits_with_death_row_within_0_to_4_ticks"] = best.height
R["lethal_head_hits_with_death_row_share"] = round(best.height / lethal_head.height, 4)
R["lethal_head_death_is_headshot_share"] = round(float(best["is_headshot"].mean()), 4) if best.height else None
R["lethal_head_death_dt_ticks_distribution"] = {str(k): int(v) for k, v in best.group_by("dt").len().sort("dt").select(["dt", "len"]).iter_rows()}

sample20 = lethal_head.sort(["match_key", "tick"]).head(20)
s20 = sample20.join(best.select(["hid", "death_tick", "is_headshot", "dt"]), on="hid", how="left")
R["check_20_rows_lethal_head_vs_next_death"] = [
    {"match_key": r["match_key"][-45:-5] if False else r["match_key"], "round": int(r["round"]), "tick": int(r["tick"]), "weapon": r["weapon_name"],
     "health_removed": int(r["health_removed"]), "hp_before_seq": int(r["hp_seq"]), "hp_before_analyst": int(r["hp_a"]),
     "next_death_tick": None if r["death_tick"] is None else int(r["death_tick"]),
     "death_is_headshot": None if r["is_headshot"] is None else bool(r["is_headshot"]),
     "dt_ticks": None if r["dt"] is None else int(r["dt"])}
    for r in s20.to_dicts()]
R["check_20_rows_all_have_death_row"] = bool(s20["death_tick"].is_not_null().all())

# ---------------------------------------------------------------- 7. full-buy helmet shares
st0 = state.filter(pl.col("seconds_since_live") == 0)
R["state_live0_rows"] = st0.height
team_cnt = st0.group_by(["match_key", "round", "team_code"]).agg(pl.len().alias("n_players"), pl.col("equipment_value_calc").sum().alias("team_eq"))
R["live0_team_cells_with_5_players_share"] = round(float((team_cnt["n_players"] == 5).mean()), 4)
full = team_cnt.filter(pl.col("team_eq") >= 20000).select(["match_key", "round", "team_code"])
R["full_buy_team_cells"] = full.height
st0f = st0.join(full, on=["match_key", "round", "team_code"], how="inner")
fb = {}
for tc, nm in [(3, "CT"), (2, "T")]:
    sub = st0f.filter(pl.col("team_code") == tc)
    fb[nm] = {"n_players": sub.height, "n_rounds_team_cells": int(full.filter(pl.col("team_code") == tc).height),
              "share_has_helmet": round(float(sub["has_helmet"].mean()), 4) if sub.height else None,
              "n_has_helmet": int(sub["has_helmet"].sum())}
    ugrid = universe.join(sub.group_by("match_key").agg(pl.col("has_helmet").cast(pl.Float64).sum().alias("num"), pl.len().cast(pl.Float64).alias("den")), on="match_key", how="left").fill_null(0.0)
    fb[nm]["share_ci_match_bootstrap"] = [round(c, 4) for c in boot_ratio(ugrid["num"].to_numpy(), ugrid["den"].to_numpy())]
R["full_buy_helmet_at_first_live_second"] = fb
R["full_buy_all_teams_helmet_share"] = round(float(st0f["has_helmet"].mean()), 4)
# full-buy restricted to state-sample universe is already the case (state only has sample matches)

# also the claim's full-buy threshold sensitivity (>= 19000 and >= 21000 check)
sens = {}
for thr in [19000, 20000, 21000]:
    f2 = team_cnt.filter(pl.col("team_eq") >= thr).select(["match_key", "round", "team_code"])
    s2 = st0.join(f2, on=["match_key", "round", "team_code"], how="inner")
    sens[str(thr)] = {nm: {"n": int(s2.filter(pl.col("team_code") == tc).height),
                           "share": round(float(s2.filter(pl.col("team_code") == tc)["has_helmet"].mean()), 4) if s2.filter(pl.col("team_code") == tc).height else None}
                      for tc, nm in [(3, "CT"), (2, "T")]}
R["full_buy_threshold_sensitivity"] = sens

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, default=str)
print(json.dumps(R, indent=1, default=str)[:60000])
print("WROTE", OUT)
