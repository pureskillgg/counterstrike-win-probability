import glob, os, json
import numpy as np
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample"
OUT = r"G:\pureskill\csds-cs2\data\viz\V2N4_final_results.json"
B = 500
SEED = 0
RES = {}

def pages(ch):
    d = os.path.join(BASE, f"basic_{ch}.COMBINED")
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

def load(ch):
    return pl.concat([pl.read_parquet(p) for p in pages(ch)], how="diagonal_relaxed")

def boot_ratio(num, den):
    rng = np.random.default_rng(SEED)
    M = len(num)
    idx = rng.integers(0, M, size=(B, M))
    nb = num[idx].sum(axis=1)
    db = den[idx].sum(axis=1)
    ok = db > 0
    r = nb[ok] / db[ok]
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]

def ci_for(df, num_col, den_col, universe):
    g = universe.join(df.group_by("match_key").agg(pl.col(num_col).cast(pl.Float64).sum().alias("num"),
                                                   pl.col(den_col).cast(pl.Float64).sum().alias("den")),
                      on="match_key", how="left").fill_null(0.0)
    num = g["num"].to_numpy(); den = g["den"].to_numpy()
    est = float(num.sum() / den.sum())
    return round(est, 4), boot_ratio(num, den)

re_ = load("round_end")
re26 = re_.filter(pl.col("match_key").str.starts_with("csds/2026/"))
clean = re26.filter(pl.col("player_count") == 10).select(["match_key", "round"]).unique()
universe = pl.DataFrame({"match_key": sorted(clean["match_key"].unique().to_list())})

ph = load("player_hurt").filter(pl.col("match_key").str.starts_with("csds/2026/"))
ph = ph.join(clean, on=["match_key", "round"], how="inner").with_columns(
    pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("attacker_id").cast(pl.Int64))
ph = ph.sort(["match_key", "round", "player_id", "tick", "health"], descending=[False, False, False, False, True])
ph = ph.with_columns(pl.col("health").shift(1).over(["match_key", "round", "player_id"]).alias("prev_health"))
ph = ph.with_columns(
    pl.when(pl.col("prev_health").is_null()).then(pl.lit(100)).otherwise(pl.col("prev_health")).alias("hp_seq"),
    pl.col("prev_health").is_null().alias("is_first"),
    (pl.col("prev_health") == 0).alias("post_death"))
ph = ph.with_columns((pl.col("hp_seq") == 100).alias("full"), (pl.col("health") == 0).alias("lethal_h0"))
ph = ph.with_columns((pl.col("health_removed") >= pl.col("hp_seq")).alias("one"))

head_all = ph.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"]) & (pl.col("hit_box_code") == 1)).with_row_index("hid")
RES["head_rows_all_clean"] = head_all.height
RES["head_rows_post_death_prev_health_0"] = head_all.filter(pl.col("post_death")).height
RES["head_rows_post_death_by_weapon"] = {w: head_all.filter((pl.col("weapon_name") == w) & pl.col("post_death")).height for w in ["ak47", "m4a1"]}

# team-kill exclusion (the spec that reproduces the claimed numbers)
head = head_all.filter(pl.col("attacker_team_code") != pl.col("player_team_code"))
RES["head_rows_after_TK_and_null_team_filter"] = {w: head.filter(pl.col("weapon_name") == w).height for w in ["ak47", "m4a1"]}
RES["team_kill_head_rows_full_hp"] = {w: head_all.filter((pl.col("weapon_name") == w) & pl.col("full") & (pl.col("attacker_team_code") == pl.col("player_team_code"))).height for w in ["ak47", "m4a1"]}
RES["null_team_head_rows_full_hp"] = {w: head_all.filter((pl.col("weapon_name") == w) & pl.col("full") & pl.col("attacker_team_code").is_null()).height for w in ["ak47", "m4a1"]}

out = {}
for w in ["ak47", "m4a1"]:
    s = head.filter(pl.col("weapon_name") == w).with_columns(pl.lit(1).alias("one_row"), (pl.col("full") & pl.col("one")).alias("strict"))
    sf = s.filter(pl.col("full")).with_columns(pl.lit(1).alias("fone"))
    d = {}
    d["n_head"] = s.height
    d["mean_hr_uncapped"], d["mean_hr_ci"] = ci_for(s, "health_removed", "one_row", universe)
    d["strict_share_all_head"], d["strict_share_ci"] = ci_for(s, "strict", "one_row", universe)
    d["n_full_hp"] = sf.height
    d["one_shot_given_full_hp"], d["one_shot_given_full_hp_ci"] = ci_for(s, "strict", "full", universe)
    d["one_shot_all_head"], d["one_shot_all_head_ci"] = ci_for(s, "one", "one_row", universe)
    # death-row validation for full-HP lethal hits
    out[w] = d
RES["headline_TK_excluded_with_bootstrap"] = out

# death-row validation (full-HP lethal head hits)
pdth = load("player_death").filter(pl.col("match_key").str.starts_with("csds/2026/"))
pdth = pdth.join(clean, on=["match_key", "round"], how="inner").with_columns(
    pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("attacker_id").cast(pl.Int64))
lh = head_all.filter(pl.col("lethal_h0") & pl.col("full")).select(["hid", "match_key", "round", "tick", "player_id", "attacker_id", "weapon_name"])
mj = lh.join(pdth.select(["match_key", "round", "player_id", "attacker_id", "weapon_name", "tick", "is_headshot"]).rename({"tick": "death_tick"}),
             on=["match_key", "round", "player_id", "attacker_id", "weapon_name"], how="left")
mj = mj.with_columns((pl.col("death_tick") - pl.col("tick")).alias("dt")).filter((pl.col("dt") >= 0) & (pl.col("dt") <= 4))
best = mj.sort(["hid", "dt"]).unique(subset=["hid"], keep="first")
RES["full_hp_lethal_head_rows"] = lh.height
RES["full_hp_lethal_head_with_death_row_same_tick_or_later_within_4"] = best.height
RES["full_hp_lethal_head_death_row_share"] = round(best.height / lh.height, 4)
RES["full_hp_lethal_head_death_is_headshot_share"] = round(float(best["is_headshot"].mean()), 4)
unm = lh.join(best.select("hid"), on="hid", how="anti")
RES["full_hp_lethal_unmatched_sample"] = unm.head(6).select(["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name"]).to_dicts()

# all-lethal head rows death match (for context)
lh_all = head_all.filter(pl.col("lethal_h0") & ~pl.col("post_death")).select(["hid", "match_key", "round", "tick", "player_id", "attacker_id", "weapon_name"])
mj2 = lh_all.join(pdth.select(["match_key", "round", "player_id", "attacker_id", "weapon_name", "tick", "is_headshot"]).rename({"tick": "death_tick"}),
                  on=["match_key", "round", "player_id", "attacker_id", "weapon_name"], how="left")
mj2 = mj2.with_columns((pl.col("death_tick") - pl.col("tick")).alias("dt")).filter((pl.col("dt") >= 0) & (pl.col("dt") <= 4))
best2 = mj2.sort(["hid", "dt"]).unique(subset=["hid"], keep="first")
RES["nonpostdeath_lethal_head_rows"] = lh_all.height
RES["nonpostdeath_lethal_head_death_row_share"] = round(best2.height / lh_all.height, 4)

# 20-row check: the first 20 full-HP lethal head rows by match and tick (the ASSUMPTION check)
s20 = lh.sort(["match_key", "tick"]).head(20).join(best.select(["hid", "death_tick", "is_headshot", "dt"]), on="hid", how="left")
RES["check20_full_hp_lethal_matched"] = int(s20["death_tick"].is_not_null().sum())
RES["check20_rows"] = [{"round": int(r["round"]), "tick": int(r["tick"]), "weapon": r["weapon_name"], "death_tick": r["death_tick"], "death_is_headshot": r["is_headshot"]} for r in s20.to_dicts()]

# T-attacker-only AK (note says "On T side an AK")
tk = head.filter(pl.col("weapon_name") == "ak47")
for side, code in [("attacker_T", 2), ("attacker_CT", 3)]:
    s = tk.filter(pl.col("attacker_team_code") == code)
    sf = s.filter(pl.col("full"))
    RES[f"ak47_{side}"] = {"n_head": s.height, "n_full_hp": sf.height,
                           "one_shot_given_full_hp": round(float(sf["one"].mean()), 4) if sf.height else None,
                           "strict_share_all_head": round(float((s["full"] & s["one"]).mean()), 4) if s.height else None}

# post-round / out-of-window check within state sample (rounds.parquet)
state = pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet"))
rounds_s = pl.read_parquet(os.path.join(SAMPLE, "rounds.parquet")).select(["match_key", "round", "live_start_tick", "round_end_tick"]).with_columns(pl.col("round").cast(pl.Int64))
samp_keys = state["match_key"].unique().to_list()
sample_head = head.filter(pl.col("match_key").is_in(samp_keys)).join(rounds_s, on=["match_key", "round"], how="left")
RES["sample_head_rows"] = sample_head.height
RES["sample_head_outside_live_to_end"] = int(sample_head.filter((pl.col("tick") < pl.col("live_start_tick")) | (pl.col("tick") > pl.col("round_end_tick"))).height)
RES["sample_fullhp_head_outside_window"] = int(sample_head.filter(pl.col("full") & ((pl.col("tick") < pl.col("live_start_tick")) | (pl.col("tick") > pl.col("round_end_tick")))).height)

# helmet split with state sample, TK excluded, asof backward, plus bootstrap over the 150-match sample
state = state.with_columns(pl.col("tick").cast(pl.Int64).alias("state_tick"), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
st = state.select(["match_key", "round", "player_id", "state_tick", "has_helmet"]).sort("state_tick")
hits = head.filter(pl.col("match_key").is_in(samp_keys)).with_columns(pl.col("tick").alias("hit_tick")).sort("hit_tick")
j = hits.join_asof(st, left_on="hit_tick", right_on="state_tick", by=["match_key", "round", "player_id"], strategy="backward").filter(pl.col("has_helmet").is_not_null())
univ_s = pl.DataFrame({"match_key": sorted(samp_keys)})
RES["helmet_join_rows_TKexcl"] = j.height
hel = {}
for w in ["ak47", "m4a1"]:
    for hv, lbl in [(True, "helmet"), (False, "no_helmet")]:
        s = j.filter((pl.col("weapon_name") == w) & (pl.col("has_helmet") == hv)).with_columns(
            pl.lit(1).alias("one_row"), (pl.col("full")).cast(pl.Int64).alias("fullc"),
            (pl.col("full") & pl.col("one")).cast(pl.Int64).alias("strict_c"))
        sf = s.filter(pl.col("full"))
        key = f"{w}_{lbl}"
        d = {"n_head": s.height, "n_full_hp": sf.height, "n_full_hp_one_shot": int(sf["one"].sum()) if sf.height else 0}
        if sf.height:
            d["one_shot_given_full_hp"], d["one_shot_given_full_hp_ci"] = ci_for(s, "strict_c", "fullc", univ_s)
        hel[key] = d
RES["helmet_split_TKexcl_asof"] = hel

# full-buy helmet share with bootstrap (helmet at first live second; rounds with team equipment >= 20000)
st0 = state.filter(pl.col("seconds_since_live") == 0)
team = st0.group_by(["match_key", "round", "team_code"]).agg(pl.col("equipment_value_calc").sum().alias("team_eq"), pl.len().alias("n"))
fb = team.filter(pl.col("team_eq") >= 20000).select(["match_key", "round", "team_code"])
s0 = st0.join(fb, on=["match_key", "round", "team_code"], how="inner")
RES["fullbuy_helmet"] = {}
for tc, nm in [(3, "CT"), (2, "T")]:
    s = s0.filter(pl.col("team_code") == tc).with_columns(pl.col("has_helmet").cast(pl.Float64).alias("hh"), pl.lit(1.0).alias("one_row"))
    est, ci = ci_for(s, "hh", "one_row", univ_s)
    RES["fullbuy_helmet"][nm] = {"n_players": s.height, "share_helmet": est, "ci": ci,
                                 "n_no_helmet": int(s.filter(~pl.col("has_helmet")).height)}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(RES, f, indent=1, default=str)
print(json.dumps(RES, indent=1, default=str)[:30000])
print("WROTE", OUT)
