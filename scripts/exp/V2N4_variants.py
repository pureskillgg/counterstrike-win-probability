import glob, os
import numpy as np
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample"

def pages(ch):
    d = os.path.join(BASE, f"basic_{ch}.COMBINED")
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

def load(ch):
    return pl.concat([pl.read_parquet(p) for p in pages(ch)], how="diagonal_relaxed")

re_ = load("round_end")
re26 = re_.filter(pl.col("match_key").str.starts_with("csds/2026/"))
clean = re26.filter(pl.col("player_count") == 10).select(["match_key", "round"]).unique()
ph = load("player_hurt").filter(pl.col("match_key").str.starts_with("csds/2026/"))
ph = ph.join(clean, on=["match_key", "round"], how="inner").with_columns(
    pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("attacker_id").cast(pl.Int64))
ph = ph.with_row_index("rid")
ph = ph.sort(["match_key", "round", "player_id", "tick", "health"], descending=[False, False, False, False, True])
ph = ph.with_columns(pl.col("health").shift(1).over(["match_key", "round", "player_id"]).alias("prev_health"))
ph = ph.with_columns(
    pl.when(pl.col("prev_health").is_null()).then(pl.lit(100)).otherwise(pl.col("prev_health")).alias("hp_seq"),
    pl.col("prev_health").is_null().alias("is_first_any"))
ph = ph.with_columns((pl.col("hp_seq") == 100).alias("full_any"))

rif = ph.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"]) & (pl.col("hit_box_code") == 1))
rif = rif.with_columns((pl.col("health_removed") >= pl.col("hp_seq")).alias("one_seq"), (pl.col("health") == 0).alias("lethal_h0"))
# variant B: first HEAD hit per victim-round (ignores body damage before it)
rif = rif.sort(["match_key", "round", "player_id", "tick"]).with_columns(
    pl.int_range(pl.len()).over(["match_key", "round", "player_id"]).alias("head_seq_idx"))
rif = rif.with_columns((pl.col("head_seq_idx") == 0).alias("full_firsthead"))

def rate(df, num, den):
    n = df.filter(den).height
    k = df.filter(den & num).height
    return n, (round(k / n, 4) if n else None)

print("== A: full HP = first hurt (any weapon) of victim in round (our definition)")
for w in ["ak47", "m4a1"]:
    s = rif.filter(pl.col("weapon_name") == w)
    print(w, "n_fullHP", s.filter(pl.col("full_any")).height, "one_shot|fullHP", rate(s, pl.col("one_seq"), pl.col("full_any")), "lethal_h0|fullHP", rate(s, pl.col("lethal_h0"), pl.col("full_any")))
print("== B: full HP = first HEAD hit in victim-round (ignores body damage before it)")
for w in ["ak47", "m4a1"]:
    s = rif.filter(pl.col("weapon_name") == w)
    print(w, "n_firsthead", s.filter(pl.col("full_firsthead")).height, "one_shot|firsthead", rate(s, pl.col("one_seq"), pl.col("full_firsthead")))
print("== D: exclude team kills (attacker_team_code == player_team_code) from A")
for w in ["ak47", "m4a1"]:
    s = rif.filter(pl.col("weapon_name") == w).filter(pl.col("attacker_team_code") != pl.col("player_team_code"))
    print(w, "n_fullHP_noTK", s.filter(pl.col("full_any")).height, "one_shot|fullHP", rate(s, pl.col("one_seq"), pl.col("full_any")))
ak_all = rif.filter(pl.col("weapon_name") == "ak47")
print("AK all head rows team-kill count:", ak_all.filter(pl.col("attacker_team_code") == pl.col("player_team_code")).height)

print("== E: distance terciles, AK full-HP: 3D vs 2D; cut points from full-HP subset vs all head")
def dist(df, two_d):
    if two_d:
        return ((pl.col("player_x_pos") - pl.col("attacker_x_pos")) ** 2 + (pl.col("player_y_pos") - pl.col("attacker_y_pos")) ** 2).sqrt()
    return ((pl.col("player_x_pos") - pl.col("attacker_x_pos")) ** 2 + (pl.col("player_y_pos") - pl.col("attacker_y_pos")) ** 2 + (pl.col("player_z_pos") - pl.col("attacker_z_pos")) ** 2).sqrt()

for two_d in [False, True]:
    s = ak_all.with_columns(dist(None, two_d).alias("d")).filter(pl.col("d").is_not_null())
    fsub = s.filter(pl.col("full_any"))
    q = fsub["d"].quantile(1 / 3), fsub["d"].quantile(2 / 3)
    qa = s["d"].quantile(1 / 3), s["d"].quantile(2 / 3)
    for lbl, (q1, q2) in [("cut_from_fullHP", q), ("cut_from_all_head", qa)]:
        out = []
        for lo, hi in [(-1e9, q1), (q1, q2), (q2, 1e9)]:
            sub = fsub.filter((pl.col("d") > lo) & (pl.col("d") <= hi))
            out.append((sub.height, round(float(sub["one_seq"].mean()), 4)))
        print("2D" if two_d else "3D", lbl, [round(float(x), 1) for x in (q1, q2)], "fullHP (n, one_shot) by tercile:", out)
    # all head rows terciles (any HP)
    out = []
    for lo, hi in [(-1e9, qa[0]), (qa[0], qa[1]), (qa[1], 1e9)]:
        sub = s.filter((pl.col("d") > lo) & (pl.col("d") <= hi))
        out.append((sub.height, round(float(sub["one_seq"].mean()), 4)))
    print("2D" if two_d else "3D", "ALL head rows (any HP) by tercile:", out)

print("== F: helmet split variants, state sample")
state = pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet")).with_columns(
    pl.col("tick").cast(pl.Int64).alias("state_tick"), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
st = state.select(["match_key", "round", "player_id", "state_tick", "has_helmet", "seconds_since_live"]).sort("state_tick")
hits = rif.with_columns(pl.col("tick").alias("hit_tick")).filter(pl.col("match_key").is_in(state["match_key"].unique().to_list())).sort("hit_tick")
print("head hits in sample matches:", hits.height)

def summarize(j, label):
    out = {}
    for w in ["ak47", "m4a1"]:
        for hv in [True, False]:
            s = j.filter((pl.col("weapon_name") == w) & (pl.col("has_helmet") == hv) & pl.col("full_any"))
            out[f"{w}_{'H' if hv else 'noH'}"] = (s.height, round(float(s["one_seq"].mean()), 4) if s.height else None, int(s["one_seq"].sum()) if s.height else 0)
    print(label, out)

j_bwd = hits.join_asof(st, left_on="hit_tick", right_on="state_tick", by=["match_key", "round", "player_id"], strategy="backward")
j_bwd = j_bwd.filter(pl.col("has_helmet").is_not_null())
summarize(j_bwd, "asof backward (all gaps):")
gap = j_bwd["hit_tick"] - j_bwd["state_tick"]
for g in [64, 96, 128]:
    summarize(j_bwd.filter(gap <= g), f"asof backward gap<={g}:")
j_fwd = hits.join_asof(st, left_on="hit_tick", right_on="state_tick", by=["match_key", "round", "player_id"], strategy="forward")
j_fwd = j_fwd.filter(pl.col("has_helmet").is_not_null())
summarize(j_fwd, "asof forward (next state, all gaps):")
j_near = hits.join_asof(st, left_on="hit_tick", right_on="state_tick", by=["match_key", "round", "player_id"], strategy="nearest")
j_near = j_near.filter(pl.col("has_helmet").is_not_null())
summarize(j_near, "asof nearest (all gaps):")
# helmet from first live second of the round (round-level constant)
st0 = st.filter(pl.col("seconds_since_live") == 0).select(["match_key", "round", "player_id", "has_helmet"]).rename({"has_helmet": "helmet_live0"})
j_r = hits.join(st0, on=["match_key", "round", "player_id"], how="inner")
summarize(j_r.rename({"helmet_live0": "has_helmet"}), "helmet at live second 0 (round-level):")
print("print helmet consistency: rounds where helmet changes between state rows (count):")
chg = st.with_columns(pl.col("has_helmet").shift(1).over(["match_key", "round", "player_id"]).alias("ph")).filter(pl.col("ph").is_not_null() & (pl.col("ph") != pl.col("has_helmet")))
print("changes:", chg.height)
print("ALL DONE")
