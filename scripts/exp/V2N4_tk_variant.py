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
ph = ph.sort(["match_key", "round", "player_id", "tick", "health"], descending=[False, False, False, False, True])
ph = ph.with_columns(pl.col("health").shift(1).over(["match_key", "round", "player_id"]).alias("prev_health"))
ph = ph.with_columns(
    pl.when(pl.col("prev_health").is_null()).then(pl.lit(100)).otherwise(pl.col("prev_health")).alias("hp_seq"))
ph = ph.with_columns((pl.col("hp_seq") == 100).alias("full"), (pl.col("health") == 0).alias("lethal_h0"))

head = ph.filter(pl.col("weapon_name").is_in(["ak47", "m4a1"]) & (pl.col("hit_box_code") == 1)).with_row_index("hid")
head = head.with_columns((pl.col("health_removed") >= pl.col("hp_seq")).alias("one"))

# team-code audit for full-HP rows (AK and M4)
fh = head.filter(pl.col("full"))
print("== team code audit (full-HP head hits)")
for w in ["ak47", "m4a1"]:
    s = fh.filter(pl.col("weapon_name") == w)
    same = s.filter(pl.col("attacker_team_code") == pl.col("player_team_code")).height
    diff = s.filter(pl.col("attacker_team_code") != pl.col("player_team_code")).height
    null_att = s.filter(pl.col("attacker_team_code").is_null()).height
    null_vic = s.filter(pl.col("player_team_code").is_null()).height
    print(w, "n_full", s.height, "same_team(TK)", same, "diff_team", diff, "null_attacker_team", null_att, "null_victim_team", null_vic)
    print(w, "one-shot among same-team TK:", round(float(s.filter(pl.col("attacker_team_code") == pl.col("player_team_code"))["one"].mean()), 4) if same else None)
    print(w, "attacker_team_code value counts (full):", s.group_by("attacker_team_code").len().sort("attacker_team_code").to_dicts())

def stats(h, label):
    out = {"label": label}
    for w in ["ak47", "m4a1"]:
        s = h.filter(pl.col("weapon_name") == w)
        n = s.height
        strict = s.filter(pl.col("full") & pl.col("one")).height
        nfull = s.filter(pl.col("full")).height
        k = s.filter(pl.col("full") & pl.col("one")).height
        out[w] = {
            "n_head": n,
            "mean_hr_uncapped": round(float(s["health_removed"].mean()), 3),
            "strict_share_all_head": round(strict / n, 4) if n else None,
            "n_full_hp": nfull,
            "one_shot_given_full_hp": round(k / nfull, 4) if nfull else None,
            "n_full_hp_one_shot": k,
            "share_full_hp": round(nfull / n, 4) if n else None,
            "one_shot_all_head": round(float(s["one"].mean()), 4) if n else None,
        }
    print(label, out)
    return out

print("== stats with and without team kills")
R_withTK = stats(head, "ALL head rows (TK included)")
noTK = head.filter(pl.col("attacker_team_code") != pl.col("player_team_code"))
R_noTK = stats(noTK, "team kills excluded (attacker_team != victim_team; nulls dropped)")

# distance terciles (3D) on AK full-HP, TK excluded
def d3(df):
    return df.with_columns(((pl.col("player_x_pos") - pl.col("attacker_x_pos")) ** 2 + (pl.col("player_y_pos") - pl.col("attacker_y_pos")) ** 2 + (pl.col("player_z_pos") - pl.col("attacker_z_pos")) ** 2).sqrt().alias("d"))
print("== AK full-HP one-shot by 3D-distance tercile (TK excluded)")
ak = d3(noTK.filter(pl.col("weapon_name") == "ak47")).filter(pl.col("d").is_not_null())
fakefull = ak.filter(pl.col("full"))
q = fakefull["d"].quantile(1 / 3), fakefull["d"].quantile(2 / 3)
print("cut from full-HP subset:", [round(float(x), 2) for x in q])
rows = []
for lo, hi in [(-1e9, q[0]), (q[0], q[1]), (q[1], 1e9)]:
    sub = fakefull.filter((pl.col("d") > lo) & (pl.col("d") <= hi))
    rows.append((sub.height, round(float(sub["one"].mean()), 4)))
print("full-HP (n, one_shot) T1,T2,T3:", rows)
qa = ak["d"].quantile(1 / 3), ak["d"].quantile(2 / 3)
print("cut from all AK head (TK excl):", [round(float(x), 2) for x in qa])
rows2 = []
for lo, hi in [(-1e9, qa[0]), (qa[0], qa[1]), (qa[1], 1e9)]:
    sub = ak.filter((pl.col("d") > lo) & (pl.col("d") <= hi))
    fsub = sub.filter(pl.col("full"))
    rows2.append({"all_head_one": round(float(sub["one"].mean()), 4), "n_full": fsub.height, "full_one": round(float(fsub["one"].mean()), 4)})
print("AK (all-head one-shot, full n, full one-shot) by tercile (cut from all AK):", rows2)
m4 = d3(noTK.filter(pl.col("weapon_name") == "m4a1")).filter(pl.col("d").is_not_null())
mf = m4.filter(pl.col("full"))
qm = mf["d"].quantile(1 / 3), mf["d"].quantile(2 / 3)
rows3 = []
for lo, hi in [(-1e9, qm[0]), (qm[0], qm[1]), (qm[1], 1e9)]:
    sub = mf.filter((pl.col("d") > lo) & (pl.col("d") <= hi))
    rows3.append((sub.height, round(float(sub["one"].mean()), 4)))
print("M4 full-HP tercile (n, one_shot):", rows3)

# helmet join (asof backward), TK excluded
state = pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet")).with_columns(
    pl.col("tick").cast(pl.Int64).alias("state_tick"), pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
st = state.select(["match_key", "round", "player_id", "state_tick", "has_helmet"]).sort("state_tick")
hits = noTK.filter(pl.col("match_key").is_in(state["match_key"].unique().to_list())).with_columns(pl.col("tick").alias("hit_tick")).sort("hit_tick")
j = hits.join_asof(st, left_on="hit_tick", right_on="state_tick", by=["match_key", "round", "player_id"], strategy="backward").filter(pl.col("has_helmet").is_not_null())
gap = j["hit_tick"] - j["state_tick"]
print("== helmet split (asof backward, TK excluded)")
for label, jj in [("all gaps", j), ("gap<=64 ticks", j.filter(gap <= 64))]:
    res = {}
    for w in ["ak47", "m4a1"]:
        for hv in [True, False]:
            s = jj.filter((pl.col("weapon_name") == w) & (pl.col("has_helmet") == hv) & pl.col("full"))
            res[f"{w}_{'helmet' if hv else 'nohelmet'}"] = {"n_full": s.height, "one_shot": round(float(s["one"].mean()), 4) if s.height else None, "k": int(s["one"].sum()) if s.height else 0}
    print(label, res)

# helmet on AK including all head rows (no full filter) for context
print("ALL DONE")
