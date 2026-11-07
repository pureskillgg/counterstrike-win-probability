# A2: label audit with the freeze_end (F) convention the bake-off uses.
#  a. per match: min/max label of round_freeze_end (F), round_end (E), round_start (S); counts by group.
#  b. score-vs-winner disagreements by F label k (score delta freeze_end(k)->freeze_end(k+1) vs round_end label k winner).
#  c. first F label where phase == 'second' per match, by group.
import os, sys, json, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
def grp(d):
    return "01" if d == "01" else ("tail02_07" if d <= "07" else "tail12_25")
st = ch_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "phase", "match_key"])
st = st.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
fe = st.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "day", "round", "t_score", "ct_score", "phase"]).unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"]).unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
rs = ch_frame("round_start", ["round", "tick", "match_key"]).unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
# a. per-match label ranges
fa = fe.group_by("match_key").agg(pl.col("round").min().alias("fe_min"), pl.col("round").max().alias("fe_max"), pl.col("day").first())
ea = re_.group_by("match_key").agg(pl.col("round").min().alias("re_min"), pl.col("round").max().alias("re_max"))
sa = rs.group_by("match_key").agg(pl.col("round").min().alias("rs_min"), pl.col("round").max().alias("rs_max"))
m = fa.join(ea, on="match_key", how="left").join(sa, on="match_key", how="left")
m = m.with_columns(pl.col("day").map_elements(grp, return_dtype=pl.Utf8).alias("grp"))
m = m.with_columns([(pl.col("re_min") == 1).alias("re_has_1"), (pl.col("fe_min") == 1).alias("fe_has_1"), (pl.col("rs_min") == 1).alias("rs_has_1")])
print("a. matches by group (n):", m.group_by("grp").len().sort("grp").rows(), flush=True)
print("   fe_min distribution:", m.group_by(["grp", "fe_min"]).len().sort(["grp", "fe_min"]).rows()[:12], flush=True)
print("   re_min distribution:", m.group_by(["grp", "re_min"]).len().sort(["grp", "re_min"]).rows()[:12], flush=True)
print("   matches with round_end label 1 present:", m.group_by(["grp", "re_has_1"]).len().sort(["grp", "re_has_1"]).rows(), flush=True)
print("   matches with freeze_end label 1 present:", m.group_by(["grp", "fe_has_1"]).len().sort(["grp", "fe_has_1"]).rows(), flush=True)
print("   matches with round_start label 1 present:", m.group_by(["grp", "rs_has_1"]).len().sort(["grp", "rs_has_1"]).rows(), flush=True)
# b. disagreements by label k
nxt = fe.with_columns([pl.col("round").shift(-1).over("match_key").alias("r_next"),
                       pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                       pl.col("ct_score").shift(-1).over("match_key").alias("ct_next")]).filter(pl.col("r_next") == pl.col("round") + 1)
nxt = nxt.with_columns([(pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT")])
nxt = nxt.with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
b = nxt.join(re_.select(["match_key", "round", "winner_team_code"]).rename({"round": "round_k", "winner_team_code": "w_end"}), left_on=["match_key", "round"], right_on=["match_key", "round_k"], how="left")
b = b.filter(pl.col("w_score") != 0).with_columns(pl.col("day").map_elements(grp, return_dtype=pl.Utf8).alias("grp"))
b = b.with_columns((pl.col("w_end") != pl.col("w_score")).alias("disagree"))
print("b. disagreements (score delta vs round_end label k), by group:", b.group_by(["grp", "disagree"]).len().sort(["grp", "disagree"]).rows(), flush=True)
bk = b.group_by(["grp", "round"]).agg(pl.len().alias("n"), pl.col("disagree").sum().alias("bad")).sort(["grp", "round"])
for g in ["01", "tail02_07", "tail12_25"]:
    rows = bk.filter(pl.col("grp") == g).rows()
    print(f"   {g} by label k (k, n, bad): ", [(r[1], r[2], r[3]) for r in rows if r[1] <= 30][:32], flush=True)
# c. halftime: first F label with phase == second, per match
ph = fe.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("first_second_F"))
ph = ph.join(m.select(["match_key", "grp"]), on="match_key", how="left")
print("c. first F label with phase 'second' (grp: label:count):")
for g in ["01", "tail02_07", "tail12_25"]:
    c = ph.filter(pl.col("grp") == g).group_by("first_second_F").len().sort("first_second_F").rows()
    print("  ", g, c, flush=True)
print("DONE", flush=True)
