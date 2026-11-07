# A2: same score-vs-winner test as A2_label_shift_test.py, on the 2026 tome, split by header build_num.
import os, sys, json
import polars as pl
T26 = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def cdir(ch):
    c = [d for d in os.listdir(T26) if d.startswith("basic_" + ch + ".")]
    return os.path.join(T26, c[0])
def ch_frame(ch, cols):
    dp = cdir(ch)
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
hdr = pl.read_parquet(os.path.join(T26, "header", "dataframe_00000"), columns=["key", "build_num"]).rename({"key": "match_key"})
st = ch_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"])
fe = st.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "t_score", "ct_score"]).unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
re_ = ch_frame("round_end", ["round", "winner_team_code", "match_key"]).unique(["match_key", "round"], keep="first")
print("2026 freeze_end rows:", fe.height, " sum==k-1:", fe.filter((pl.col("t_score") + pl.col("ct_score")) == (pl.col("round") - 1)).height, flush=True)
nxt = fe.with_columns([pl.col("round").shift(-1).over("match_key").alias("r_next"), pl.col("t_score").shift(-1).over("match_key").alias("t_next"), pl.col("ct_score").shift(-1).over("match_key").alias("ct_next")]).filter(pl.col("r_next") == pl.col("round") + 1)
nxt = nxt.with_columns([(pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT")]).with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
b = nxt.join(re_.select(["match_key", "round", "winner_team_code"]).rename({"round": "rk", "winner_team_code": "w_end"}), left_on=["match_key", "round"], right_on=["match_key", "rk"], how="left").filter(pl.col("w_score") != 0)
b = b.join(hdr, on="match_key", how="left").with_columns((pl.col("w_end") != pl.col("w_score")).cast(pl.Int64).alias("dis"))
print("2026 pairs:", b.height, " disagree:", int(b["dis"].sum()), flush=True)
print("by build (build, pairs, disagree, rate):", [(r[0], r[1], r[2], round(r[2] / r[1], 4)) for r in b.group_by("build_num").agg(pl.len().alias("n"), pl.col("dis").sum().alias("bad")).sort("build_num").rows()], flush=True)
t = b.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("bad")).sort(["build_num", "round"])
for bn in sorted(set(hdr["build_num"].to_list())):
    rows = t.filter(pl.col("build_num") == bn).rows()
    print(f"build {bn} bad rounds (k,n,bad) with bad>0:", [(r[1], r[2], r[3]) for r in rows if r[3] > 0][:20], flush=True)
print("DONE", flush=True)
