# A2: disagreement (score delta vs round_end winner, F convention) by F label k, split by header build_num; phase 'second' first label by build.
import os, sys, json
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
hdr = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=["key", "build_num"]).rename({"key": "match_key"})
st = ch_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "phase", "match_key"])
fe = st.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "t_score", "ct_score", "phase"]).unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"]).unique(["match_key", "round"], keep="first")
nxt = fe.with_columns([pl.col("round").shift(-1).over("match_key").alias("r_next"), pl.col("t_score").shift(-1).over("match_key").alias("t_next"), pl.col("ct_score").shift(-1).over("match_key").alias("ct_next")]).filter(pl.col("r_next") == pl.col("round") + 1)
nxt = nxt.with_columns([(pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT")]).with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
b = nxt.join(re_.select(["match_key", "round", "winner_team_code"]).rename({"round": "rk", "winner_team_code": "w_end"}), left_on=["match_key", "round"], right_on=["match_key", "rk"], how="left").filter(pl.col("w_score") != 0)
b = b.join(hdr, on="match_key", how="left").with_columns((pl.col("w_end") != pl.col("w_score")).cast(pl.Int64).alias("dis"))
t = b.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("bad")).sort(["build_num", "round"])
for bn in [9842, 9961]:
    rows = t.filter(pl.col("build_num") == bn).rows()
    print(f"build {bn} by F label k (k,n,bad) for k<=30:", [(r[1], r[2], r[3]) for r in rows if r[1] <= 30], flush=True)
ph = fe.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("first_second")).join(hdr, on="match_key", how="left")
print("first F label with phase 'second' by build:", ph.group_by(["build_num", "first_second"]).len().sort(["build_num", "first_second"]).rows()[:12], flush=True)
ph_halftime = fe.filter(pl.col("phase") == "halftime").group_by("match_key").agg(pl.col("round").min().alias("first_halftime")).join(hdr, on="match_key", how="left")
print("first F label with phase 'halftime' by build:", ph_halftime.group_by(["build_num", "first_halftime"]).len().sort(["build_num", "first_halftime"]).rows()[:12], flush=True)
print("DONE", flush=True)
