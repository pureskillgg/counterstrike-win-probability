# A2: label-only test of the round_end label convention as used by bakeoff.py (keys: round_freeze_end label k and round_end label k).
# Score change between freeze_end(k) and freeze_end(k+1) = winner of round k. Compare to round_end label k and label k+1 winners.
import os, sys, json, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
st = ch_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"])
fe = st.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick", "t_score", "ct_score"]).sort(["match_key", "round"])
fe = fe.unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
print("freeze_end rows:", fe.height, "unique (match,label):", fe.height, flush=True)
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"]).unique(["match_key", "round"], keep="first")
# sum test at freeze_end label k: t+ct == k-1 ?
s = fe.with_columns((pl.col("t_score") + pl.col("ct_score")).alias("sum"))
s_ok = s.filter(pl.col("sum") == pl.col("round") - 1).height
print(f"freeze_end label k with t+ct == k-1: {s_ok} of {s.height}", flush=True)
# consecutive freeze_end labels
nxt = fe.with_columns([pl.col("round").shift(-1).over("match_key").alias("r_next"),
                       pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                       pl.col("ct_score").shift(-1).over("match_key").alias("ct_next")])
nxt = nxt.filter(pl.col("r_next") == pl.col("round") + 1)
nxt = nxt.with_columns([(pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT")])
nxt = nxt.with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_from_score"))
print("consecutive freeze_end pairs:", nxt.height, " score-derived winner distribution:", nxt.group_by("w_from_score").len().sort("w_from_score").rows(), flush=True)
# compare to round_end label k (same label as the earlier freeze_end k) and label k+1
a = nxt.join(re_.select(["match_key", "round", "winner_team_code"]).rename({"round": "k", "winner_team_code": "w_end_k"}), left_on=["match_key", "round"], right_on=["match_key", "k"], how="left")
a = a.join(re_.select(["match_key", "round", "winner_team_code"]).rename({"round": "k1", "winner_team_code": "w_end_k1"}), left_on=["match_key", "r_next"], right_on=["match_key", "k1"], how="left")
ok = a.filter(pl.col("w_from_score") != 0)
print(f"pairs with score-derived winner: {ok.height}", flush=True)
print("  agree with round_end label k   :", ok.filter(pl.col("w_from_score") == pl.col("w_end_k")).height, flush=True)
print("  agree with round_end label k+1 :", ok.filter(pl.col("w_from_score") == pl.col("w_end_k1")).height, flush=True)
# split by day group
ok = ok.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
ok = ok.with_columns(pl.col("day").map_elements(lambda d: "01" if d == "01" else ("tail02_07" if d <= "07" else "tail12_25"), return_dtype=pl.Utf8).alias("grp"))
g = ok.group_by("grp").agg([pl.len().alias("n"), (pl.col("w_from_score") == pl.col("w_end_k")).sum().alias("eq_k"), (pl.col("w_from_score") == pl.col("w_end_k1")).sum().alias("eq_k1")])
print(g.sort("grp"), flush=True)
print("DONE", flush=True)
