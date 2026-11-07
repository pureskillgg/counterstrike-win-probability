# A2: label-free truth test of round windows. Score delta between consecutive round_prestart states (P_{k-1}, P_k) should equal
# the single round winner whose round_end tick lies in that window. Compare two windows for each round_end:
#   window W1 = [P_{k-1}, P_k)  (round_end's tick falls in the round that started at P_{k-1})
#   window W2 = [P_k, P_{k+1})  (round_end's tick falls in the round that starts at P_k)
# Also tests score deltas at round_officially_ended ticks. Reports agreement counts per day group.
import os, sys, json, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
st = ch_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
pre = st.filter(pl.col("event_type") == "round_prestart").select(["match_key", "day", "round", "tick", "t_score", "ct_score"]).sort(["match_key", "tick"])
pre = pre.with_columns([
    pl.col("tick").shift(-1).over("match_key").alias("P_next"),
    pl.col("tick").shift(1).over("match_key").alias("P_prev"),
    pl.col("t_score").shift(-1).over("match_key").alias("T_next"),
    pl.col("ct_score").shift(-1).over("match_key").alias("CT_next")])
pre = pre.with_columns([(pl.col("T_next") - pl.col("t_score")).alias("dT"), (pl.col("CT_next") - pl.col("ct_score")).alias("dCT")])
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
# for each round_end, find the prestart window it falls in: latest prestart with tick <= te
j = re_.sort("tick").join_asof(pre.select(["match_key", "tick", "P_next", "dT", "dCT", "t_score", "ct_score"]).rename({"tick": "Pk"}).sort("Pk"),
                              left_on="tick", right_on="Pk", by="match_key", strategy="backward")
def agree(w, dT, dCT):
    if dT is None or dCT is None: return "no_state"
    if w == 2: return "T_ok" if (dT == 1 and dCT == 0) else "T_bad"
    if w == 3: return "CT_ok" if (dCT == 1 and dT == 0) else "CT_bad"
    return "other"
# W1: window that contains te is [Pk, P_next). Score delta across that window = (score at P_next - score at Pk).
j = j.with_columns(pl.struct(["winner_team_code", "dT", "dCT"]).map_elements(lambda r: agree(r["winner_team_code"], r["dT"], r["dCT"]), return_dtype=pl.Utf8).alias("W1"))
# W2 test: window is [P_prev_of_Pk, Pk): score delta between previous prestart and Pk
prev = pre.select(["match_key", "tick", "P_prev", "t_score", "ct_score"]).rename({"tick": "Pk"})
prev = prev.with_columns([(pl.col("t_score") - pl.col("t_score")).alias("x")])  # placeholder
pre2 = pre.with_columns([(pl.col("t_score") - pl.col("t_score").shift(1).over("match_key")).alias("dT_in"), (pl.col("ct_score") - pl.col("ct_score").shift(1).over("match_key")).alias("dCT_in")])
j2 = j.join(pre2.select(["match_key", "tick", "dT_in", "dCT_in"]).rename({"tick": "Pk"}), on=["match_key", "Pk"], how="left")
j2 = j2.with_columns(pl.struct(["winner_team_code", "dT_in", "dCT_in"]).map_elements(lambda r: agree(r["winner_team_code"], r["dT_in"], r["dCT_in"]), return_dtype=pl.Utf8).alias("W2"))
j2 = j2.with_columns(pl.col("day").map_elements(lambda d: "01" if d == "01" else ("tail02_07" if d <= "07" else "tail12_25"), return_dtype=pl.Utf8).alias("grp"))
print("rows:", j2.height, flush=True)
print("W1 (window starting at round_prestart P_k, i.e. label convention of round_start):", j2.group_by(["grp", "W1"]).len().sort(["grp", "W1"]).rows(), flush=True)
print("W2 (window ending at round_prestart P_k, i.e. one-earlier window):", j2.group_by(["grp", "W2"]).len().sort(["grp", "W2"]).rows(), flush=True)
print("DONE", flush=True)
