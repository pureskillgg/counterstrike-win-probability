# A2: label-free check. Score change across each round_end tick (round_state t_score/ct_score before vs after)
# must match winner_team_code (2 = T, 3 = CT). Compares day groups. Also reports round_state event_type values.
import os, sys, json, csv, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SCR = sys.argv[1]
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
st = ch_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "phase", "is_warmup", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
print("round_state rows:", st.height, "event_type:", st.group_by("event_type").len().sort("len", descending=True).head(12).rows(), flush=True)
print("round_state phase:", st.group_by("phase").len().sort("len", descending=True).head(8).rows(), flush=True)
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
st = st.sort("tick")
# before = last state with tick < te ; after = first state with tick > te
before = re_.sort("tick").join_asof(st.select(["match_key", "tick", "t_score", "ct_score"]).rename({"tick": "tb", "t_score": "tsb", "ct_score": "csb"}).sort("tb"),
                                    left_on="tick", right_on="tb", by="match_key", strategy="backward")
after = re_.sort("tick").join_asof(st.select(["match_key", "tick", "t_score", "ct_score"]).rename({"tick": "ta", "t_score": "tsa", "ct_score": "csa"}).sort("ta"),
                                   left_on="tick", right_on="ta", by="match_key", strategy="forward")
j = before.with_columns(pl.col("tick").alias("te")).join(after.select(["match_key", "tick", "tsa", "csa"]), on=["match_key", "tick"], how="left")
j = j.with_columns([(pl.col("tsa") - pl.col("tsb")).alias("dt"), (pl.col("csa") - pl.col("csb")).alias("dct")])
def cls(r):
    w, dt, dct = r["winner_team_code"], r["dt"], r["dct"]
    if dt is None or dct is None: return "no_state"
    if w == 2:
        return "T_win_matches" if (dt == 1 and dct == 0) else ("T_win_other" if (dt, dct) != (0, 0) else "T_win_no_change")
    if w == 3:
        return "CT_win_matches" if (dct == 1 and dt == 0) else ("CT_win_other" if (dt, dct) != (0, 0) else "CT_win_no_change")
    return "other_winner"
j = j.with_columns(pl.struct(["winner_team_code", "dt", "dct"]).map_elements(cls, return_dtype=pl.Utf8).alias("cls"))
print("overall class counts:", j.group_by("cls").len().sort("len", descending=True).rows(), flush=True)
pd_ = j.group_by(["day", "cls"]).len().sort(["day", "cls"])
dd = collections.defaultdict(dict)
for r in pd_.iter_rows(named=True): dd[r["day"]][r["cls"]] = r["len"]
for d in sorted(dd): print(" ", d, dd[d], flush=True)
# same check using label L against round_state label-based score (label = round_state round at te)
with open(os.path.join(SCR, "A2_score_tick_rows.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["day", "match_key", "label", "tick", "winner", "dt", "dct", "cls"])
    for r in j.iter_rows(named=True):
        w.writerow([r["day"], r["match_key"], r["round"], r["tick"], r["winner_team_code"], r["dt"], r["dct"], r["cls"]])
print("DONE", flush=True)
