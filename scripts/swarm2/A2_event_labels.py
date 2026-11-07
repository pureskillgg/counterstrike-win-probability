# A2: which round label convention do round_state events use? Compare round_state event labels/ticks with round_start and round_end channels.
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
rs = ch_frame("round_start", ["round", "tick", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
# join event rows to round_start rows by (match, tick) and compare labels
for ev in ["round_prestart", "round_poststart", "round_officially_ended", "round_freeze_end", "round_end", "round_start"]:
    e = st.filter(pl.col("event_type") == ev).select(["match_key", "round", "tick", "day"])
    m = e.join(rs.select(["match_key", "round", "tick"]).rename({"round": "rs_round"}), on=["match_key", "tick"], how="left")
    m2 = e.join(re_.select(["match_key", "round", "tick"]).rename({"round": "re_round"}), on=["match_key", "tick"], how="left")
    print(f"{ev}: rows={e.height} | same-tick round_start found={m['rs_round'].is_not_null().sum()} label_eq_rs={(m['round']==m['rs_round']).sum()} "
          f"| same-tick round_end found={m2['re_round'].is_not_null().sum()} label_eq_re={(m2['round']==m2['re_round']).sum()}", flush=True)
    print("   label minus round_start label at same tick:", m.filter(pl.col("rs_round").is_not_null()).with_columns((pl.col("round")-pl.col("rs_round")).alias("d")).group_by("d").len().sort("d").rows()[:8], flush=True)
# tick-relative: for round_officially_ended, is the tick after round_start(label) of the same label?
roe = st.filter(pl.col("event_type") == "round_officially_ended").select(["match_key", "round", "tick", "t_score", "ct_score"]).sort("tick")
rsj = roe.join_asof(rs.select(["match_key", "round", "tick"]).rename({"round": "s_round", "tick": "s_tick"}).sort("s_tick"),
                    left_on="tick", right_on="s_tick", by="match_key", strategy="backward")
print("round_officially_ended: label minus round in progress (latest round_start before it):", rsj.with_columns((pl.col("round")-pl.col("s_round")).alias("d")).group_by("d").len().sort("d").rows()[:6], flush=True)
# score-based winner: for round_officially_ended, compare score delta vs next event with score
print("DONE", flush=True)
