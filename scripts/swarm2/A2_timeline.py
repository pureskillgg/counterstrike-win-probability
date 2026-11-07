# A2: event timeline for sample matches (round_state events, round_start channel, round_end channel), sorted by tick.
import os, sys, json
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
st = ch_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"]).with_columns(pl.col("event_type").alias("ev"), pl.lit(-1).alias("winner"))
rs = ch_frame("round_start", ["round", "tick", "match_key"]).with_columns(pl.lit("CH_round_start").alias("ev"), pl.lit(-1).alias("t_score"), pl.lit(-1).alias("ct_score"), pl.lit(-1).alias("winner"))
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"]).with_columns(pl.lit("CH_round_end").alias("ev"), pl.lit(-1).alias("t_score"), pl.lit(-1).alias("ct_score"), pl.col("winner_team_code").cast(pl.Int64).alias("winner"))
cols = ["match_key", "round", "tick", "ev", "t_score", "ct_score", "winner"]
ev = pl.concat([st.select(cols), rs.select(cols), re_.select(cols)], how="vertical_relaxed")
wanted = {"round_prestart", "round_freeze_end", "round_officially_ended", "CH_round_end", "round_end", "CH_round_start", "bomb_planted", "round_poststart", "cs_round_start_beep", "buytime_ended"}
for mk in ["csds/2024/02/01/b04f0e79-0350-4228-b07d-5c74a4a55fa2/csds", "csds/2024/02/12/734e42e0-3ad6-4c1e-af4d-dc3385177620/csds"]:
    e = ev.filter((pl.col("match_key") == mk) & (pl.col("round") <= 3) & pl.col("ev").is_in(list(wanted))).sort("tick")
    print("==", mk, flush=True)
    for r in e.iter_rows(named=True):
        print(f"  tick={r['tick']:>7} label={r['round']:>3} {r['ev']:<24} t={r['t_score']} ct={r['ct_score']} winner={r['winner']}", flush=True)
print("DONE", flush=True)
