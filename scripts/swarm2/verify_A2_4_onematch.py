# verify_A2_4_onematch.py: raw rows for one 2024 match (round_start, round_end, freeze_end, win panels, bundle events).
import os, json, glob
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
MID = "510dddb3-cb81-4cdc-9d42-de4423c0bf14"
def folder(ch):
    return [os.path.join(TOME, x) for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch][0]
for ch in ["round_start", "round_end"]:
    f = folder(ch)
    files = sorted(glob.glob(os.path.join(f, "dataframe_*")))
    df = pl.concat([pl.read_parquet(x) for x in files], how="vertical_relaxed")
    print(ch, "files:", [os.path.basename(x) for x in files], "rows total:", df.height, flush=True)
    sub = df.filter(pl.col("match_key").str.contains(MID))
    cols = [c for c in sub.columns if c not in ("match_key", "second")]
    print(sub.select(cols).sort("round").head(40), flush=True)
rsd = pl.concat([pl.read_parquet(x) for x in sorted(glob.glob(os.path.join(folder("round_state"), "dataframe_*")))], how="vertical_relaxed")
sub = rsd.filter(pl.col("match_key").str.contains(MID)).filter(pl.col("event_type").is_in(["round_freeze_end", "round_end", "round_start", "round_prestart", "round_officially_ended", "cs_win_panel_round", "cs_win_panel_match", "round_poststart", "begin_new_match", "round_announce_match_start"]))
print("round_state key events:", sub.select(["round", "tick", "event_type", "t_score", "ct_score", "phase"]).sort("tick").head(80), flush=True)
mk = rsd.filter(pl.col("match_key").str.contains(MID)).select("match_key").unique()
print("match_key(s):", mk.rows(), flush=True)
print("DONE", flush=True)
