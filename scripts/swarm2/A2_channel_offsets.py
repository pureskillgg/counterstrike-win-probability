# A2: for event channels with (round, tick), offset = event label - round in progress (latest round_start/prestart tick <= event tick).
# Shows which label convention each channel follows. Day 01 vs tail days reported separately.
import os, sys, json, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
rs = ch_frame("round_start", ["round", "tick", "match_key"]).rename({"round": "s_round", "tick": "s_tick"}).sort("s_tick")
rs = rs.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
def grp(d):
    return "01" if d == "01" else ("tail07" if d in ("02","03","04","05","06","07") else "tail12_25")
for ch in ["player_death", "weapon_fire", "bomb_state", "bomb_action", "bomb_defuse", "round_mvp", "round_state", "player_hurt", "weapon_action", "player_spawn"]:
    try:
        ev = ch_frame(ch, ["round", "tick", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day")).sort("tick")
    except Exception as e:
        print(ch, "ERR", str(e)[:100]); continue
    j = ev.join_asof(rs, left_on="tick", right_on="s_tick", by="match_key", strategy="backward")
    j = j.with_columns((pl.col("round") - pl.col("s_round")).alias("d"))
    j = j.with_columns(pl.col("day").map_elements(grp, return_dtype=pl.Utf8).alias("grp"))
    tot = j.group_by("d").len().sort("d").rows()[:6]
    g = j.group_by(["grp", "d"]).len().sort(["grp", "d"]).rows()
    print(f"{ch}: rows={j.height} overall d:count {tot}  | by group {g[:12]}", flush=True)
print("DONE", flush=True)
