# A2: inspect round numbering in round_start / round_end for sample matches (day 01 merged, day 02, day 12) and the dup round_end group.
import os, json, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
rs = ch_frame("round_start", ["round", "tick", "match_key"])
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "player_count", "match_key"])
for mk in ["csds/2024/01", "csds/2024/02/", "csds/2024/12/"]:
    pass
samples = {}
for d in ["01", "02", "12", "20"]:
    m = rs.filter(pl.col("match_key").str.contains(f"/2024/02/{d}/")).select("match_key").unique().head(1)
    if m.height: samples[d] = m["match_key"][0]
for d, mk in samples.items():
    a = rs.filter(pl.col("match_key") == mk).sort("tick").select(["round", "tick"])
    b = re_.filter(pl.col("match_key") == mk).sort("tick").select(["round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    print(f"== day {d} {mk}")
    print("round_start rounds:", a["round"].to_list())
    print("round_start ticks :", a["tick"].to_list())
    print("round_end rounds  :", b["round"].to_list())
    print("round_end ticks   :", b["tick"].to_list())
    print("round_end winner  :", b["winner_team_code"].to_list())
    print("round_end reason  :", b["win_reason_code"].to_list())
dup = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("dup round_end groups:", dup.to_dicts())
for r in dup.iter_rows(named=True):
    print(re_.filter((pl.col("match_key") == r["match_key"]) & (pl.col("round") == r["round"])).to_dicts())
mk2 = "csds/2024/02/12/2f5bcb89-9ef8-4788-9f3a-2fa3d633ea4d/csds"
print("match without round_start:", mk2)
print("  round_end rows:", re_.filter(pl.col("match_key") == mk2).sort("tick").select(["round", "tick", "winner_team_code", "win_reason_code"]).rows())
print("  round_start rows:", rs.filter(pl.col("match_key") == mk2).height)
print("round dtype rs:", rs.schema["round"], "re:", re_.schema["round"])
print("DONE")
