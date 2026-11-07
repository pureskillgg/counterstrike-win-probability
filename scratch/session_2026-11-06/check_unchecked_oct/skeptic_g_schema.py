import os, json, collections
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2"
W = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
files = []
names_hit = []
for dp, dn, fn in os.walk(ROOT):
    # skip the pixi env and anything that looks like a holdout-B path
    if ".pixi" in dp or "holdout" in dp.lower():
        dn[:] = []
        continue
    for f in fn:
        p = os.path.join(dp, f)
        if f.lower().endswith(".parquet"):
            files.append(p)
        if f.lower() in ("player_info.parquet", "basic_player_info.parquet", "roster.parquet") or "player_info" in f.lower() or "roster" in f.lower():
            names_hit.append(p)
print("parquet files (excl. .pixi and holdout paths)", len(files), flush=True)
print("files with player_info/roster in name", len(names_hit), names_hit[:10], flush=True)

team_code_files = []
unreadable = []
for p in files:
    try:
        sch = pq.read_schema(p)
    except Exception as e:
        unreadable.append((p, str(e)[:60]))
        continue
    if "team_code" in sch.names:
        team_code_files.append(p)

out = {
    "parquet_files_checked": len(files) - len(unreadable),
    "unreadable_count": len(unreadable),
    "unreadable_examples": unreadable[:8],
    "files_with_team_code": team_code_files,
    "name_hits": names_hit[:20],
}
with open(W + "skeptic_g_schema.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)[:3000], flush=True)
print("DONE", flush=True)
