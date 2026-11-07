import json, os, sys, glob
import polars as pl
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
sys.path.insert(0, S + "/scripts2")
from loadout_rule import INPUT_COLS
req = set(json.load(open(S + "/out/local_req_keys.json")))
os.makedirs(S + "/out/reduced", exist_ok=True)
cols = ["match_key", "round", "tick", "player_id", "side"] + INPUT_COLS
tot_in = tot_out = 0
for f in sorted(glob.glob("G:/pureskill/csds-cs2/data/status_1hz/2026-10-*.parquet")):
    name = os.path.basename(f)[:10]
    if not (("2026-10-09" <= name <= "2026-10-12") or name == "2026-10-20"):
        continue
    raw = pl.read_parquet(f, columns=cols)
    n_in = raw.height
    red = raw.filter(pl.col("match_key").is_in(list(req)))
    tot_in += n_in
    tot_out += red.height
    red.write_parquet(S + "/out/reduced/" + os.path.basename(f), compression="zstd")
    print(name, "rows in", n_in, "rows kept", red.height, "matches kept", red["match_key"].n_unique(), flush=True)
print("total rows in", tot_in, "kept", tot_out)
