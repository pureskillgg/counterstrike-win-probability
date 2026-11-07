import polars as pl, sys, json
sys.stdout.reconfigure(encoding="utf-8")
R = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/refute_oct/"
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
base = "G:/pureskill/csds-cs2/data/"
# extra keys: from the space listing copied locally by the other agent (read-only use)
extra = pl.read_csv(S + "s3_extra_oct21_25_keys.csv")
print("extra csv cols", extra.columns, extra.height, flush=True)
col = [c for c in extra.columns if "key" in c.lower()][0]
ek = pl.DataFrame({"match_key": extra[col].to_list()}).lazy()
# compare with the space listing computed just now
try:
    j = json.load(open(R + "refute_s3_extra_space.json"))
except Exception:
    j = None
tot = {}
for src in ["positions_1hz", "status_1hz"]:
    hits = 0; allk = set()
    for d in ["21","22","23","24","25"]:
        lf = pl.scan_parquet(f"{base}{src}/2026-10-{d}.parquet").select("match_key")
        ks = lf.unique().collect()["match_key"].to_list()
        allk.update(ks)
    print(src, "distinct match keys 10/21..25 (all, batch+HB+other):", len(allk), flush=True)
    print(src, "extra156 present:", len(set(extra[col].to_list()) & allk), flush=True)
    tot[src] = {"all_keys": len(allk), "extra_present": len(set(extra[col].to_list()) & allk)}
json.dump(tot, open(R + "refute_extra_local.json", "w"), indent=1)
