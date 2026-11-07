import json, re, collections
import numpy as np
import polars as pl
S = r"C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/inputs"
mk_v2 = json.load(open(S + "/match_keys_v2.json"))
built = json.load(open(S + "/built_match_keys.json"))
split = np.load(S + "/match_split.npy")
hb = json.load(open(r"C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/combined/holdout_22_25_test_keys.json"))["match_keys"]
h26 = pl.read_parquet(r"C:/Users/13144/sprint/holdout_oct26_31_keys.parquet", columns=["match_key"])["match_key"].to_list()
h26t = pl.read_parquet(r"C:/Users/13144/sprint/holdout_oct26_31/keys.parquet", columns=["match_key"])["match_key"].unique().to_list()
print("export match_keys:", len(mk_v2), "unique:", len(set(mk_v2)))
print("built keys:", len(built), "unique:", len(set(built)))
print("split counts (match_split.npy):", {int(c): int((split == c).sum()) for c in np.unique(split)})
print("holdout B keys:", len(hb), "unique:", len(set(hb)))
print("Oct26-31 keys parquet rows:", len(h26), "unique:", len(set(h26)))
print("Oct26-31 tome keys.parquet unique match_key:", len(h26t))
print("sample formats:")
print("  export:", mk_v2[0], "| built:", built[0], "| holdB:", hb[0], "| oct26 parquet:", h26[0], "| oct26 tome:", h26t[0])
uuid = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
def ids(lst):
    out = set()
    for k in lst:
        m = uuid.search(k)
        out.add(m.group(0) if m else k)
    return out
exp_set = set(mk_v2)
built_set = set(built)
frozen_test = {mk_v2[i] for i in range(len(mk_v2)) if split[i] == 2}
trainval = {mk_v2[i] for i in range(len(mk_v2)) if split[i] in (0, 1)}
print("frozen test keys (split 2):", len(frozen_test))
print("built keys not in export index (exact):", len(built_set - exp_set))
print("built keys not in export index (uuid):", len(ids(built) - ids(mk_v2)))
print("built keys whose split is 2 (frozen test), exact:", len(built_set & frozen_test))
print("built keys in holdout B (exact):", len(built_set & set(hb)), " uuid:", len(ids(built) & ids(hb)))
print("built keys in Oct26-31 keys parquet (exact):", len(built_set & set(h26)), " uuid:", len(ids(built) & ids(h26)))
print("built keys in Oct26-31 tome keys (exact):", len(built_set & set(h26t)), " uuid:", len(ids(built) & ids(h26t)))
print("holdout B in export index (exact):", len(set(hb) & exp_set), " uuid:", len(ids(hb) & ids(mk_v2)))
print("Oct26-31 parquet in export index (exact):", len(set(h26) & exp_set), " uuid:", len(ids(h26) & ids(mk_v2)))
print("Oct26-31 tome in export index (exact):", len(set(h26t) & exp_set), " uuid:", len(ids(h26t) & ids(mk_v2)))
print("holdout B in frozen test (split 2, exact):", len(set(hb) & frozen_test))
print("holdout B in train/val split (exact):", len(set(hb) & trainval))
dates = [k.split("/")[1:4] for k in built]
dstr = ["-".join(d) for d in dates]
print("built path-date min/max:", min(dstr), max(dstr))
print("built keys with path date >= 2026-10-21:", sum(1 for d in dstr if d >= "2026-10-21"))
print("built keys with path date in 2026-10-21..2026-10-31:", sum(1 for d in dstr if "2026-10-21" <= d <= "2026-10-31"))
print("built keys with path date >= 2026-10-26:", sum(1 for d in dstr if d >= "2026-10-26"))
hbdates = sorted(["-".join(k.split("/")[1:4]) for k in hb])
print("holdout B path-date min/max:", hbdates[0], hbdates[-1])
print("built keys in holdout B by path date range overlap (Oct 21-25 count):", sum(1 for d in dstr if "2026-10-21" <= d <= "2026-10-25"))
