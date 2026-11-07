"""Counts-only scan of the three swarm2 D7 parquet files, including the list column evs.
Exclusion uses match_key as in value_screen.py. Counts only; no raw values printed."""
import json
import re
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_KEYS_FROZEN = "C:/Users/<user>/Documents/pureskill_internship/csds-cs2/data/benchmark/frozen_test_keys.json"
HOLDOUT = "C:/Users/<user>/Documents/pureskill_internship/csds-cs2/data/combined/holdout_22_25_test_keys.json"
DATE_OCT = r"/[0-9]{4}/10/(2[6-9]|3[01])/"
PATTERNS = {
    "steam2": r"STEAM_[0-5]:[01]:[0-9]+", "steam3": r"\[U:1:[0-9]+\]",
    "steamid64": r"7656119[0-9]{10}", "ipv4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "win_user_path": r"[A-Za-z]:\\+Users\\+", "steamcommunity": r"steamcommunity",
    "digits_9plus": r"[0-9]{9,}",
}
frozen = json.load(open(REPO_KEYS_FROZEN, encoding="utf-8"))
hd = json.load(open(HOLDOUT, encoding="utf-8"))
excl = pa.array(sorted(set(frozen) | set(hd["match_keys"])), type=pa.string())
files = [
    "C:/Users/<user>/Documents/pureskill/csds-cs2/logs/swarm2/D7_rounds_all.parquet",
    "C:/Users/<user>/Documents/pureskill/csds-cs2/logs/swarm2/D7_rounds_with_ledger.parquet",
    "C:/Users/<user>/Documents/pureskill/csds-cs2/logs/swarm2/D7_rounds_with_ledger_dp.parquet",
]
files = ["G:/pureskill/csds-cs2/logs/swarm2/" + f.split("/")[-1] for f in files]
for f in files:
    t = pq.read_table(f)
    key = t.column("match_key").combine_chunks()
    ex = pc.fill_null(pc.is_in(key, value_set=excl), False)
    oc = pc.fill_null(pc.match_substring_regex(key, DATE_OCT), False)
    keep = pc.invert(pc.or_(ex, oc))
    kept = t.filter(keep)
    res = {"file": f.split("/")[-1], "rows": t.num_rows, "excl_fh": int(pc.sum(ex).as_py()),
           "excl_oct": int(pc.sum(pc.and_(pc.invert(ex), oc)).as_py()), "scanned": kept.num_rows, "str_pattern_rows": {}}
    for c in kept.column_names:
        typ = kept.schema.field(c).type
        if pa.types.is_string(typ) or pa.types.is_large_string(typ):
            arr = kept.column(c).combine_chunks()
            for pn, pat in PATTERNS.items():
                n = int(pc.sum(pc.fill_null(pc.match_substring_regex(arr, pat), False)).as_py() or 0)
                if n:
                    res["str_pattern_rows"][f"{c}:{pn}"] = n
        if pa.types.is_list(typ) or pa.types.is_large_list(typ):
            arr = kept.column(c).combine_chunks()
            flat = pc.list_flatten(arr)
            for pn, pat in PATTERNS.items():
                n = int(pc.sum(pc.fill_null(pc.match_substring_regex(flat, pat), False)).as_py() or 0)
                if n:
                    res["str_pattern_rows"][f"{c}[elements]:{pn}"] = n
    print(json.dumps(res))
