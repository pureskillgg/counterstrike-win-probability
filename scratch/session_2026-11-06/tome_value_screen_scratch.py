"""Value screen for PII-bearing raw tome tables. Counts only, no raw values printed.

Same hard rules as value_screen.py: frozen and holdout match keys and any
match dated Oct 26-31 are dropped before anything is counted. Folders and
tables whose names contain test or frozen markers, or Oct 26-31 dates, are
schema-only. The key filter runs on match_key only.

Usage: python.exe -I tome_value_screen.py <out_jsonl>
"""
import json
import os
import re
import sys
import time
import collections

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
T = "G:/pureskill/scratch"
OUT = sys.argv[1]
REPO_KEYS_FROZEN = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/benchmark/frozen_test_keys.json"
HOLDOUT = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/combined/holdout_22_25_test_keys.json"
DATE_OCT = r"/[0-9]{4}/10/(2[6-9]|3[01])/"
SCHEMA_ONLY_FOLDER = re.compile(r"test|holdout|2026-10-(2[6-9]|3[01])|_bad_pages", re.I)
TARGET_TABLES = {
    "basic_player_personal": ["steam_id", "name", "clan_tag"],
    "basic_player_name": ["name_new", "name_old"],
    "basic_player_chat": ["text"],
    "header": ["server_name", "client_name", "map_name", "demo_id", "sharecode", "match_date", "game_directory", "addons"],
    "header.new": ["server_name", "client_name", "map_name", "demo_id", "sharecode", "match_date", "game_directory", "addons", "unique_steamids"],
    "basic_player_info": [],
    "basic_player_connect": [],
}
ID_COLS = ["player_id", "player_id_pawn", "player_controller_id", "player_index", "attacker_id"]
STEAMID64_LO = 76561190000000000
STEAMID64_HI = 76561199999999999
PATTERNS = {
    "steamid64_17digit_7656119": r"7656119[0-9]{10}",
    "steam2_STEAM_x:y:z": r"STEAM_[0-5]:[01]:[0-9]+",
    "steam3_[U:1:n]": r"\[U:1:[0-9]+\]",
    "ipv4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "win_user_path": r"[A-Za-z]:\\+Users\\+",
    "steamcommunity": r"steamcommunity",
    "url": r"https?://",
    "digits_9plus": r"[0-9]{9,}",
}


def load_excl():
    frozen = json.load(open(REPO_KEYS_FROZEN, encoding="utf-8"))
    hd = json.load(open(HOLDOUT, encoding="utf-8"))
    hold = hd["match_keys"] if isinstance(hd, dict) else hd
    return set(frozen) | set(hold)


def bool_on_values(arr, fn):
    if pa.types.is_dictionary(arr.type):
        return pc.take(fn(arr.dictionary), arr.indices)
    return fn(arr)


def count_true(mask):
    return int(pc.sum(pc.fill_null(mask, False)).as_py() or 0)


def table_name(dirname):
    base = re.sub(r"\.[0-9]{4}-[0-9]{2}-[0-9]{2}.*$", "", dirname)
    if base == "header.new":
        return base
    return re.sub(r"\.COMBINED.*$", "", base)


def collect_files():
    out = []
    for folder in sorted(os.listdir(T)):
        fpath = os.path.join(T, folder)
        if not os.path.isdir(fpath):
            continue
        if SCHEMA_ONLY_FOLDER.search(folder):
            continue
        for dp, dns, fns in os.walk(fpath):
            parts = dp.replace("\\", "/").split("/")
            if "csds" not in parts:
                continue
            i = parts.index("csds")
            if len(parts) != i + 2:
                continue
            tname = table_name(parts[i + 1])
            if tname not in TARGET_TABLES:
                continue
            for fn in fns:
                if fn.startswith("dataframe_") and fn.endswith(".parquet") or (fn.startswith("dataframe_") and not fn.endswith(".json")):
                    out.append((folder, tname, os.path.join(dp, fn).replace("\\", "/")))
    return out


def main():
    excl = load_excl()
    excl_arr = pa.array(sorted(excl), type=pa.string())
    files = collect_files()
    print("candidate files:", len(files), flush=True)
    per = collections.defaultdict(lambda: {
        "files": 0, "rows_total": 0, "rows_excluded_frozen_holdout": 0, "rows_excluded_oct26_31": 0,
        "rows_scanned": 0, "key_nonuuid_or_missing": 0, "rows_key_without_date_prefix": 0,
        "str": collections.defaultdict(lambda: {"non_empty_rows": 0, "distinct": set(), "patterns": collections.Counter()}),
        "num": collections.defaultdict(lambda: {"non_null": 0, "steam_range": 0, "ge_1e7": 0, "distinct": set()}),
        "errors": 0})
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as out:
        for n, (folder, tname, p) in enumerate(files):
            rec = per[tname]
            rec["files"] += 1
            try:
                sch = pq.read_schema(p)
            except Exception:  # noqa
                rec["errors"] += 1
                continue
            names = sch.names
            key_col = "match_key" if "match_key" in names else ("key" if "key" in names else None)
            if key_col is None:
                rec["key_nonuuid_or_missing"] += 1
                continue
            types = {f.name: str(f.type) for f in sch}
            strs = [c for c in TARGET_TABLES[tname] if c in names and ("str" in types[c])]
            idc = [c for c in ID_COLS if c in names and types[c] in ("int64", "int32", "uint64", "int16", "int8", "uint8", "uint32", "uint16")]
            read_cols = list(dict.fromkeys([key_col] + strs + idc))
            try:
                pf = pq.ParquetFile(p, read_dictionary=[c for c in read_cols if "str" in types.get(c, "")])
                for batch in pf.iter_batches(batch_size=2_000_000, columns=read_cols):
                    tbl = pa.Table.from_batches([batch])
                    n_all = tbl.num_rows
                    rec["rows_total"] += n_all
                    kcol = tbl.column(key_col).combine_chunks()
                    in_excl = pc.fill_null(bool_on_values(kcol, lambda a: pc.is_in(a, value_set=excl_arr)), False)
                    in_oct = pc.fill_null(bool_on_values(kcol, lambda a: pc.match_substring_regex(a, DATE_OCT)), False)
                    rec["rows_excluded_frozen_holdout"] += count_true(in_excl)
                    rec["rows_excluded_oct26_31"] += count_true(pc.and_(pc.invert(in_excl), in_oct))
                    no_date = bool_on_values(kcol, lambda a: pc.invert(pc.match_substring_regex(a, r"^[a-z]{4}/[0-9]{4}/[0-9]{2}/[0-9]{2}/")))
                    rec["rows_key_without_date_prefix"] += count_true(no_date)
                    keep = pc.invert(pc.or_(in_excl, in_oct))
                    kept = tbl.filter(keep)
                    rec["rows_scanned"] += kept.num_rows
                    if kept.num_rows == 0:
                        continue
                    for c in strs:
                        arr = kept.column(c).combine_chunks()
                        if pa.types.is_dictionary(arr.type):
                            present = pc.unique(arr.indices).drop_null()
                            vals = arr.dictionary.take(present)
                        else:
                            vals = pc.unique(arr).drop_null()
                        s = rec["str"][c]
                        s["non_empty_rows"] += count_true(bool_on_values(arr, lambda a: pc.greater(pc.utf8_length(a), 0)))
                        s["distinct"].update(v for v in vals.to_pylist() if v)
                        for pname, pat in PATTERNS.items():
                            m = bool_on_values(arr, lambda a, pat=pat: pc.match_substring_regex(a, pat))
                            s["patterns"][pname] += count_true(m)
                    for c in idc:
                        arr = kept.column(c).combine_chunks()
                        arr64 = arr if arr.type == pa.int64() else pc.cast(arr, pa.int64(), safe=False)
                        nn = pc.drop_null(arr64)
                        ns = rec["num"][c]
                        ns["non_null"] += len(nn)
                        ns["steam_range"] += count_true(pc.and_(pc.greater_equal(arr64, STEAMID64_LO), pc.less_equal(arr64, STEAMID64_HI)))
                        ns["ge_1e7"] += count_true(pc.greater_equal(arr64, 10_000_000))
                        ns["distinct"].update(pc.unique(nn).to_pylist())
            except Exception as e:  # noqa
                rec["errors"] += 1
                out.write(json.dumps({"error_file": p.split("tomes/")[-1][:160], "error": f"{type(e).__name__}: {str(e)[:160]}"}) + "\n")
            if (n + 1) % 200 == 0:
                print(f"progress {n+1}/{len(files)} elapsed={time.time()-t0:.0f}s", flush=True)
        for tname, rec in per.items():
            summ = {
                "table": tname, "files": rec["files"], "rows_total": rec["rows_total"],
                "rows_excluded_frozen_holdout": rec["rows_excluded_frozen_holdout"],
                "rows_excluded_oct26_31": rec["rows_excluded_oct26_31"],
                "rows_scanned": rec["rows_scanned"], "rows_key_without_date_prefix": rec["rows_key_without_date_prefix"], "errors": rec["errors"],
                "str_cols": {c: {"non_empty_rows": v["non_empty_rows"], "distinct_count": len(v["distinct"]),
                                  "pattern_rows": {k: int(n) for k, n in v["patterns"].items() if n}}
                             for c, v in rec["str"].items()},
                "id_cols": {c: {"non_null": v["non_null"], "steamid64_range_rows": v["steam_range"],
                                 "ge_1e7_rows": v["ge_1e7"], "distinct_count": len(v["distinct"])}
                            for c, v in rec["num"].items()},
            }
            out.write(json.dumps({"summary": summ}) + "\n")
    print("done elapsed=%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
