"""Read-only PII value screen, training/validation parquet only.

Hard rules implemented here:
- Frozen-test keys, holdout keys and any match whose key date is Oct 26-31 are
  excluded from every row-level read. Exclusion uses the match key only.
- Files named holdout, frozen or test, and day-named files dated Oct 26-31, are
  never opened for values (schema-only, handled by pii_inventory.py).
- Only counts are written. No raw values are printed or written.

Usage: python.exe -I value_screen.py <out_jsonl> [substring_filter]
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
INV = SCR + "/inventory.json"
OUT = sys.argv[1]
FILTER = sys.argv[2] if len(sys.argv) > 2 else None

REPO_KEYS_FROZEN = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/benchmark/frozen_test_keys.json"
HOLDOUT = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/combined/holdout_22_25_test_keys.json"

STEAMID64_LO = 76561190000000000
STEAMID64_HI = 76561199999999999
UUIDISH = re.compile(r"^[a-z]{4}/\d{4}/\d{2}/\d{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/[a-z]{4}$")
DATE_OCT = r"/\d{4}/10/(2[6-9]|3[01])/"
PATTERNS = {
    "steam2_STEAM_x:y:z": r"STEAM_[0-5]:[01]:[0-9]+",
    "steam3_[U:1:n]": r"\[U:1:[0-9]+\]",
    "ipv4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "win_user_path": r"[A-Za-z]:\\+Users\\+",
    "steamcommunity": r"steamcommunity",
    "digits_9plus": r"[0-9]{9,}",
}
ID_NAME = re.compile(r"(player|attacker|victim|steam|account|user|sid|owner|_id$|^id$|mk_id|^rid$)", re.I)
TEST_NAME = re.compile(r"test|frozen|holdout", re.I)
DAY_OCT_NAME = re.compile(r"2026-10-(2[6-9]|3[01])")


def load_keys():
    frozen = json.load(open(REPO_KEYS_FROZEN, encoding="utf-8"))
    hd = json.load(open(HOLDOUT, encoding="utf-8"))
    hold = hd["match_keys"] if isinstance(hd, dict) else hd
    s = set(frozen) | set(hold)
    # Only counts are reported for these sets.
    return s, len(set(frozen)), len(set(hold))


def bool_on_values(arr, fn):
    """Apply a boolean string predicate to a (possibly dictionary-encoded) array."""
    if pa.types.is_dictionary(arr.type):
        dmask = fn(arr.dictionary)
        return pc.take(dmask, arr.indices)
    return fn(arr)


def count_true(mask):
    return int(pc.sum(pc.fill_null(mask, False)).as_py() or 0)


def main():
    inv = json.load(open(INV, encoding="utf-8"))
    excl_set, n_frozen, n_hold = load_keys()
    excl_arr = pa.array(sorted(excl_set), type=pa.string())
    print(f"exclusion keys: frozen={n_frozen} holdout={n_hold} union={len(excl_set)}", flush=True)

    files = [r for r in inv if r.get("kind") == "parquet" and "error" not in r]
    if FILTER:
        files = [r for r in files if FILTER in r["path"]]

    results = []
    skipped = collections.Counter()
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as out:
        for i, r in enumerate(files):
            p = r["path"].replace("\\", "/")
            base = os.path.basename(p)
            low = p.lower()
            if "holdout" in low:
                skipped["holdout_path_schema_only"] += 1
                continue
            if TEST_NAME.search(base) or any(TEST_NAME.search(seg) for seg in p.split("/")[-3:-1]):
                skipped["test_or_frozen_named_schema_only"] += 1
                continue
            if DAY_OCT_NAME.search(base):
                skipped["day_file_oct26_31_schema_only"] += 1
                continue
            cols = [c for c, _ in r["columns"]]
            key_col = next((c for c in ("match_key", "match", "key") if c in cols), None)
            if key_col is None:
                skipped["no_match_key_schema_only"] += 1
                continue

            types = {c: str(t) for c, t in r["columns"]}
            str_cols = [c for c in cols if c != key_col and ("string" in types[c] or "dictionary" in types[c] and "str" in types[c])]
            int64_cols = [c for c in cols if types[c] in ("int64", "uint64")]
            id_cols = [c for c in cols if ID_NAME.search(c) and types[c] in ("int64", "uint64", "int32", "uint32", "int16", "uint16", "int8", "uint8")]
            NUM_COLS = list(dict.fromkeys(int64_cols + id_cols))
            read_cols = list(dict.fromkeys([key_col] + str_cols + NUM_COLS))

            rec = {"path": p, "rows_total": r["rows"], "key_col": key_col,
                   "n_frozen_holdout_rows": 0, "n_oct26_31_rows": 0, "rows_scanned": 0,
                   "key_distinct": set(), "key_nonuuid_distinct": 0,
                   "str": {c: {"n_distinct": set(), "pattern_rows": collections.Counter(), "rows": 0} for c in str_cols},
                   "num": {c: {"steamid64_range_rows": 0, "ge_1e7_rows": 0, "distinct": set(), "non_null": 0} for c in int64_cols + id_cols}}
            try:
                pf = pq.ParquetFile(p, read_dictionary=[c for c in read_cols if c == key_col or c in str_cols])
                for batch in pf.iter_batches(batch_size=2_000_000, columns=read_cols):
                    tbl = pa.Table.from_batches([batch])
                    kcol = tbl.column(key_col).combine_chunks()
                    in_excl = bool_on_values(kcol, lambda a: pc.is_in(a, value_set=excl_arr))
                    in_oct = bool_on_values(kcol, lambda a: pc.match_substring_regex(a, DATE_OCT))
                    in_excl = pc.fill_null(in_excl, False)
                    in_oct = pc.fill_null(in_oct, False)
                    drop = pc.or_(in_excl, in_oct)
                    rec["n_frozen_holdout_rows"] += count_true(in_excl)
                    rec["n_oct26_31_rows"] += count_true(pc.and_(pc.invert(in_excl), in_oct))
                    keep = pc.invert(drop)
                    rec["rows_scanned"] += count_true(keep)
                    kept = tbl.filter(keep)
                    if kept.num_rows == 0:
                        continue
                    kc = kept.column(key_col)
                    kc_vals = pc.unique(kc.chunk(0) if kc.num_chunks == 1 else kc.combine_chunks())
                    if pa.types.is_dictionary(kc_vals.type):
                        kc_vals = kc_vals.dictionary_decode() if hasattr(kc_vals, "dictionary_decode") else pc.take(kc_vals.dictionary, kc_vals.indices)
                    for v in kc_vals.to_pylist():
                        if v is None:
                            continue
                        rec["key_distinct"].add(v)
                    for c in str_cols:
                        arr = kept.column(c).combine_chunks()
                        arr_nn = arr
                        rec["str"][c]["rows"] += len(arr) - arr.null_count
                        if pa.types.is_dictionary(arr.type):
                            present = pc.unique(arr.indices)
                            vals = arr.dictionary.take(present.drop_null())
                        else:
                            vals = pc.unique(arr).drop_null()
                        for v in vals.to_pylist():
                            rec["str"][c]["n_distinct"].add(v)
                        for pname, pat in PATTERNS.items():
                            m = bool_on_values(arr, lambda a, pat=pat: pc.match_substring_regex(a, pat, ignore_case=False))
                            rec["str"][c]["pattern_rows"][pname] += count_true(m)
                    for c in NUM_COLS:
                        arr = kept.column(c).combine_chunks()
                        arr64 = arr if arr.type == pa.int64() else pc.cast(arr, pa.int64(), safe=False)
                        nn = arr64.drop_null() if arr64.null_count else arr64
                        rec["num"][c]["non_null"] += len(nn)
                        rec["num"][c]["steamid64_range_rows"] += count_true(
                            pc.and_(pc.greater_equal(arr64, STEAMID64_LO), pc.less_equal(arr64, STEAMID64_HI)))
                        rec["num"][c]["ge_1e7_rows"] += count_true(pc.greater_equal(arr64, 10_000_000))
                        if c in id_cols:
                            rec["num"][c]["distinct"].update(pc.unique(nn).to_pylist())
            except Exception as e:  # noqa
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"

            # Serialise: counts only, no raw values.
            key_nonuuid = sum(1 for k in rec["key_distinct"] if not UUIDISH.match(k))
            summary = {
                "path": rec["path"], "rows_total": rec["rows_total"], "key_col": rec["key_col"],
                "rows_scanned": rec["rows_scanned"],
                "n_frozen_holdout_rows_excluded": rec["n_frozen_holdout_rows"],
                "n_oct26_31_rows_excluded": rec["n_oct26_31_rows"],
                "key_distinct_count": len(rec["key_distinct"]),
                "key_nonuuid_shape_count": key_nonuuid,
                "str_cols": {c: {"n_distinct": len(v["n_distinct"]), "rows": v["rows"],
                                  "pattern_rows": {k: int(n) for k, n in v["pattern_rows"].items() if n}}
                             for c, v in rec["str"].items()},
                "num_cols": {c: {"non_null": v["non_null"], "steamid64_range_rows": v["steamid64_range_rows"],
                                  "ge_1e7_rows": v["ge_1e7_rows"], "distinct_count": len(v["distinct"]) if v["distinct"] else None}
                             for c, v in rec["num"].items()},
            }
            if "error" in rec:
                summary["error"] = rec["error"]
            out.write(json.dumps(summary) + "\n")
            out.flush()
            results.append(summary)
            if (i + 1) % 25 == 0:
                print(f"progress {i+1}/{len(files)} elapsed={time.time()-t0:.0f}s", flush=True)

    print("skipped:", dict(skipped), flush=True)
    print("scanned files:", len(results), "elapsed=%.0fs" % (time.time() - t0), flush=True)
    errs = [s["path"][-80:] for s in results if "error" in s]
    print("files with errors:", len(errs), errs[:10], flush=True)


if __name__ == "__main__":
    main()
