"""Shape-only check of identifier-like string columns in eligible tome tables.

Prints, per column: number of distinct values, length histogram, and character
class shapes (letters->a/A, digits->9, other kept). Raw values are never printed.
Eligibility and exclusions are identical to tome_value_screen.py.
"""
import json
import os
import re
import collections
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

T = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/tomes"
REPO_KEYS_FROZEN = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/benchmark/frozen_test_keys.json"
HOLDOUT = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/combined/holdout_22_25_test_keys.json"
DATE_OCT = r"/[0-9]{4}/10/(2[6-9]|3[01])/"
SCHEMA_ONLY_FOLDER = re.compile(r"test|holdout|2026-10-(2[6-9]|3[01])|_bad_pages", re.I)
WANT = {
    "basic_player_personal": ["steam_id", "name", "clan_tag"],
    "basic_player_name": ["name_new", "name_old"],
    "basic_player_chat": ["text"],
    "header": ["client_name", "demo_id", "sharecode", "server_name"],
    "header.new": ["client_name", "demo_id", "sharecode", "server_name"],
}


def base_name(d):
    b = re.sub(r"\.[0-9]{4}-[0-9]{2}-[0-9]{2}.*$", "", d)
    if b == "header.new":
        return b
    return re.sub(r"\.COMBINED.*$", "", b)


def shape(v):
    out = []
    for ch in v:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha() and ch.islower():
            out.append("a")
        elif ch.isalpha():
            out.append("A")
        else:
            out.append(ch if ch in "_-.:/\\[]@ " else "?")
    s = "".join(out)
    return re.sub(r"(.)\1{3,}", r"\1{4+}", s)


def main():
    frozen = json.load(open(REPO_KEYS_FROZEN, encoding="utf-8"))
    hd = json.load(open(HOLDOUT, encoding="utf-8"))
    excl = set(frozen) | set(hd["match_keys"] if isinstance(hd, dict) else hd)
    excl_arr = pa.array(sorted(excl), type=pa.string())
    vals = collections.defaultdict(set)
    lens = collections.defaultdict(collections.Counter)
    shapes = collections.defaultdict(collections.Counter)
    rows = collections.Counter()
    for folder in sorted(os.listdir(T)):
        fpath = os.path.join(T, folder)
        if not os.path.isdir(fpath) or SCHEMA_ONLY_FOLDER.search(folder):
            continue
        for dp, dns, fns in os.walk(fpath):
            parts = dp.replace("\\", "/").split("/")
            if "csds" not in parts:
                continue
            i = parts.index("csds")
            if len(parts) != i + 2:
                continue
            tb = base_name(parts[i + 1])
            if tb not in WANT:
                continue
            for fn in fns:
                if not fn.startswith("dataframe_"):
                    continue
                p = os.path.join(dp, fn).replace("\\", "/")
                try:
                    names = pq.read_schema(p).names
                except Exception:  # noqa
                    continue
                key = "match_key" if "match_key" in names else ("key" if "key" in names else None)
                cols = [c for c in WANT[tb] if c in names]
                if key is None or not cols:
                    continue
                pf = pq.ParquetFile(p, read_dictionary=[key] + cols)
                for batch in pf.iter_batches(batch_size=2_000_000, columns=[key] + cols):
                    t = pa.Table.from_batches([batch])
                    kc = t.column(key).combine_chunks()
                    ex = pc.fill_null(
                        kc.dictionary.dictionary_encode().indices if False else pc.is_in(
                            kc.dictionary if pa.types.is_dictionary(kc.type) else kc, value_set=excl_arr), False)
                    if pa.types.is_dictionary(kc.type):
                        ex = pc.fill_null(pc.take(ex if False else pc.is_in(kc.dictionary, value_set=excl_arr), kc.indices), False)
                        oc = pc.fill_null(pc.take(pc.match_substring_regex(kc.dictionary, DATE_OCT), kc.indices), False)
                    else:
                        ex = pc.fill_null(pc.is_in(kc, value_set=excl_arr), False)
                        oc = pc.fill_null(pc.match_substring_regex(kc, DATE_OCT), False)
                    keep = pc.invert(pc.or_(ex, oc))
                    kept = t.filter(keep)
                    if kept.num_rows == 0:
                        continue
                    rows[tb] += kept.num_rows
                    for c in cols:
                        arr = kept.column(c).combine_chunks()
                        if pa.types.is_dictionary(arr.type):
                            present = pc.unique(arr.indices).drop_null()
                            u = arr.dictionary.take(present).to_pylist()
                        else:
                            u = pc.unique(arr).drop_null().to_pylist()
                        for v in u:
                            if v is None:
                                continue
                            vals[(tb, c)].add(v)
    # Report counts and shapes only. Raw values are not printed.
    for (tb, c), vs in sorted(vals.items()):
        lens_c = collections.Counter(len(v) for v in vs)
        shp = collections.Counter(shape(v) for v in vs)
        charset = collections.Counter("ascii" if all(ord(ch) < 128 for ch in v) else "non-ascii" for v in vs)
        print(f"{tb}.{c}: rows_scanned={rows[tb]} distinct={len(vs)} len_hist={dict(sorted(lens_c.items())[:8])} charset={dict(charset)}")
        for s, n in shp.most_common(4):
            print(f"    shape(n_distinct={n}): {s[:60]}")


if __name__ == "__main__":
    main()
