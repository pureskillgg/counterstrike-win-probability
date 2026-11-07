"""Counts-only screen of raw per-match personal tables under E:/csds_final.

Exclusion is decided from the folder path (csds/YYYY/MM/DD/<uuid>) before any file is
opened: frozen, holdout and Oct 26-31 matches are skipped unopened. Only counts,
distinct-counts and shape summaries are printed. Raw values are never printed.
"""
import os
import re
import json
import sys
import time
import collections
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = "E:/csds_final/csds"
REPO_KEYS_FROZEN = "C:/Users/<user>/Documents/pureskill_internship/csds-cs2/data/benchmark/frozen_test_keys.json"
HOLDOUT = "C:/Users/<user>/Documents/pureskill_internship/csds-cs2/data/combined/holdout_22_25_test_keys.json"
PATS = {"steamid64": r"7656119[0-9]{10}", "steam2": r"STEAM_[0-5]:[01]:[0-9]+", "steam3": r"\[U:1:[0-9]+\]",
        "ipv4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "win_user_path": r"[A-Za-z]:\\+Users\\+", "steamcommunity": r"steamcommunity", "url": r"https?://"}
COLS = {
    "player_personal": ["name", "clan_tag", "steam_id"],
    "player_name": ["name_new", "name_old"],
    "player_chat": ["text"],
    "header": ["client_name", "server_name", "demo_id", "sharecode"],
}
KEY_FMT = re.compile(r"^csds/[0-9]{4}/[0-9]{2}/[0-9]{2}/[0-9a-f-]{36}/csds$")


def shape(v):
    s = re.sub(r"[0-9]", "9", v); s = re.sub(r"[a-z]", "a", s); s = re.sub(r"[A-Z]", "A", s)
    return re.sub(r"(.)\1{3,}", r"\1{4+}", s)[:40]


def main():
    frozen = set(json.load(open(REPO_KEYS_FROZEN, encoding="utf-8")))
    hd = json.load(open(HOLDOUT, encoding="utf-8"))
    excl = frozen | set(hd["match_keys"])
    print("exclusion keys in expected format (count only):", sum(1 for k in excl if KEY_FMT.match(k)), "of", len(excl), flush=True)
    st = collections.defaultdict(lambda: {"matches": 0, "rows": 0, "cols": collections.defaultdict(lambda: {"non_empty": 0, "distinct": set(), "pats": collections.Counter()})})
    skipped = collections.Counter()
    t0 = time.time()
    n_m = 0
    for y in sorted(os.listdir(ROOT)):
        yp = os.path.join(ROOT, y)
        if not os.path.isdir(yp):
            continue
        for mth in sorted(os.listdir(yp)):
            mp = os.path.join(yp, mth)
            for dd in sorted(os.listdir(mp)):
                dp = os.path.join(mp, dd)
                for uu in os.listdir(dp):
                    up = os.path.join(dp, uu)
                    key = f"csds/{y}/{mth}/{dd}/{uu}/csds"
                    if key in excl:
                        skipped["frozen_or_holdout_by_key_unopened"] += 1
                        continue
                    if mth == "10" and dd in ("26", "27", "28", "29", "30", "31"):
                        skipped["oct26_31_unopened"] += 1
                        continue
                    n_m += 1
                    for tb, cols in COLS.items():
                        f = os.path.join(up, tb)
                        if not os.path.isfile(f):
                            skipped[f"missing_{tb}"] += 1
                            continue
                        tab = pq.read_table(f)
                        a = st[tb]
                        a["matches"] += 1
                        a["rows"] += tab.num_rows
                        for c in cols:
                            if c not in tab.column_names:
                                continue
                            col = tab.column(c).combine_chunks()
                            if pa.types.is_null(col.type):
                                continue
                            cc = a["cols"][c]
                            cc["non_empty"] += int(pc.sum(pc.fill_null(pc.greater(pc.utf8_length(col), 0), False)).as_py() or 0)
                            vals = pc.unique(col).drop_null().to_pylist()
                            cc["distinct"].update(vals)
                            for pn, pat in PATS.items():
                                cc["pats"][pn] += int(pc.sum(pc.fill_null(pc.match_substring_regex(col, pat), False)).as_py() or 0)
                    if n_m % 2000 == 0:
                        print(f"progress matches_opened={n_m} elapsed={time.time()-t0:.0f}s", flush=True)
    print("matches opened:", n_m, "skipped:", dict(skipped), "elapsed=%.0fs" % (time.time() - t0), flush=True)
    for tb, a in st.items():
        print(f"== {tb}: matches={a['matches']} rows={a['rows']}")
        for c, cc in a["cols"].items():
            shp = collections.Counter(shape(v) for v in cc["distinct"])
            print(f"   {c}: non_empty={cc['non_empty']} distinct={len(cc['distinct'])} patterns={ {k: v for k, v in cc['pats'].items() if v} }")
            for s_, n in shp.most_common(3):
                print(f"        shape(n={n}): {s_}")


if __name__ == "__main__":
    main()
