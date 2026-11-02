"""Are two tome collections the same data? (the check for build_tomes_incremental.py: extend / resume must equal a full build)

  python -I -X utf8 scripts\\data\\compare_tome_collections.py <collection A> <collection B> [--ds-type csds] [--channels a,b] [--list-only]

A collection is a folder that holds tome/<ds_type>/<tome name>/ (header + basic_<channel>[.<dates>]). Tomes are matched by CHANNEL, not by folder
name (basic_player_death and basic_player_death.2026-10-08,2026-10-12 are the same channel), and each is read with the dsdk loader, so the
comparison is of what a consumer reads, whatever the page boundaries are. For the header and every channel it checks

  - the same tomes (channels) in both, each complete
  - the same columns in the same order, with the same dtypes (pandas dtypes of the loaded tome: int8, Int64, float32, object, ...)
  - the same row count and the same SET of match keys (the key order is reported as information)
  - the same SUM of per-row hashes (pandas.util.hash_pandas_object over the columns sorted by name, wrapped to 64 bits), which does not depend on
    row order or on where the pages are cut: two tomes with equal sums hold the same rows
  - the same for the header copy that every channel tome carries (its keys and row hashes)

Whole tomes are loaded one at a time: use it on samples or on tomes that fit in memory (the per-tick channels of a big day do not).
Exit code 0 = the same, 1 = a difference (listed).
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
from pureskillgg_dsdk.tome.loader import TomeLoader  # noqa: E402  (after the logger is configured)
from pureskillgg_dsdk.tome.reader_fs import TomeReaderFs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def tomes_of(root, ds_type):
    """channel label -> tome folder name, for the tomes of a collection (the header is labelled 'header')."""
    base = os.path.join(root, "tome", ds_type)
    if not os.path.isdir(base):
        sys.exit(f"{root}: no tome/{ds_type} folder")
    out = {}
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        manifest_path = os.path.join(path, "tome")
        if name.startswith(".") or not os.path.isfile(manifest_path):
            continue
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        label = "header" if manifest.get("isHeader") else (name[len("basic_"):] if name.startswith("basic_") else name).split(".")[0]
        if label in out:
            sys.exit(f"{root}: two tomes for channel {label!r}: {out[label]} and {name}")
        out[label] = name
    return out


def load(root, ds_type, name):
    return TomeLoader(reader=TomeReaderFs(root_path=root, manifest_key=f"tome/{ds_type}/{name}/tome"))


def row_hash_sum(df):
    """Sum of the per-row hashes, wrapped to 64 bits: independent of row order and of the pages."""
    if len(df) == 0:
        return 0
    hashes = pd.util.hash_pandas_object(df[sorted(df.columns)], index=False).to_numpy(dtype=np.uint64)
    return int(np.sum(hashes, dtype=np.uint64))


def summary(loader, keys=None):
    df = loader.get_dataframe()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "keys": loader.get_keyset() if keys is None else keys,
        "hash": row_hash_sum(df),
        "pages": len(loader.manifest["pages"]),
        "complete": bool(loader.manifest.get("isComplete")),
    }


def diff_summaries(label, a, b, diffs, info):
    if not (a["complete"] and b["complete"]):
        diffs.append(f"{label}: not complete in both (A {a['complete']}, B {b['complete']})")
    if a["columns"] != b["columns"]:
        if set(a["columns"]) == set(b["columns"]):
            diffs.append(f"{label}: same columns in another order")
        else:
            diffs.append(f"{label}: columns differ: only A {sorted(set(a['columns']) - set(b['columns']))}, only B {sorted(set(b['columns']) - set(a['columns']))}")
    for c in sorted(set(a["dtypes"]) & set(b["dtypes"])):
        if a["dtypes"][c] != b["dtypes"][c]:
            diffs.append(f"{label}.{c}: dtype {a['dtypes'][c]} (A) vs {b['dtypes'][c]} (B)")
    if a["rows"] != b["rows"]:
        diffs.append(f"{label}: rows {a['rows']} (A) vs {b['rows']} (B)")
    ka, kb = a["keys"], b["keys"]
    if len(ka) != len(set(ka)) or len(kb) != len(set(kb)):
        info.append(f"{label}: duplicate keys in a keyset (A {len(ka) - len(set(ka))}, B {len(kb) - len(set(kb))})")
    if set(ka) != set(kb):
        diffs.append(f"{label}: key sets differ: {len(set(ka) - set(kb))} only in A, {len(set(kb) - set(ka))} only in B")
    elif ka != kb:
        info.append(f"{label}: same keys in another order")
    if a["rows"] == b["rows"] and a["hash"] != b["hash"]:
        diffs.append(f"{label}: row hashes differ (sum A {a['hash']:#018x}, B {b['hash']:#018x}): the values are not the same")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="tome collection A (the folder that holds tome/<ds_type>/)")
    ap.add_argument("b", help="tome collection B")
    ap.add_argument("--ds-type", default="csds")
    ap.add_argument("--channels", help="comma-separated channels to compare (default: all of them; the header is always compared)")
    ap.add_argument("--list-only", action="store_true", help="only list the tomes of both collections")
    args = ap.parse_args()

    ta, tb = tomes_of(args.a, args.ds_type), tomes_of(args.b, args.ds_type)
    diffs, info = [], []
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    if only_a:
        diffs.append(f"channels only in A: {only_a}")
    if only_b:
        diffs.append(f"channels only in B: {only_b}")
    print(f"A {args.a}: {len(ta)} tomes   B {args.b}: {len(tb)} tomes   in both: {len(set(ta) & set(tb))}", flush=True)
    if args.list_only:
        for label in sorted(set(ta) | set(tb)):
            print(f"  {label:<24} A={ta.get(label, '-')}   B={tb.get(label, '-')}")
        return
    wanted = None if not args.channels else {c.strip() for c in args.channels.split(",") if c.strip()} | {"header"}
    t0 = time.time()
    total_rows = 0
    for label in sorted(set(ta) & set(tb)):
        if wanted is not None and label not in wanted:
            continue
        la, lb = load(args.a, args.ds_type, ta[label]), load(args.b, args.ds_type, tb[label])
        sa, sb = summary(la), summary(lb)
        before = len(diffs)
        diff_summaries(label, sa, sb, diffs, info)
        if label != "header":  # the header copy each channel tome carries
            ha, hb = summary(la.header), summary(lb.header)
            diff_summaries(f"{label} (header copy)", ha, hb, diffs, info)
            if set(sa["keys"]) != set(ha["keys"]):
                diffs.append(f"{label}: A's header copy lists other matches than the tome")
            if set(sb["keys"]) != set(hb["keys"]):
                diffs.append(f"{label}: B's header copy lists other matches than the tome")
        total_rows += sa["rows"]
        status = "same" if len(diffs) == before else f"{len(diffs) - before} DIFFERENCE(S)"
        print(f"  {label:<20} rows={sa['rows']:>9,} cols={len(sa['columns']):>3} keys={len(set(sa['keys'])):>5} pages A/B={sa['pages']}/{sb['pages']}  {status}", flush=True)
    reordered = [i.split(":")[0] for i in info if i.endswith("same keys in another order")]
    for i in info:
        if not i.endswith("same keys in another order"):
            print("info:", i)
    if reordered:
        print(f"info: {len(reordered)} keysets list the same keys in another order than B (matches were appended, not sorted)")
    if diffs:
        print(f"\nDIFFERENT: {len(diffs)} difference(s)")
        for d in diffs[:80]:
            print(" -", d)
        sys.exit(1)
    print(f"\nSAME: {len(set(ta) & set(tb))} tomes, {total_rows:,} rows, same columns and dtypes, same key sets, same row-hash sums ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
