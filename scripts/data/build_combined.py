"""Build ONE combined tome folder from several tome collections (2024 and 2026 eras, all 2026 batches).

Layout is identical to a single dsdk tome, so the model scripts only need their TOME / SUFFIX constants changed:

    tomes/combined/tome/csds/header/dataframe_*, keyset_00000
    tomes/combined/tome/csds/basic_<channel>.COMBINED/dataframe_*, keyset_00000

Every channel page is read from its source collection, cast to one canonical schema (rules below) and re-written
as its own page. No row is dropped; values are only cast (dtypes unified, missing columns filled with nulls, the documented sentinel mapping
of BOOL_MAP for one column, 0/1 integers or floats to Boolean) and round_end gains the player_count_norm column.
The dsdk `tome` metadata file is not written (the store is read by the project's own scripts, not by dsdk).

Canonical dtype rules (anything else raises, nothing is guessed):
  string / large_string                -> String
  bool + int*                          -> Boolean   (every int value must be 0, 1 or null, else raise)
  several int widths                   -> Int64
  int + float/double                   -> Float64
  null + X                             -> X
  column missing in a source           -> filled with nulls of the canonical dtype

This script rebuilds the whole store. To ADD new sources to an existing store without rebuilding it, use append_combined.py
(it appends pages, and rebuilds only the channels whose schema genuinely changes, with build_channel() below).

  pixi run python -I -X utf8 scripts\\data\\build_combined.py [--out tomes\\combined] [--sources name1,name2,...]
"""
import argparse
import collections
import glob
import json
import os
import shutil
import sys
import time

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
TOMES = os.path.join(ROOT, "tomes")
SUFFIX = ".COMBINED"
INTS = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
FLOATS = {"float", "double"}
STRS = {"string", "large_string"}
PL_BY_ARROW = {
    "int8": pl.Int8, "int16": pl.Int16, "int32": pl.Int32, "int64": pl.Int64, "uint8": pl.UInt8,
    "uint16": pl.UInt16, "uint32": pl.UInt32, "uint64": pl.UInt64, "float": pl.Float32, "double": pl.Float64,
    "string": pl.String, "large_string": pl.String, "bool": pl.Boolean, "null": pl.Null,
}
# (name, era). A source other than the two vetted ones must carry a BUILD_OK marker from run_tome_queue.ps1.
DEFAULT_SOURCES = [("2024-02-g", "2024"), ("2026-09", "2026"), ("2026-10-01_to_08", "2026"), ("2026-10-09", "2026"),
                   ("2026-10-10", "2026"), ("2026-10-11", "2026"), ("2026-10-12", "2026"), ("2026-10-13", "2026")]
VETTED = {"2024-02-g", "2026-10-01_to_08"}
# Documented sentinel conventions, checked on the data: molotov_state.was_extinguished_by_smoke is int64 {-1, 1} in 2024
# (346,057 and 11,425 rows), int64 {0, 1} in the 2026-09 collection, and bool in 2026-10. -1 and 0 mean not extinguished,
# 1 means extinguished.
BOOL_MAP = {("molotov_state", "was_extinguished_by_smoke"): {-1: False, 0: False, 1: True}}


def canon(arrow_types):
    s = set(arrow_types)
    nonnull = s - {"null"}
    if not nonnull:
        return pl.Null
    if nonnull <= STRS:
        return pl.String
    if "bool" in nonnull and (nonnull - {"bool"}) <= INTS | FLOATS:
        return pl.Boolean
    if nonnull <= INTS:
        return PL_BY_ARROW[next(iter(nonnull))] if len(nonnull) == 1 else pl.Int64
    if nonnull <= INTS | FLOATS and nonnull & FLOATS:
        return pl.Float64
    if len(nonnull) == 1 and next(iter(nonnull)) in PL_BY_ARROW:
        return PL_BY_ARROW[next(iter(nonnull))]
    raise SystemExit(f"no canonical rule for dtype combination {sorted(s)}")


def list_channels(coll):
    base = os.path.join(TOMES, coll, "tome", "csds")
    out = {}
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if d.startswith(".") or not os.path.isdir(p):
            continue
        ch = "header" if d == "header" else d[len("basic_"):].split(".")[0]
        out[ch] = {"pages": sorted(glob.glob(os.path.join(p, "dataframe_*"))),
                   "keysets": sorted(glob.glob(os.path.join(p, "keyset_*")))}
    return out


def resolve_sources(want):
    """Collection names -> [(source name, era)], expanding a day built in chunks. Shared with append_combined.py."""
    era_of = dict(DEFAULT_SOURCES)
    sources = []
    for base_name in want:
        # a day built in chunks (run_tome_queue.ps1) has tomes/<name>.CHUNKS_DONE listing tomes/<name>_cNN; every chunk is a source
        chunks_file = os.path.join(TOMES, base_name + ".CHUNKS_DONE")
        names = [ln.strip() for ln in open(chunks_file, encoding="utf-8") if ln.strip()] if os.path.exists(chunks_file) else [base_name]
        for name in names:
            d = os.path.join(TOMES, name)
            from_chunks = names != [base_name]
            if not os.path.isdir(os.path.join(d, "tome")):
                if from_chunks:  # CHUNKS_DONE promises every chunk is built: a missing one means something went wrong; never drop its rows silently
                    raise SystemExit(f"{base_name}.CHUNKS_DONE lists {name}, which has no tome folder: refusing to build a combined store without it")
                print(f"skip {name}: not built")
                continue
            if name not in VETTED and not os.path.exists(os.path.join(d, "BUILD_OK")):
                if from_chunks:
                    raise SystemExit(f"{base_name}.CHUNKS_DONE lists {name}, which has no BUILD_OK marker: refusing to build a combined store without it")
                print(f"skip {name}: no BUILD_OK marker (build unfinished)")
                continue
            sources.append((name, era_of.get(base_name, "2026")))
    return sources


def convert_page(df, ch, order, canon_t, fname):
    """One page cast to the channel's canonical schema (columns in `order`, dtypes in `canon_t`). Shared by the full build and by
    append_combined.py, so the two paths cannot drift apart."""
    exprs = []
    for n in order:
        t = canon_t[n]
        if n not in df.columns:
            exprs.append(pl.lit(None, dtype=t).alias(n))
            continue
        if df[n].dtype == t:
            exprs.append(pl.col(n))
        elif t == pl.Boolean and df[n].dtype not in (pl.Boolean, pl.Null):
            mp = BOOL_MAP.get((ch, n))
            if mp is not None:
                bad = df.filter(pl.col(n).is_not_null() & ~pl.col(n).is_in(list(mp))).height
                if bad:
                    raise SystemExit(f"{ch}.{n}: {bad} values outside {sorted(mp)} ({fname})")
                exprs.append(pl.col(n).replace_strict(mp, default=None, return_dtype=pl.Boolean).alias(n))
                continue
            src = pl.col(n).fill_nan(None) if df[n].dtype in (pl.Float32, pl.Float64) else pl.col(n)
            bad = df.filter(src.is_not_null() & (src != 0) & (src != 1)).height
            if bad:
                raise SystemExit(f"{ch}.{n}: {bad} non-0/1 values cannot become Boolean ({fname})")
            exprs.append(src.cast(pl.Boolean).alias(n))
        else:  # includes an all-null (Null dtype) column: casting it keeps the nulls
            exprs.append(pl.col(n).cast(t))
    out = df.select(exprs)
    if ch == "round_end":
        # 2024 logs player_count at twice the 2026 scale (20 for 5v5, 8 for wingman, 19/18 with a missing
        # player). player_count_norm puts both eras on the 2026 scale (10 / 4 / 9); the raw column is kept.
        out = out.with_columns(
            pl.when(pl.col("match_key").str.contains("^csds/2024/"))
            .then(pl.col("player_count") // 2).otherwise(pl.col("player_count")).alias("player_count_norm"))
    assert out.height == df.height
    return out


def build_channel(ch, sources, per, folder):
    """Write one channel of the combined store (all pages of all `sources`, cast to the canonical schema) into the NEW folder `folder`.
    Returns the ledger entry for the channel, or None when no source has a column for it."""
    # 1. canonical schema from footers of every page of every source
    types = collections.defaultdict(set)
    order = []
    for name, _ in sources:
        for f in per[name].get(ch, {}).get("pages", []):
            sch = pq.read_schema(f)
            if len(sch.names) == 0:
                continue  # empty 0-column page (e.g. 2024 bot_takeover)
            for n, t in zip(sch.names, sch.types):
                if n not in types:
                    order.append(n)
                types[n].add(str(t))
    if not order:
        print(f"{ch}: no columns in any source, skipped")
        return None
    canon_t = {n: canon(types[n]) for n in order}
    os.makedirs(folder)
    page = 0
    rows_by_src = {}
    keys = set()
    for name, _ in sources:
        rows_by_src[name] = 0
        for f in per[name].get(ch, {}).get("pages", []):
            if len(pq.read_schema(f).names) == 0:
                continue
            df = pl.read_parquet(f)
            out = convert_page(df, ch, order, canon_t, f)
            out.write_parquet(os.path.join(folder, f"dataframe_{page:05d}"), compression="zstd")
            page += 1
            rows_by_src[name] += df.height
        for k in per[name].get(ch, {}).get("keysets", []):
            keys.update(pl.read_parquet(k)["_"].to_list())
    pl.DataFrame({"_": sorted(keys)}).write_parquet(os.path.join(folder, "keyset_00000"), compression="zstd")
    first_page = os.path.join(folder, "dataframe_00000")
    n_cols = len(pq.read_schema(first_page).names) if page else len(order)  # the number of columns actually written (round_end gains player_count_norm)
    info = {"pages": page, "rows_by_source": rows_by_src, "keys": len(keys), "columns": n_cols,
            "cast_columns": sorted(n for n in order if len(types[n]) > 1)}
    print(f"{ch:<18} pages={page:>3} rows={sum(rows_by_src.values()):>10} keys={len(keys):>6} cast={info['cast_columns']}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(TOMES, "combined"))
    ap.add_argument("--sources", default="")
    args = ap.parse_args()
    args.out = os.path.normpath(args.out)  # a trailing separator would put the work folder INSIDE the store
    want = [s for s in args.sources.split(",") if s] or [n for n, _ in DEFAULT_SOURCES]
    sources = resolve_sources(want)
    print("sources:", sources)
    per = {n: list_channels(n) for n, _ in sources}
    channels = sorted(set().union(*[set(v) for v in per.values()]))

    tmp = args.out + ".building"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    base = os.path.join(tmp, "tome", "csds")
    os.makedirs(base)
    manifest = {"sources": sources, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "channels": {}}
    t0 = time.time()
    for ch in channels:
        folder = os.path.join(base, "header" if ch == "header" else f"basic_{ch}{SUFFIX}")
        info = build_channel(ch, sources, per, folder)
        if info is not None:
            manifest["channels"][ch] = info
    with open(os.path.join(tmp, "COMBINED_BUILD.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    old = args.out + ".replaced"
    if os.path.exists(args.out):
        if not os.path.exists(os.path.join(args.out, "COMBINED_BUILD.json")):
            raise SystemExit(f"{args.out} exists and is not a combined store; refusing to replace it")
        if os.path.exists(old):
            shutil.rmtree(old)
        os.replace(args.out, old)  # the old store is set aside first and removed only after the new one is in place: a failure leaves one intact store
    os.replace(tmp, args.out)
    shutil.rmtree(old, ignore_errors=True)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
