# verify_A1_3_tome.py
# Independent verifier for finding A1-3 (2026 tome, page-level schema drift).
# Recomputes: per-page schema, row/match counts, keyset-to-page mapping, id/def_index nulls,
# value range of *_id_fixed, pandas writer metadata, and naive polars read behaviour.
# Reads only the 2026 tome. Column projection everywhere; no full-channel materialisation.
# ASSUMPTION: the dataframe_NNNNN / keyset_NNNNN pairing is by the same numeric suffix (checked, not assumed).
import os, re, glob, json, collections, inspect, statistics
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CHANS = ["item_equip", "item_pickup", "player_hurt"]
ID_COLS = {"item_equip": ["player_id_fixed"], "item_pickup": ["player_id_fixed"],
           "player_hurt": ["player_id_fixed", "attacker_id_fixed"]}
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
DAY_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")
TWO53 = 2 ** 53


def P(*a):
    print(*a, flush=True)


def folder_of(ch):
    return os.path.join(TOME, f"basic_{ch}.2026-09-18,2026-10-08")


def uuid_of(s):
    m = UUID_RE.search(str(s))
    return m.group(0).lower() if m else None


def day_of(s):
    m = DAY_RE.search(str(s))
    return m.group(3) if m else None


def pandas_meta(schema):
    raw = (schema.metadata or {}).get(b"pandas")
    if raw is None:
        return None
    return {c["name"]: (c.get("pandas_type"), c.get("numpy_type")) for c in json.loads(raw).get("columns", [])}


def scan_channel(ch):
    folder = folder_of(ch)
    pages = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    ksets = sorted(glob.glob(os.path.join(folder, "keyset_*")))
    P(f"\n== {ch}: dataframe pages={len(pages)} keyset pages={len(ksets)}")
    assert [os.path.basename(p).split("_")[1] for p in pages] == [os.path.basename(k).split("_")[1] for k in ksets]
    recs, sig_groups = [], collections.OrderedDict()
    all_pages_uuids = []
    prev_last = None
    keyset_global_sorted = True
    for i, (p, k) in enumerate(zip(pages, ksets)):
        s = pq.read_schema(p)
        names = list(s.names)
        types = {n: str(t) for n, t in zip(s.names, s.types)}
        sig = tuple((n, types[n]) for n in names)
        sig_groups.setdefault(sig, []).append(i)
        nrows = pq.ParquetFile(p).metadata.num_rows
        idc = [c for c in ID_COLS[ch] if c in types]
        cols = ["match_key"] + idc + (["def_index"] if "def_index" in types else [])
        t = pq.read_table(p, columns=cols)
        mk = t.column("match_key")
        vc = pc.value_counts(mk).to_pylist()
        per_uuid, days_rows = {}, collections.Counter()
        for d in vc:
            u = uuid_of(d["values"])
            per_uuid[u] = per_uuid.get(u, 0) + d["counts"]
            days_rows[day_of(d["values"])] += d["counts"]
        kt = pq.read_table(k).column(0).to_pylist()
        kset = set(uuid_of(x) for x in kt)
        days_key = collections.Counter(day_of(x) for x in kt)
        if kt != sorted(kt):
            keyset_global_sorted = False
        if prev_last is not None and kt and kt[0] < prev_last:
            keyset_global_sorted = False
        if kt:
            prev_last = kt[-1]
        cnts = sorted(per_uuid.values())
        nulls = {c: t.column(c).null_count for c in cols if c != "match_key"}
        pm = pandas_meta(s)
        pmi = {c: pm.get(c) for c in idc + ["def_index"]} if pm else None
        recs.append(dict(i=i, nrows=nrows, ncols=len(names), has_def_index=("def_index" in types),
                         def_index_pos=(names.index("def_index") if "def_index" in types else None),
                         id_types={c: types[c] for c in idc}, nulls=nulls,
                         n_matches=len(per_uuid), keyset_n=len(kt), uuid_equal=(set(per_uuid) == kset),
                         days_rows=dict(sorted(days_rows.items())), days_keyset=dict(sorted(days_key.items())),
                         rows_per_match=(cnts[0], statistics.median(cnts), round(statistics.mean(cnts), 1), cnts[-1]),
                         pandas_types=pmi))
        all_pages_uuids.append(set(per_uuid))
        P(f"page {i:02d} {os.path.basename(p)} rows={nrows} ncols={len(names)} def_index={'def_index' in types}"
          f"{' (pos '+str(names.index('def_index'))+')' if 'def_index' in types else ''} id_types={ {c: types[c] for c in idc} }"
          f" nulls={nulls} matches={len(per_uuid)} keyset_n={len(kt)} uuid_sets_equal={set(per_uuid) == kset}")
        P(f"        days_rows={dict(sorted(days_rows.items()))} days_keyset={dict(sorted(days_key.items()))}")
        P(f"        rows_per_match min/med/mean/max={recs[-1]['rows_per_match']} pandas_meta={pmi}")
    P(f"schema signature groups (name+type, order-sensitive): {len(sig_groups)} distinct")
    for gi, (sig, members) in enumerate(sig_groups.items()):
        P(f"  group {gi}: pages {members} ncols={len(sig)} order_head={[n for n, _ in sig][:6]} "
          f"def_index_pos={[n for n, _ in sig].index('def_index') if 'def_index' in [n for n, _ in sig] else None}")
    total_rows = sum(r["nrows"] for r in recs)
    union = set().union(*all_pages_uuids)
    dup_across = sum(len(s) for s in all_pages_uuids) - len(union)
    P(f"TOTAL rows={total_rows} distinct_matches={len(union)} duplicate_match_across_pages={dup_across}")
    P(f"keyset lists globally sorted by path (contiguous page slices of one sorted list): {keyset_global_sorted}")
    return pages, recs, sig_groups


def naive_and_workaround(ch, pages):
    idc = ID_COLS[ch][0]
    P(f"\n-- naive polars reads for {ch} (polars {pl.__version__})")
    checks = [
        ("scan_parquet(pages).head(5).collect()", lambda: pl.scan_parquet(pages).head(5).collect()),
        ("scan_parquet(pages).select(pl.len()).collect()", lambda: pl.scan_parquet(pages).select(pl.len()).collect().item()),
        (f"scan_parquet(pages).select(null_count({idc})).collect()",
         lambda: pl.scan_parquet(pages).select(pl.col(idc).null_count()).collect().item()),
        ("read_parquet(pages, n_rows=5)", lambda: pl.read_parquet(pages, n_rows=5).shape),
    ]
    for label, fn in checks:
        try:
            P(f"   {label}: OK -> {fn()}")
        except Exception as e:
            P(f"   {label}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:170]}")
    P(f"-- ScanCastOptions signature: {inspect.signature(pl.ScanCastOptions)}")


def workaround(ch, pages, recs):
    idc = ID_COLS[ch][0]
    P(f"\n-- workaround test for {ch}: scan with cast options (lazy, aggregated only)")
    opts_list = [
        dict(integer_cast="allow-float", missing_columns="insert"),
        dict(integer_cast="allow-float"),
    ]
    for kw in opts_list:
        try:
            opts = pl.ScanCastOptions(**kw)
        except Exception as e:
            P(f"   ScanCastOptions{kw}: construct FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            continue
        try:
            agg = pl.scan_parquet(pages, cast_options=opts).select(
                pl.len().alias("n"), pl.col(idc).null_count().alias("id_nulls"),
                pl.col(idc).cast(pl.Float64).max().alias("id_max"),
                pl.col(idc).cast(pl.Float64).min().alias("id_min")).collect()
            P(f"   ScanCastOptions{kw}: OK -> {agg.row(0, named=True)}")
            break
        except Exception as e:
            P(f"   ScanCastOptions{kw}: scan FAIL {type(e).__name__}: {str(e).splitlines()[0][:170]}")
    # native (pyarrow) aggregates per page for comparison
    tot_n, tot_null, mx, mn = 0, 0, None, None
    for p in pages:
        col = pq.read_table(p, columns=[idc]).column(idc)
        tot_n += len(col)
        tot_null += col.null_count
        pmx, pmn = pc.max(col).as_py(), pc.min(col).as_py()
        mx = pmx if mx is None else max(mx, pmx)
        mn = pmn if mn is None else min(mn, pmn)
    P(f"   native per-page aggregates (pyarrow): n={tot_n} id_nulls={tot_null} max={mx} min={mn}")
    # magnitude check: are id values beyond 2^53 (float64 exact-integer limit)?
    for i, p in enumerate(pages):
        s = pq.read_schema(p)
        dt = str(s.field(idc).type)
        col = pq.read_table(p, columns=[idc]).column(idc)
        mxa = pc.max(col).as_py()
        mna = pc.min(col).as_py()
        if dt == "int64":
            beyond = int(pc.sum(pc.greater(pc.abs(col), pa.scalar(TWO53, pa.int64()))).as_py() or 0)
        else:
            beyond = int(pc.sum(pc.greater(pc.abs(col), float(TWO53))).as_py() or 0)
        nonint = None
        if dt == "double":
            nonint = int(pc.sum(pc.not_equal(col, pc.floor(col))).as_py() or 0)
        P(f"   page {i:02d} {idc} type={dt} min={mna} max={mxa} count(|v|>2^53)={beyond} non_integral={nonint}")


def per_page_projection(ch, pages):
    idc = ID_COLS[ch][0]
    P(f"\n-- per-page projection test for {ch}: read [match_key, {idc}] per page, cast to Float64, concat vertical_relaxed")
    frames = []
    for p in pages:
        frames.append(pl.read_parquet(p, columns=["match_key", idc]).with_columns(pl.col(idc).cast(pl.Float64)))
    df = pl.concat(frames, how="vertical_relaxed")
    P(f"   rows={df.height} distinct_match_keys={df.get_column('match_key').n_unique()} "
      f"{idc} dtype={df.schema[idc]} nulls={df.get_column(idc).null_count()}")


if __name__ == "__main__":
    P("verify_A1_3_tome: 2026 tome, item_equip / item_pickup / player_hurt")
    results = {}
    for ch in CHANS:
        pages, recs, sig_groups = scan_channel(ch)
        results[ch] = (pages, recs)
        naive_and_workaround(ch, pages)
        workaround(ch, pages, recs)
        per_page_projection(ch, pages)
    P("\nDONE tome checks")
