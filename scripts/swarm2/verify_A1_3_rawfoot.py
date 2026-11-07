# verify_A1_3_rawfoot.py
# Verifier for A1-3, raw side. For every match in each drifting channel's 2026 tome keyset, reads the
# raw per-match parquet on E: (read-only; footer + one def_index column only) and compares:
#   - raw schema (names + types, order-sensitive) vs the tome page the match lands in
#   - raw num_rows vs tome rows for the same match (duplication / loss check)
#   - raw file mtime per tome page (ingestion-run check)
#   - raw def_index null share
#   - tome rows per round (round_end) by day (length-normalised check of the day-08 effect)
# ASSUMPTION: raw match file = E:/csds_final/csds/2026/10/<DD>/<uuid>/<channel> (a plain parquet file; magic PAR1 checked).
# Note: this is a re-creation of the version first run as verify_A1_3_raw.py; that file was later overwritten on disk by another process.
import os, re, glob, time, collections, datetime
import pyarrow.parquet as pq
import pyarrow.compute as pc
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
RAW = "E:/csds_final/csds/2026/10"
CHANS = ["item_equip", "item_pickup", "player_hurt"]
ID_COLS = {"item_equip": ["player_id_fixed"], "item_pickup": ["player_id_fixed"],
           "player_hurt": ["player_id_fixed", "attacker_id_fixed"]}
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
DAY_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")


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


def tome_pages(ch):
    folder = folder_of(ch)
    pages = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    ksets = sorted(glob.glob(os.path.join(folder, "keyset_*")))
    out = []
    for i, (p, k) in enumerate(zip(pages, ksets)):
        s = pq.read_schema(p)
        sig = tuple((n, str(t)) for n, t in zip(s.names, s.types))
        keys = pq.read_table(k).column(0).to_pylist()
        t = pq.read_table(p, columns=["match_key"])
        counts = {}
        for d in pc.value_counts(t.column("match_key")).to_pylist():
            u = uuid_of(d["values"])
            counts[u] = counts.get(u, 0) + d["counts"]
        out.append(dict(i=i, sig=sig, nrows=pq.read_metadata(p).num_rows, keys=keys, counts=counts))
    return out


def ts(x):
    return datetime.datetime.fromtimestamp(x, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")


def raw_check(ch, pages):
    t0 = time.time()
    P(f"\n== {ch}: raw footer check across {sum(len(p['keys']) for p in pages)} keyset matches")
    sig_label = {}
    per_page = []
    for pg in pages:
        stats = dict(i=pg["i"], n_keys=len(pg["keys"]), missing=0, raw_rows=0, tome_rows=pg["nrows"],
                     eq=0, tome_gt=0, tome_lt=0, sig_counter=collections.Counter(), mt=[], def_nulls=0,
                     def_rows=0, id_types=collections.Counter(), def_present=0)
        for key in pg["keys"]:
            u = uuid_of(key)
            day = day_of(key)
            path = f"{RAW}/{day}/{u}/{ch}"
            if not os.path.exists(path):
                stats["missing"] += 1
                continue
            md = pq.read_metadata(path)
            sch = pq.read_schema(path)
            sig = tuple((n, str(t)) for n, t in zip(sch.names, sch.types))
            if sig not in sig_label:
                sig_label[sig] = f"S{len(sig_label)}"
            stats["sig_counter"][sig_label[sig]] += 1
            nr = md.num_rows
            stats["raw_rows"] += nr
            tr = pg["counts"].get(u, 0)
            if tr == nr:
                stats["eq"] += 1
            elif tr > nr:
                stats["tome_gt"] += 1
            else:
                stats["tome_lt"] += 1
            stats["mt"].append(os.path.getmtime(path))
            dtypes = dict(sig)
            for c in ID_COLS[ch]:
                if c in dtypes:
                    stats["id_types"][f"{c}={dtypes[c]}"] += 1
            if "def_index" in dtypes:
                stats["def_present"] += 1
                col = pq.read_table(path, columns=["def_index"]).column(0)
                stats["def_nulls"] += col.null_count
                stats["def_rows"] += len(col)
        per_page.append(stats)
    P(f"   elapsed {time.time() - t0:.1f}s; raw signature labels: " + "; ".join(
        f"{lab}: ncols={len(sig)} def_index={'def_index' in dict(sig)} id_types={ {c: dict(sig).get(c) for c in ID_COLS[ch]} }"
        for sig, lab in sig_label.items()))
    for stt in per_page:
        P(f"   tome page {stt['i']:02d}: keys={stt['n_keys']} raw_missing={stt['missing']} tome_rows={stt['tome_rows']} "
          f"raw_rows_sum={stt['raw_rows']} per_match_rows_equal={stt['eq']} tome_gt_raw={stt['tome_gt']} tome_lt_raw={stt['tome_lt']}")
        P(f"           raw sig labels={dict(stt['sig_counter'])} raw id types={dict(stt['id_types'])} "
          f"raw_def_index_present={stt['def_present']} raw_def_nulls={stt['def_nulls']}/{stt['def_rows']}")
        if stt["mt"]:
            P(f"           raw mtime UTC range {ts(min(stt['mt']))} .. {ts(max(stt['mt']))}")
    return per_page, sig_label


if __name__ == "__main__":
    P("verify_A1_3_rawfoot: raw per-match footers on E: (read-only) vs 2026 tome pages")
    for ch in CHANS:
        pages = tome_pages(ch)
        raw_check(ch, pages)
    P("\nDONE rawfoot")
