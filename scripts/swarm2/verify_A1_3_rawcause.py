# Verifier for A1-3, root-cause probe on RAW per-match files (E:, read-only, footers + statistics only):
# for each raw id column: is the raw dtype double vs int64 associated with (a) null count, (b) the parquet
# writer recorded in the footer (created_by), (c) row count? Also counts of the raw def_index presence.
import os, sys, collections
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
RAW = r"E:\csds_final\csds\2026\10"
CH = {"basic_item_equip": ("item_equip", "player_id_fixed"),
      "basic_item_pickup": ("item_pickup", "player_id_fixed"),
      "basic_player_hurt": ("player_hurt", "attacker_id_fixed")}

for tome_ch, (raw_ch, idc) in CH.items():
    keyfile = os.path.join(T, tome_ch + ".2026-09-18,2026-10-08")
    keys = []
    for i in range(10):
        kp = os.path.join(keyfile, f"keyset_{i:05d}")
        if not os.path.exists(kp):
            break
        keys += pq.read_table(kp).column(0).to_pylist()
    stat = collections.defaultdict(lambda: {"n": 0, "rows": 0, "null_total": 0, "matches_with_null": 0,
                                            "created_by": collections.Counter(), "int_nullable_zero": 0})
    for k in keys:
        parts = k.split("/")
        day, uuid = parts[3], parts[4]
        rp = os.path.join(RAW, day, uuid, raw_ch)
        pf = pq.ParquetFile(rp)
        md = pf.metadata
        s = pf.schema_arrow
        if idc not in s.names:
            continue
        rt = str(s.field(idc).type)
        d = stat[rt]
        d["n"] += 1
        d["rows"] += md.num_rows
        d["created_by"][str(md.created_by)[:60]] += 1
        # null count via column statistics when present, else read the single column
        nulls = 0
        ci = s.names.index(idc)
        have_stats = True
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(ci).statistics
            if st is None or not st.has_null_count:
                have_stats = False
                break
            nulls += st.null_count
        if not have_stats:
            nulls = pq.read_table(rp, columns=[idc]).column(0).null_count
        d["null_total"] += nulls
        if nulls > 0:
            d["matches_with_null"] += 1
    print("=" * 110)
    print(tome_ch, "raw matches scanned:", len(keys))
    for rt, d in sorted(stat.items()):
        print(f"  raw {idc} dtype={rt}: matches={d['n']} rows={d['rows']} null_total={d['null_total']} "
              f"matches_with_any_null={d['matches_with_null']} created_by_top={d['created_by'].most_common(3)}")
print("done")
