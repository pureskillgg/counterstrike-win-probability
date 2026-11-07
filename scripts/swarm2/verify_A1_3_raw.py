# Verifier for A1-3: compare the tome page storage classes against the RAW per-match parquet files on E:
# (read-only, metadata/footers only). For each keyset-listed match of the three drifting channels:
#   raw dtype of the id column, raw presence of def_index, raw row count;
# then cross-tab raw dtype vs tome page class, and reconcile raw row sums per tome page vs tome page rows.
# Usage: python verify_A1_3_raw.py [max_matches_per_channel or 0 for all]
import os, sys, glob, time, collections
import pyarrow.parquet as pq

LIM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
RAW = r"E:\csds_final\csds\2026\10"
CH = {"basic_item_equip": ("item_equip", "player_id_fixed"),
      "basic_item_pickup": ("item_pickup", "player_id_fixed"),
      "basic_player_hurt": ("player_hurt", "attacker_id_fixed")}

t0 = time.time()
for tome_ch, (raw_ch, idc) in CH.items():
    folder = os.path.join(T, tome_ch + ".2026-09-18,2026-10-08")
    pages = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    print("=" * 110)
    print(tome_ch, "idc", idc, "tome pages", len(pages))
    raw_dtype = collections.Counter()
    raw_hasdef = collections.Counter()
    cross = collections.Counter()           # (raw dtype, tome page class)
    missing_raw = 0
    per_page = []
    n_done = 0
    for dp in pages:
        pn = os.path.basename(dp).split("_")[1]
        tome_rows = pq.ParquetFile(dp).metadata.num_rows
        tome_cls = str(pq.ParquetFile(dp).schema_arrow.field(idc).type)
        tome_names = pq.ParquetFile(dp).schema_arrow.names
        keys = pq.read_table(os.path.join(folder, "keyset_" + pn)).column(0).to_pylist()
        if LIM:
            keys = keys[:LIM]
        raw_sum = 0
        page_raw_types = set()
        page_raw_def = []
        raw_null_in_double = 0
        for k in keys:
            parts = k.split("/")
            day, uuid = parts[3], parts[4]
            rp = os.path.join(RAW, day, uuid, raw_ch)
            if not os.path.exists(rp):
                missing_raw += 1
                continue
            pf = pq.ParquetFile(rp)
            s = pf.schema_arrow
            rt = str(s.field(idc).type) if idc in s.names else "absent"
            raw_dtype[(day, rt)] += 1
            has_def = "def_index" in s.names
            raw_hasdef[(day, has_def)] += 1
            cross[(rt, tome_cls)] += 1
            page_raw_types.add(rt)
            page_raw_def.append(has_def)
            raw_sum += pf.metadata.num_rows
            n_done += 1
        pred_cls = "double" if "double" in page_raw_types else ("int64" if page_raw_types == {"int64"} else "mixed/absent")
        pred_def = any(page_raw_def)
        all_def = all(page_raw_def) if page_raw_def else False
        per_page.append((pn, len(keys), tome_rows, tome_cls, raw_sum, "def_index" in tome_names, pred_cls, pred_def, all_def))
        print(f"  page {pn}: matches_checked={len(keys)} tome_class={tome_cls} tome_rows={tome_rows} "
              f"raw_rows_sum={raw_sum} raw_types_in_page={sorted(page_raw_types)} "
              f"pred_class_by_union={pred_cls} tome_has_def={'def_index' in tome_names} "
              f"raw_def_any={pred_def} raw_def_all={all_def} (elapsed {time.time() - t0:.0f}s)")
    print(tome_ch, "raw files read:", n_done, "missing raw matches:", missing_raw)
    print(tome_ch, "raw dtype by day:", dict(sorted(raw_dtype.items())))
    print(tome_ch, "raw def_index presence by day:", dict(sorted(raw_hasdef.items())))
    print(tome_ch, "cross-tab (raw dtype, tome page class):", dict(cross))
    print(tome_ch, "pages where 'any raw double -> double, else int64' predicts tome class:",
          sum(1 for p in per_page if p[6] == p[3]), "of", len(per_page))
    print(tome_ch, "pages where 'any raw has def_index' predicts tome def_index presence:",
          sum(1 for p in per_page if p[7] == p[5]), "of", len(per_page))
    print(tome_ch, "per-page (tome_rows - raw_rows_sum):", [(p[0], p[2] - p[4]) for p in per_page])
    print(tome_ch, "per-page raw rows sum total:", sum(p[4] for p in per_page), "tome rows total:", sum(p[2] for p in per_page))
print("elapsed seconds:", round(time.time() - t0, 1))
print("done")
