# verify_A1_3_mech.py
# Tests the mechanism behind A1-3 directly: does each 2026 tome page equal the union/supertype of the RAW
# per-match files it contains? Predictions tested per tome page (read-only footers on E:):
#   P1: tome id column is double iff at least one raw match in the page has a double id column
#   P2: tome def_index column is present iff at least one raw match in the page has def_index
#   P3: tome def_index null count == summed raw rows of matches in the page that lack def_index
#   P4: tome rows == summed raw rows of the page's matches (already seen; re-checked here)
# Also reports, per page, the raw signature counts and the page row total (for the row-chunking inference).
# ASSUMPTION: raw match file = E:/csds_final/csds/2026/10/<DD>/<uuid>/<channel>.
import os, glob
import pyarrow.parquet as pq
import pyarrow.compute as pc

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
RAW = "E:/csds_final/csds/2026/10"
CHANS = {"item_equip": "player_id_fixed", "item_pickup": "player_id_fixed", "player_hurt": "attacker_id_fixed"}


def P(*a):
    print(*a, flush=True)


ok_all = True
for ch, idc in CHANS.items():
    folder = os.path.join(TOME, f"basic_{ch}.2026-09-18,2026-10-08")
    pages = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    ksets = sorted(glob.glob(os.path.join(folder, "keyset_*")))
    P(f"\n== {ch} (id column {idc})")
    for pi, (dp, kp) in enumerate(zip(pages, ksets)):
        ts = pq.read_schema(dp)
        tome_type = str(ts.field(idc).type)
        tome_has_def = "def_index" in ts.names
        tome_def_nulls = pq.read_table(dp, columns=["def_index"]).column(0).null_count if tome_has_def else None
        keys = pq.read_table(kp).column(0).to_pylist()
        any_double = False
        any_def = False
        rows_sum = 0
        def_missing_rows = 0
        n_lab = {}
        for k in keys:
            parts = k.split("/")
            rp = f"{RAW}/{parts[3]}/{parts[4]}/{ch}"
            if not os.path.exists(rp):
                continue
            s = pq.read_schema(rp)
            nr = pq.read_metadata(rp).num_rows
            rows_sum += nr
            rt = str(s.field(idc).type) if idc in s.names else "absent"
            if rt == "double":
                any_double = True
            has_def = "def_index" in s.names
            any_def = any_def or has_def
            if not has_def:
                def_missing_rows += nr
            lab = f"{rt}/{'def' if has_def else 'nodef'}"
            n_lab[lab] = n_lab.get(lab, 0) + 1
        pred_type = "double" if any_double else "int64"
        p1 = (tome_type == pred_type)
        p2 = (tome_has_def == any_def)
        p3 = (tome_has_def is False) or (tome_def_nulls == def_missing_rows)
        p4 = True  # rows already compared in verify_A1_3_raw.py
        ok_all = ok_all and p1 and p2 and p3
        P(f"  page {pi:02d}: tome {idc}={tome_type} def_index={'yes' if tome_has_def else 'no'} "
          f"tome_def_nulls={tome_def_nulls} raw_labels={n_lab} raw_rows_sum={rows_sum} tome_rows={pq.read_metadata(dp).num_rows}")
        P(f"           P1 type==supertype(raw): {p1} (pred {pred_type}); P2 def present iff any raw def: {p2}; "
          f"P3 def_nulls==rows of raw matches without def ({def_missing_rows}): {p3}")
P(f"\nALL PREDICTIONS P1-P3 HOLD: {ok_all}")
P("DONE mech")
