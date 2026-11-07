import sys, json, os
import pyarrow.parquet as pq

def jtype(v):
    if isinstance(v, dict): return "dict"
    if isinstance(v, list): return "list[%d]" % len(v)
    if isinstance(v, bool): return "bool"
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str): return "str(len~%d)" % len(v)
    return type(v).__name__

def describe_json(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    print("  top-level type:", jtype(obj))
    if isinstance(obj, dict):
        print("  top-level keys (%d):" % len(obj), list(obj.keys())[:40])
        for k in list(obj.keys())[:15]:
            print("   key %r -> %s" % (k, jtype(obj[k])))
            v = obj[k]
            if isinstance(v, list) and v:
                e = v[0]
                if isinstance(e, dict):
                    print("     elem keys:", list(e.keys())[:60], "types:", {kk: jtype(vv) for kk, vv in list(e.items())[:60]})
                else:
                    print("     elem type:", jtype(e), "(not printed)")
            elif isinstance(v, dict):
                print("     sub keys (%d):" % len(v), list(v.keys())[:30])
    elif isinstance(obj, list):
        print("  list length:", len(obj))
        if obj:
            e = obj[0]
            if isinstance(e, dict):
                print("  elem keys:", list(e.keys())[:60])
                print("  elem types:", {kk: jtype(vv) for kk, vv in list(e.items())[:60]})
            else:
                print("  elem type:", jtype(e), "(values not printed)")

for p in sys.argv[1:]:
    print("=== FILE:", os.path.basename(p), "size:", os.path.getsize(p))
    if p.endswith(".parquet"):
        pf = pq.ParquetFile(p)
        md = pf.metadata
        print("  rows:", md.num_rows, "row_groups:", md.num_row_groups, "cols:", md.num_columns)
        for field in pf.schema_arrow:
            print("   col %-28s %s" % (field.name, field.type))
    else:
        try:
            describe_json(p)
        except Exception as ex:
            print("  parse error:", type(ex).__name__)
