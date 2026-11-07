import json, polars as pl
VAL = "G:/pureskill/csds-cs2/data/sprint/matrix_trainctr/val_keys.parquet"
PRED83 = "G:/pureskill/csds-cs2/data/sprint/run/gpu_fast/preds/full_d4_e0.03_m20_s0.parquet"
HB = "G:/pureskill/csds-cs2/data/combined/holdout_22_25_test_keys.json"
OCT = "C:/Users/13144/sprint/holdout_oct26_31_keys.parquet"
FZ = "G:/pureskill/csds-cs2/data/benchmark/frozen_test_keys.json"
for name, p in [("val_keys", VAL), ("pred83", PRED83), ("oct_keys", OCT)]:
    s = pl.scan_parquet(p)
    print(name, "schema:", dict(s.collect_schema()), "rows(footer):", s.select(pl.len()).collect().item(), flush=True)
for name, p in [("holdoutB_json", HB), ("frozen_json", FZ)]:
    o = json.load(open(p, encoding="utf-8"))
    if isinstance(o, dict):
        print(name, "dict keys:", list(o.keys())[:20], {k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in list(o.items())[:20]}, flush=True)
    else:
        print(name, "type list len", len(o), "elem type", type(o[0]).__name__, flush=True)
