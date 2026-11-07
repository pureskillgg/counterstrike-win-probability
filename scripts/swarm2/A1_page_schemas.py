# A1: per-channel page schema consistency (names, order, types) across dataframe_* pages.
import os, glob, json
import pyarrow.parquet as pq
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
chans = sorted(d for d in os.listdir(T) if d != "header" and not d.startswith("."))
res = {}
for c in chans:
    pages = sorted(glob.glob(os.path.join(T, c, "dataframe_*")))
    sigs = []
    for p in pages:
        s = pq.read_schema(p)
        sigs.append(tuple((n, str(t)) for n, t in zip(s.names, s.types)))
    uniq = []
    for s in sigs:
        if s not in uniq:
            uniq.append(s)
    if len(uniq) > 1 or len(pages) > 1 and len(set(tuple(x[0] for x in s) for s in sigs)) > 1:
        names_sets = [set(n for n, _ in s) for s in uniq]
        print(c, "pages", len(pages), "distinct_schemas", len(uniq))
        for i, s in enumerate(uniq):
            print("   schema", i, "ncols", len(s), "names", [n for n, _ in s][:60])
        base = set(n for n, _ in uniq[0])
        for i, s in enumerate(uniq[1:], 1):
            ns = set(n for n, _ in s)
            print("   schema", i, "only-vs-0:", sorted(ns - base), "missing-vs-0:", sorted(base - ns))
            if [n for n,_ in s] != [n for n,_ in uniq[0]] and ns == base:
                print("   schema", i, "same set, different ORDER")
            tdiff = [(n, t, dict(uniq[0]).get(n)) for n, t in s if n in dict(uniq[0]) and dict(uniq[0])[n] != t]
            if tdiff: print("   schema", i, "type diffs:", tdiff[:10])
    else:
        print(c, "pages", len(pages), "consistent; ncols", len(uniq[0]))
    res[c] = dict(pages=len(pages), distinct=len(uniq), ncols=[len(s) for s in sigs])
json.dump(res, open(r"G:\pureskill\csds-cs2\logs\swarm2\A1_page_schemas.json", "w"), indent=1)
