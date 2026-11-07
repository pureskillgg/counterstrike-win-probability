"""Read-only: print full column objects for columns whose manifest type key is missing or
that are declared in the manifest but absent from the parquet. Never writes."""
import collections
import gzip
import json
import os

E = r"E:\csds_final\csds"
M24 = os.path.join(E, "csds_tail_recovered_20261011", "csds", "2024", "02", "02",
                   "m_77fca7aed7217a25e7d1fea9")
M26 = os.path.join(E, "2026", "10", "01", "m_a661aadb0b4b87e35cdec94c")


def manifest(d):
    with open(os.path.join(d, "csds"), "rb") as fh:
        return json.loads(gzip.decompress(fh.read()).decode("utf-8"))


def pq_names(d, ch):
    import pyarrow.parquet as pq
    sch = pq.read_schema(os.path.join(d, ch))
    return [sch.field(i).name for i in range(len(sch))]


for label, d in (("2024", M24), ("2026", M26)):
    doc = manifest(d)
    print("=====", label)
    keyset = collections.Counter()
    no_type = []
    for c in doc["channels"]:
        for col in c["columns"]:
            keyset[tuple(sorted(col.keys()))] += 1
            if "type" not in col:
                no_type.append((c["channel"], col.get("name"), col.get("origin")))
    print("column key sets:")
    for k, n in keyset.most_common():
        print("  ", n, k)
    print("columns without a type key:", len(no_type))
    origins = collections.Counter(o for _c, _n, o in no_type)
    print("  origins of those:", origins.most_common(8))
    print("  examples:", no_type[:6])

    # manifest columns missing from the parquet, per channel
    ex_missing = collections.Counter()
    for c in doc["channels"]:
        names = [col.get("name") for col in c["columns"]]
        pn = pq_names(d, c["channel"])
        miss = [n for n in names if n not in pn]
        extra = [n for n in pn if n not in names]
        if miss or extra:
            ex_missing[c["channel"]] += 1
            if ex_missing[c["channel"]] == 1:
                print("  channel", c["channel"], "manifest-not-parquet:", miss[:8], " parquet-not-manifest:", extra[:8])
    print("channels with any name mismatch:", len(ex_missing), sorted(ex_missing)[:10])

    # full objects for the two known extra columns in bomb_action
    if label == "2024":
        for c in doc["channels"]:
            if c["channel"] == "bomb_action":
                for col in c["columns"]:
                    if col.get("name") in ("player_player_id", "player_tick", "player_id_fixed", "player_phi_ang"):
                        print("  FULL:", json.dumps(col, sort_keys=True)[:600])
