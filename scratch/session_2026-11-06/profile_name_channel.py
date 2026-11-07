"""Schema, then count-only profile, of basic_player_name in the 2024 training window. No values printed."""
import glob, re, sys
from collections import Counter
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
BASE = r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2024-02-g/tome/csds"
f = sorted(glob.glob(f"{BASE}/basic_player_name.*/dataframe_*"))[0]
sch = pq.read_schema(f)
print("basic_player_name columns:", [(x.name, str(x.type)) for x in sch])
t = pq.read_table(f)
print("rows:", t.num_rows)
for c in t.column_names:
    if c in ("match_key",):
        vals = t.column(c).to_pylist()
        print(f"{c}: distinct={len(set(vals))}")
        continue
    vals = t.column(c).to_pylist()
    nonempty = [v for v in vals if v not in (None, "")]
    cnt = Counter(str(v) for v in nonempty)
    redact = sum(bool(re.search(r"redact|anon|unknown|placeholder", str(v), re.I)) for v in nonempty)
    print(f"{c}: nonempty={len(nonempty)} distinct={len(cnt)} redact_anon_unknown_rows={redact} top10_freq={sorted(cnt.values(), reverse=True)[:10]}")
