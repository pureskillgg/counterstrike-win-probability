"""Count-only profile of the 2026-window basic_player_name channel, frozen-test keys excluded. No values printed."""
import json, re, sys
from collections import Counter
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
RAW = r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2026-10-01_to_08/tome/csds/basic_player_name.2026-09-18,2026-10-08/dataframe_00000"
FROZEN = r"C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/data/benchmark/frozen_test_keys.json"
fz = json.load(open(FROZEN, encoding="utf-8"))
frozen = set()
if isinstance(fz, dict):
    for v in fz.values():
        if isinstance(v, list): frozen.update(x for x in v if isinstance(x, str))
    if not frozen: frozen = set(fz.keys())
else:
    frozen = set(x for x in fz if isinstance(x, str))
t = pq.read_table(RAW)
mk = t.column("match_key").to_pylist()
keep = [i for i, k in enumerate(mk) if k not in frozen and not str(k) >= "csds/2026/10/26"]
print("rows total:", t.num_rows, "kept (non-frozen, pre-Oct 26):", len(keep))
for c in ["name_new", "name_old", "player_id", "player_id_fixed"]:
    vals = t.column(c).to_pylist()
    vals = [vals[i] for i in keep]
    nonempty = [v for v in vals if v not in (None, "")]
    cnt = Counter(str(v) for v in nonempty)
    redact = sum(bool(re.search(r"redact|anon|unknown|placeholder", str(v), re.I)) for v in nonempty)
    print(f"{c}: nonempty={len(nonempty)} distinct={len(cnt)} redact_anon_unknown_rows={redact} top10_freq={sorted(cnt.values(), reverse=True)[:10]}")
