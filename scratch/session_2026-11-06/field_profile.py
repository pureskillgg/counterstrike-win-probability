"""Count-only profile of identifier-like fields on one non-frozen training-window raw file.
Prints counts, pattern classes and frequency shapes only. No values are printed.
"""
import json, re, sys
from collections import Counter
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
RAW = r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2026-10-01_to_08/tome/csds/basic_player_personal.2026-09-18,2026-10-08/dataframe_00000"
FROZEN = r"C:/Users/13144/Documents/pureskill/counterstrike-win-probability/data/benchmark/frozen_test_keys.json"
fz = json.load(open(FROZEN, encoding="utf-8"))
frozen = set()
if isinstance(fz, dict):
    for v in fz.values():
        if isinstance(v, list): frozen.update(x for x in v if isinstance(x, str))
    if not frozen: frozen = set(fz.keys())
else:
    frozen = set(x for x in fz if isinstance(x, str))
t = pq.read_table(RAW, columns=["match_key", "steam_id", "name", "clan_tag", "is_bot", "player_id", "player_controller_id"])
mk = t.column("match_key").to_pylist()
rows = [i for i, k in enumerate(mk) if k not in frozen and not str(k) >= "csds/2026/10/26"]
def col(name): c = t.column(name).to_pylist(); return [c[i] for i in rows]
def profile(label, vals):
    nonempty = [v for v in vals if v not in (None, "")]
    c = Counter(str(v) for v in nonempty)
    pats = {
        "hex_hash_32_64": sum(bool(re.fullmatch(r"[0-9a-f]{32,64}", str(v))) for v in nonempty),
        "contains_redact_or_anon": sum(bool(re.search(r"redact|anon|unknown|placeholder", str(v), re.I)) for v in nonempty),
        "digits_only": sum(str(v).isdigit() for v in nonempty),
        "steam_style_STEAM_": sum(str(v).startswith("STEAM_") for v in nonempty),
        "steam_style_U_bracket": sum(str(v).startswith("[U:") for v in nonempty),
        "all_zero": sum(bool(re.fullmatch(r"0+", str(v))) for v in nonempty),
    }
    freq = sorted(c.values(), reverse=True)
    print(f"{label}: rows={len(vals)} nonempty={len(nonempty)} distinct={len(c)} patterns={pats}")
    print(f"    frequency of the 10 most common values (counts only): {freq[:10]}")
    print(f"    values seen once: {sum(1 for v in c.values() if v == 1)}; values seen 10+ times: {sum(1 for v in c.values() if v >= 10)}")
for f in ["steam_id", "name", "clan_tag"]:
    profile(f, col(f))
profile("player_id", col("player_id"))
print("is_bot true share:", round(sum(1 for v in col("is_bot") if v) / max(len(rows), 1), 5))
