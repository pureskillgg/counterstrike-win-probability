"""Compact console summary of notes/eda/weapons.json (reads JSON only)."""
import json

PATH = r"G:\pureskill\csds-cs2\notes\eda\weapons.json"


def short(ex):
    if not isinstance(ex, dict):
        return str(ex)[:120]
    mk = str(ex.get("match_key", ""))
    mk = mk.split("/")[-1][:8] if mk else ""
    rest = {k: v for k, v in ex.items() if k != "match_key"}
    return f"{mk} {rest}"[:160]


d = json.load(open(PATH, encoding="utf-8"))
print("universe", d["universe"])
for ch, r in d["channels"].items():
    m = r["matches"]
    print(f"\n=== {ch} rows={r['rows']} pages={r['pages']} drift={len(r['schema_drift'])}")
    print(f"  matches_with_rows={m.get('matches_with_rows')} zero_row={m.get('zero_row_matches')} "
          f"not_in_header={m.get('keys_not_in_header')} spanning_pages={m.get('matches_spanning_pages')}")
    print(f"  rows/match median={m.get('rows_per_match_median')} min={m.get('rows_per_match_min')} max={m.get('rows_per_match_max')}")
    for f in m.get("far_from_median_top3", []):
        print(f"    far: {f['match_key'].split('/')[-1][:8]} rows={f['rows']} ratio={f['ratio_to_median']} dist={f['distance_ratio']}")
    e = r["dups_exact_rows"]
    print(f"  exact dup: groups={e['groups']} extra={e['extra_rows']} matches={e['matches_affected']} top={e['top_matches'][:2]}")
    k = r["dups_key"]
    if k:
        print(f"  key dup {r['key_cols_for_dup_check']}: groups={k['groups']} extra={k['extra_rows']} matches={k['matches_affected']}")
    s = r["second_vs_tick"]
    if s:
        print(f"  second-tick/64: median={s['median_second_minus_tick_over_64']:.4f} p01={s['p01']:.4f} p99={s['p99']:.4f} "
              f"min={s['min']:.3f} max={s['max']:.3f} outside={s['rows_outside_tolerance']}")
    for p in r["pair_checks"]:
        if p["rows_differ"] or p["a"].endswith("_fixed") or p["b"].endswith("_fixed"):
            print(f"  pair {p['a']} vs {p['b']}: differ={p['rows_differ']}")
    if r.get("weapon_code_name_pairs"):
        w = r["weapon_code_name_pairs"]
        print(f"  code->names>1: {w['codes_with_more_than_one_name']}")
        print(f"  name->codes>1: {w['names_with_more_than_one_code']}")
    print(f"  impossible-by-name columns present: { {k2: v for k2, v in r['impossible_by_name_columns_present'].items() if v} or 'none' }")
    for f in r["flags"]:
        extra = ""
        if "value" in f:
            extra += f" value={f['value']}"
        if "share" in f:
            extra += f" share={f['share']}"
        ex = [short(x) for x in f.get("examples", [])][:2]
        print(f"  [{f['column']}] {f['flag']} count={f.get('count')}{extra} ex={ex}")
