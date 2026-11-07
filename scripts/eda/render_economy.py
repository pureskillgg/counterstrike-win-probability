"""Render notes/eda/economy.md and notes/eda/economy.json from the merged profile stats.

Inputs:
  notes/eda/economy.json            raw stats written by profile_economy.py (merge step)
  <scratchpad>/economy_curation.json curated goofy top-5 and per-channel commentary (hand-written)

Writes:
  notes/eda/economy.md
  notes/eda/economy.json            same data as md (stats + curation)
"""
import json
import math
import os

STATS = "G:/pureskill/csds-cs2/notes/eda/economy.json"
CURATION = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/7308a8e7-a47e-4034-b140-5c511acb0aa4/scratchpad/economy_curation.json"
OUT_MD = "G:/pureskill/csds-cs2/notes/eda/economy.md"
OUT_JSON = "G:/pureskill/csds-cs2/notes/eda/economy.json"


def g(v, nd=6):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v == int(v) and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:.{nd}g}"
    return str(v)


def pct(x):
    return f"{100*x:.2f}%"


def ex_str(ex):
    parts = []
    for e in ex:
        bits = []
        if "file" in e:
            bits.append(e["file"])
        if "page" in e and "file" not in e:
            bits.append(str(e["page"]))
        if "match_key" in e:
            bits.append(f"match={e['match_key'].split('/')[-2][:8]}..")
        for k in ("round", "tick", "player_id", "value", "rank_old", "rank_new", "rank_change",
                  "player_id_fixed", "expected_change"):
            if k in e and e[k] is not None:
                bits.append(f"{k}={g(e[k])}")
        parts.append("(" + ", ".join(bits) + ")")
    return "; ".join(parts) if parts else "-"


def flag_line(f):
    name = f["flag"]
    det = f.get("detail", {})
    if name == "constant":
        s = f"constant, only value {g(det.get('value'))}"
    elif name == "all_null":
        s = "all values null"
    elif name == "over_50pct_null":
        s = f"null rate {pct(det.get('null_rate', 0))}"
    elif name == "sentinel_like_value":
        s = "sentinel-like values: " + ", ".join(f"{g(d['value'])} x{d['count']:,}" for d in det)
    elif name == "sentinel_like_string":
        s = "sentinel-like strings: " + ", ".join(f"'{d['value']}' x{d['count']:,}" for d in det)
    elif name == "spike_top_value_gt_20pct":
        s = f"spike: value {g(det.get('value'))} holds {pct(det.get('share', 0))} ({det.get('count', 0):,} rows)"
    elif name == "impossible_value_for_name":
        s = "impossible for name: " + "; ".join(f"{d['rule']} x{d['count']:,}" for d in det)
    elif name == "schema_drift":
        s = ("schema drift: column absent or type-changed on some pages: " +
             "; ".join(f"{p}={t}" for p, t in det.get("dtype_or_presence_by_page", {}).items()) +
             f"; rows without the column: {det.get('rows_missing_column', 0):,}")
    else:
        s = name + " " + json.dumps(det, default=str)
    return s


def column_row(name, st):
    dt = st.get("dtype", "?")
    if st.get("dtype_by_page"):
        dt = dt + " (drift)"
    nr = st.get("null_rate_total", st.get("null_rate", 0))
    distinct = st.get("distinct_nonnull", "-")
    mn = g(st.get("min")) if "min" in st else "-"
    mx = g(st.get("max")) if "max" in st else "-"
    md = g(st.get("median")) if "median" in st else "-"
    top = "-"
    if isinstance(distinct, int) and distinct <= 50 and st.get("top_values"):
        top = "; ".join(f"{g(t['value'])}:{t['count']:,}({pct(t['share'])})" for t in st["top_values"])
    flags = ", ".join(f["flag"] for f in st.get("flags", [])) or "-"
    return f"| `{name}` | {dt} | {pct(nr)} | {distinct} | {mn} | {mx} | {md} | {top} | {flags} |"


def render_channel(ch, s, cur):
    L = []
    m = s["match_level"]["per_match"]
    L.append(f"## {ch}")
    L.append("")
    L.append(f"- Tome folder: `{s['tome_folder']}`, pages: {s['pages']}, rows: {s['rows']:,}")
    L.append(f"- Rows by page: {', '.join(f'{x:,}' for x in s.get('rows_by_page', []))}")
    L.append(f"- Distinct match_key: {m['distinct_match_key']:,} of {m['universe_matches_header']:,} header matches")
    L.append(f"- Matches with zero rows in this channel: {m['zero_row_matches']:,}; match_key not in header: {m['match_key_not_in_header']}")
    L.append(f"- Rows per match (matches with rows): min {g(m['rows_per_match_min'])}, p25 {g(m['rows_per_match_p25'])}, "
             f"median {g(m['rows_per_match_median'])}, p75 {g(m['rows_per_match_p75'])}, max {g(m['rows_per_match_max'])}")
    ov = s["match_level"]["page_key_overlap"]
    L.append(f"- Matches appearing on more than one page: {sum(ov.values()) if ov else 0}")
    L.append("")
    L.append("### Pages")
    L.append("")
    L.append("| file | rows | distinct match_key | null match_key |")
    L.append("|---|---:|---:|---:|")
    for p in s["match_level"]["pages"]:
        L.append(f"| `{p['file']}` | {p['rows']:,} | {p['distinct_match_key']:,} | {p['null_match_key']} |")
    L.append("")
    if s.get("schema_drift"):
        L.append("### Schema drift (checked across every page)")
        L.append("")
        for c, per in s["schema_drift"].items():
            L.append(f"- `{c}`: " + ", ".join(f"{k[-5:]}={v}" for k, v in per.items()))
        L.append(f"- Column order differs across pages in {s['column_order_variants']} variant(s) (cosmetic).")
        L.append("")
    L.append("### Columns")
    L.append("")
    L.append("Null rate includes NaN. Top 5 shown only for columns with 50 or fewer distinct non-null values.")
    L.append("")
    L.append("| column | dtype | null rate | distinct | min | max | median | top 5 | flags |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for c, st in s["columns"].items():
        L.append(column_row(c, st))
    L.append("")
    L.append("### Flags (counts and examples)")
    L.append("")
    any_flag = False
    for c, st in s["columns"].items():
        for f in st.get("flags", []):
            any_flag = True
            L.append(f"- `{c}` **{f['flag']}**: {flag_line(f)}. Examples: {ex_str(f.get('examples', []))}")
    if not any_flag:
        L.append("- none")
    L.append("")
    L.append("### Identical columns")
    L.append("")
    grp = s.get("identical_column_groups", [])
    L.append("- " + ("; ".join("`" + "` = `".join(gp) + "`" for gp in grp) if grp else "none (every column pair differs in at least one row)"))
    L.append("")
    d = s["duplicates"]
    L.append("### Duplicate rows")
    L.append("")
    L.append(f"- Exact duplicate rows (all columns identical): {d['exact_full_row_extra_copies']:,} extra copies in {d['exact_full_row_groups']:,} groups"
             f"; examples: {ex_str(d['exact_full_row_examples'])}")
    L.append(f"- Key duplicates on ({', '.join(d['key_dup_key'])}): {d['key_dup_extra_copies']:,} extra copies, "
             f"{d['key_dup_match_round_pairs']:,} (match, round) pairs affected; examples: {ex_str(d['key_dup_examples'])}")
    L.append("")
    L.append("### Matches far from the channel median row count (top 3)")
    L.append("")
    L.append(f"- Median rows per match (over matches with rows): {g(m['rows_per_match_median'])}")
    for t in m["top3_far_from_median_with_rows"]:
        L.append(f"  - `{t['match_key'].split('/')[-2][:8]}..` rows={t['rows']:,} |dev|={g(t['abs_dev_from_median'])}")
    L.append(f"- Zero-row matches: {m['zero_row_matches']:,} (examples: {', '.join(x.split('/')[-2][:8] for x in m['zero_row_match_examples'])})")
    L.append("")
    if s.get("extras"):
        L.append("### Cross-column and consistency checks")
        L.append("")
        for e in s["extras"]:
            L.append(f"- **{e['check']}**: " + "; ".join(f"{k}={g(v) if not isinstance(v, (list, dict)) else json.dumps(v, default=str)[:400]}"
                                                      for k, v in e.items() if k not in ("check", "examples")))
            if e.get("examples"):
                L.append(f"  - examples: {ex_str(e['examples']) if isinstance(e['examples'], list) and e['examples'] and isinstance(e['examples'][0], dict) and 'match_key' in e['examples'][0] else json.dumps(e['examples'], default=str)[:600]}")
        L.append("")
    L.append("### Commentary")
    L.append("")
    for line in cur.get("commentary", []):
        L.append(f"- {line}")
    L.append("")
    L.append("### What was checked")
    L.append("")
    for line in cur.get("checked", []):
        L.append(f"- {line}")
    L.append("")
    return L


def main():
    stats = json.load(open(STATS, encoding="utf-8"))
    cur_all = json.load(open(CURATION, encoding="utf-8"))
    L = []
    L.append("# Economy channel group: EDA profile")
    L.append("")
    L.append("Tome: `tomes/2026-10-01_to_08/tome/csds`, channels: item_equip, item_pickup, item_dropped, item_refund, rank_update.")
    L.append("Match universe: header tome (1752 matches). Scripts: `scripts/eda/profile_economy.py`, `scripts/eda/render_economy.py`.")
    L.append("")
    L.append("## Goofy list (top 5, ranked by how much they could mislead a model)")
    L.append("")
    for i, item in enumerate(cur_all["goofy_top5"], 1):
        L.append(f"{i}. **{item['title']}** ({item['channel']}). {item['detail']}")
    L.append("")
    L.append("## Group notes")
    L.append("")
    for line in cur_all.get("group_notes", []):
        L.append(f"- {line}")
    L.append("")
    for ch in ["item_equip", "item_pickup", "item_dropped", "item_refund", "rank_update"]:
        if ch in stats["channels"]:
            L += render_channel(ch, stats["channels"][ch], cur_all["channels"].get(ch, {}))
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    out = dict(stats)
    out["goofy_top5"] = cur_all["goofy_top5"]
    out["group_notes"] = cur_all.get("group_notes", [])
    for ch, cur in cur_all["channels"].items():
        if ch in out["channels"]:
            out["channels"][ch]["commentary"] = cur.get("commentary", [])
            out["channels"][ch]["checked"] = cur.get("checked", [])
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
