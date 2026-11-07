"""Build notes/spec/schema_2024_vs_2026.md from the sampled schema JSON.

Reads only the scratchpad JSON produced by collect_schemas.py. Writes only the markdown report.
"""
import collections
import itertools
import json
import re

SAMPLE = (
    r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
    r"\7308a8e7-a47e-4034-b140-5c511acb0aa4\scratchpad\schema_sample.json"
)
MD = r"G:\pureskill\csds-cs2\notes\spec\schema_2024_vs_2026.md"
INDEX_COL = "__index_level_0__"

INT = {
    "int8": (8, True), "int16": (16, True), "int32": (32, True), "int64": (64, True),
    "uint8": (8, False), "uint16": (16, False), "uint32": (32, False), "uint64": (64, False),
}
FLT = {"halffloat": 16, "float": 32, "double": 64}
STR = {"string", "large_string", "string_view"}


def dict_value(t):
    m = re.match(r"dictionary<values=(.*?), indices=", t)
    return m.group(1) if m else t


def family(t):
    if t in INT:
        return "int"
    if t in FLT:
        return "float"
    if t == "bool":
        return "bool"
    if t.startswith("dictionary<"):
        return "dict"
    if t in STR:
        return "string"
    if t.startswith(("timestamp", "date", "time")):
        return "temporal"
    return "other"


def risk(a, b):
    """Risk of reading data typed `a` (2024) into a declared type `b` (2026)."""
    if a == b:
        return 0, "identical type"
    if a == "null" or b == "null":
        return 25, "null type in some matches (all-null column); no value conversion"
    fa, fb = family(a), family(b)
    if fa == "dict" or fb == "dict":
        if dict_value(a) == dict_value(b):
            return 40, "dictionary-encoded in one side; same value type"
        return 75, "dictionary-encoded and different value type"
    if (fa == "bool") != (fb == "bool"):
        return 95, "bool vs non-bool: stored values change meaning"
    if fa == "int" and fb == "int":
        ba, sa = INT[a]
        bb, sb = INT[b]
        if ba > bb:
            return 100 + (ba - bb) + (15 if sa != sb else 0), "narrowing int width: overflow -> NarrowingError"
        if ba == bb:
            return 90, "same width, signedness change: upper-half values overflow"
        if sa and not sb:
            return 75, "signed to wider unsigned: negative values wrap"
        return 10, "int widening (safe)"
    if fa == "float" and fb == "float":
        if FLT[a] > FLT[b]:
            return 70, "float64 to float32: precision loss and range overflow"
        return 10, "float widening (safe)"
    if fa == "int" and fb == "float":
        return (60 if INT[a][0] == 64 else 45), "int to float: inexact above 2^53 for int64"
    if fa == "float" and fb == "int":
        return 85, "float to int: fractional values truncated, overflow"
    if fa == "string" and fb == "string":
        return 10, "string width variant (safe)"
    if "temporal" in (fa, fb):
        return 35, "temporal type or unit changed"
    return 50, "unclassified type change: review"


def worst(sa, sb):
    """Highest-risk pair from two type sets. Returns (score, note, a, b)."""
    best = (-1, "", "", "")
    for a, b in itertools.product(sorted(sa), sorted(sb)):
        sc, why = risk(a, b)
        if sc > best[0]:
            best = (sc, why, a, b)
    return best


def tier(score):
    if score >= 100:
        return "HIGH (overflow)"
    if score >= 85:
        return "HIGH"
    if score >= 70:
        return "MEDIUM-HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def col_tier(score):
    if score >= 100:
        return "HIGH (missing column)"
    if score >= 70:
        return "MEDIUM-HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def code(items):
    return ", ".join("`%s`" % i for i in items) if items else "none"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def load():
    with open(SAMPLE, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    data = load()
    f24 = data["folders"]["2024"]
    f26 = data["folders"]["2026"]
    n24, n26 = len(f24), len(f26)

    # ---- per-year aggregates ------------------------------------------------
    decl24 = collections.Counter()
    decl26 = collections.Counter()
    pres = {"2024": collections.defaultdict(collections.Counter),
            "2026": collections.defaultdict(collections.Counter)}
    ptypes = {"2024": collections.defaultdict(lambda: collections.defaultdict(set)),
              "2026": collections.defaultdict(lambda: collections.defaultdict(set))}
    mnames = {"2024": collections.defaultdict(set), "2026": collections.defaultdict(set)}
    mtypes = {"2024": collections.defaultdict(lambda: collections.defaultdict(set)),
              "2026": collections.defaultdict(lambda: collections.defaultdict(set))}
    avail = {"2024": collections.defaultdict(collections.Counter),
             "2026": collections.defaultdict(collections.Counter)}
    failures = {"2024": [], "2026": []}
    nfiles = {"2024": 0, "2026": 0}
    qa_mismatch = collections.Counter()
    qa_examples = []
    mvar = {"2024": collections.defaultdict(set), "2026": collections.defaultdict(set)}
    mpair = {"2024": collections.Counter(), "2026": collections.Counter()}

    for year, folders in (("2024", f24), ("2026", f26)):
        decl = decl24 if year == "2024" else decl26
        for f in folders:
            for ch, e in f["channels"].items():
                decl[ch] += 1
                avail[year][ch]["False" if e["available"] is False else
                                ("<absent>" if e["available"] == "<absent>" else "True")] += 1
                for name, mtype, _n, _o in e["manifest_columns"]:
                    mnames[year][ch].add(name)
                    mtypes[year][ch][name].add(mtype)
                mvar[year][ch].add(tuple(c[0] for c in e["manifest_columns"]))
                nfiles[year] += 1
                if "error" in e:
                    failures[year].append([f["day"], f["match"], ch, e["error"]])
                    continue
                seen = set()
                pq_type = {}
                for name, typ, _n in e["fields"]:
                    ptypes[year][ch][name].add(typ)
                    pq_type[name] = typ
                    seen.add(name)
                for name in seen:
                    pres[year][ch][name] += 1
                for name, mtype, _n, _o in e["manifest_columns"]:
                    if mtype is not None and name in pq_type:
                        mpair[year][(str(mtype), pq_type[name])] += 1
                m = [c[0] for c in e["manifest_columns"]]
                p = [c[0] for c in e["fields"]]
                if m != p:
                    qa_mismatch[year] += 1
                    if len(qa_examples) < 6:
                        qa_examples.append([year, f["match"], ch,
                                            code(sorted(set(m) - set(p))),
                                            code(sorted(set(p) - set(m)))])

    pn24 = {ch: set(cols) for ch, cols in ptypes["2024"].items()}
    pn26 = {ch: set(cols) for ch, cols in ptypes["2026"].items()}
    decl_only24 = sorted(set(decl24) - set(decl26))
    decl_only26 = sorted(set(decl26) - set(decl24))
    both = sorted(set(decl24) & set(decl26))

    # ---- item 1: channel presence -----------------------------------------
    item1 = []
    for ch in decl_only24:
        item1.append([60, "MEDIUM", "2024 only", "`%s`" % ch,
                      "%d of %d" % (decl24[ch], n24), "not declared",
                      "2024 channel the 2026 list drops; a 2026 reader will ignore it or fail if strict"])
    for ch in decl_only26:
        flagged_false = avail["2026"][ch].get("False", 0)
        if flagged_false > n26 / 2:
            score, note = 60, "2026 channel the 2024 data lacks; manifest flags it unavailable (False) in most 2026 matches"
        else:
            score, note = 80, "2026 channel the 2024 data lacks; not flagged unavailable in most 2026 matches"
        item1.append([score, tier(score), "2026 only", "`%s`" % ch, "not declared",
                      "%d of %d" % (decl26[ch], n26), note])
    item1.sort(key=lambda r: (-r[0], r[3]))

    # ---- item 2: column names present in one year only (shared channels) --
    grouped = collections.defaultdict(lambda: {"channels": [], "score": 0, "note": ""})

    def add_col(level, present, col, ch, score, note, cnt=None, total=None, yr=None):
        key = (level, present, col)
        g = grouped[key]
        g["channels"].append(ch)
        g["score"] = max(g["score"], score)
        g["note"] = note
        if cnt is not None:
            g.setdefault("cnts", []).append((cnt, total, yr))

    index_24 = [ch for ch in both if pres["2024"][ch].get(INDEX_COL, 0) > 0]
    index_26 = [ch for ch in both if pres["2026"][ch].get(INDEX_COL, 0) > 0]

    for ch in both:
        for col in sorted(pn26[ch] - pn24[ch]):
            add_col("parquet", "2026 only", col, ch, 100,
                    "no sampled 2024 parquet has it; a 2026 reader gets a missing column",
                    cnt=pres["2026"][ch][col], total=decl26[ch], yr="2026")
        for col in sorted(pn24[ch] - pn26[ch]):
            if col == INDEX_COL:
                continue
            add_col("parquet", "2024 only", col, ch, 60,
                    "extra column in 2024 parquet; 2026 readers ignore it or reject it",
                    cnt=pres["2024"][ch][col], total=decl24[ch], yr="2024")
        for col in sorted(mnames["2024"][ch] - mnames["2026"][ch]):
            if col in pn24[ch]:
                add_col("manifest", "2024 only", col, ch, 40,
                        "2024 manifest declares it; 2026 manifest does not")
            else:
                add_col("manifest", "2024 only", col, ch, 70,
                        "2024 manifest declares it but the 2024 parquet does not contain it")
        for col in sorted(mnames["2026"][ch] - mnames["2024"][ch]):
            add_col("manifest", "2026 only", col, ch, 40,
                    "2026 manifest declares it; 2024 manifest does not")

    item2 = []
    for (level, present, col), g in grouped.items():
        note = g["note"]
        if g.get("cnts"):
            lo = min(c for c, _t, _y in g["cnts"])
            hi = max(c for c, _t, _y in g["cnts"])
            tot, yr = g["cnts"][0][1], g["cnts"][0][2]
            span = "%d" % lo if lo == hi else "%d-%d" % (lo, hi)
            note += "; %s parquet has it in %s of %d sampled folders" % (yr, span, tot)
        item2.append([g["score"], col_tier(g["score"]), level, present, "`%s`" % col,
                      len(g["channels"]), code(sorted(g["channels"])), note])
    item2.sort(key=lambda r: (-r[0], r[4]))

    # ---- item 3: parquet type differences ---------------------------------
    type_rows = []
    for ch in both:
        for col in sorted(pn24[ch] & pn26[ch]):
            s24, s26 = ptypes["2024"][ch][col], ptypes["2026"][ch][col]
            if s24 == s26:
                continue
            sc, why, a, b = worst(s24, s26)
            type_rows.append((sc, ch, col, ", ".join(sorted(s24)), ", ".join(sorted(s26)),
                              "`%s` -> `%s`" % (a, b), why))
    type_rows.sort(key=lambda r: (-r[0], r[1], r[2]))

    # manifest-level typed differences (both years declare a type): check against parquet table
    mdiff = []
    for ch in both:
        for col in sorted(mnames["2024"][ch] & mnames["2026"][ch]):
            s24 = {t for t in mtypes["2024"][ch][col] if t is not None}
            s26 = {t for t in mtypes["2026"][ch][col] if t is not None}
            if s24 and s26 and s24 != s26:
                mdiff.append((ch, col, s24, s26))
    pq_keys = {(r[1], r[2]) for r in type_rows}
    mdiff_not_in_pq = [m for m in mdiff if (m[0], m[1]) not in pq_keys]
    mdiff_one_sided = 0
    for ch in both:
        for col in sorted(mnames["2024"][ch] & mnames["2026"][ch]):
            t24 = {t for t in mtypes["2024"][ch][col] if t is not None}
            t26 = {t for t in mtypes["2026"][ch][col] if t is not None}
            if bool(t24) != bool(t26):
                mdiff_one_sided += 1

    # ---- within-year parquet type variance ---------------------------------
    wy_rows = []
    for year in ("2024", "2026"):
        for ch in sorted(ptypes[year]):
            for col in sorted(ptypes[year][ch]):
                s = ptypes[year][ch][col]
                if len(s) > 1:
                    sc, why, a, b = worst(s, s)
                    wy_rows.append((sc, year, ch, col, ", ".join(sorted(s)),
                                    "`%s` -> `%s`" % (a, b), why))
    wy_rows.sort(key=lambda r: (-r[0], r[1], r[2], r[3]))

    # ---- within-year parquet column presence variance ----------------------
    pvar = []
    for year, decl in (("2024", decl24), ("2026", decl26)):
        for ch in sorted(pres[year]):
            for col, cnt in sorted(pres[year][ch].items()):
                if cnt < decl[ch]:
                    pvar.append([year, "`%s`" % ch, "`%s`" % col, "%d of %d" % (cnt, decl[ch])])

    mvar24 = sorted(ch for ch, v in mvar["2024"].items() if len(v) > 1)
    mvar26 = sorted(ch for ch, v in mvar["2026"].items() if len(v) > 1)

    idx_missing_24 = [ch for ch in both if ch not in index_24]

    # ---- counts for summary ---------------------------------------------------
    n_hi = sum(1 for r in type_rows if r[0] >= 100)
    n_mid_hi = sum(1 for r in type_rows if 70 <= r[0] < 100)
    n_medium = sum(1 for r in type_rows if 40 <= r[0] < 70)
    n_low = sum(1 for r in type_rows if r[0] < 40)
    n_missing_2026_cols = sum(1 for r in item2 if r[2] == "parquet" and r[3] == "2026 only")
    n_missing_2026_cols_ch = sum(r[5] for r in item2 if r[2] == "parquet" and r[3] == "2026 only")

    # ---- markdown --------------------------------------------------------------
    lines = []
    add = lines.append
    add("# 2024 vs 2026 csds schema comparison")
    add("")
    add("Read-only comparison of csds channel manifests and parquet schemas. No data rows were read, "
        "and nothing under E: was written.")
    add("")
    add("## Verdict")
    add("")
    add("2024 data cannot use the 2026 channel list unchanged. The channel lists differ, the 2024 parquet "
        "files carry an extra `__index_level_0__` column, and several shared columns change integer width, "
        "signedness, or type between years. The highest-risk items are the `player_status` and `player_vector` "
        "integer narrowings: 2024 int64 data read into the int8, int16 and int32 types that some 2026 matches "
        "declare. Next are the bool/int changes and the parquet columns that 2026 has and the 2024 files do not.")
    add("")
    add("## Sample and method")
    add("")
    add("- 2024: `2024\\02\\02` and `2024\\02\\12` (tree `csds_tail_recovered_20261011`), 5 match folders each.")
    add("- 2026: `2026\\10\\01` through `2026\\10\\08`, 5 match folders each (40 total).")
    add("- A folder qualifies when its `csds` manifest is gzip JSON that parses and every channel it declares "
        "has a parquet file of that name. Picks are a seeded shuffle, first 5 that qualify per day. "
        "Every candidate that was checked qualified.")
    add("- Parquet types come from `pyarrow.parquet.read_schema` (footer only). The manifest `type` is the csds "
        "logical type (for example `int`, `int64`, `string`). Columns with origin `merged` or `calculated` "
        "have no `type` key, so manifest type comparisons skip them.")
    add("- Risk scores read each 2024 type into the 2026 declared type. Higher is riskier. Int narrowing scores "
        "100 plus the bit drop (+15 if signedness also changes). Bool vs non-bool scores 95. Same-width sign "
        "change scores 90. Float to int scores 85. Dictionary encoding scores 40, or 75 with a different value "
        "type. Float64 to float32 scores 70. Null-typed columns score 25. Widening scores 10. Identical scores 0.")
    add("- Coverage limit: 10 of 1,192 2024 folders and 40 of 1,752 2026 folders (Oct 1-8) were read. Other "
        "columns or types may vary in the unsampled folders.")
    add("")
    add("Sample counts: " + "; ".join(
        "%s: %d of %d match folders qualify" % (d["day"], d["chosen"], d["match_folders"]) for d in data["days"]))
    add("")
    add("Schemas read: 2024 = %d parquet files across %d folders; 2026 = %d across %d folders." % (
        nfiles["2024"], n24, nfiles["2026"], n26))
    add("")
    add("## Summary")
    add("")
    add("- Channels declared in both years: %d. 2024 only: %d. 2026 only: %d." % (
        len(both), len(decl_only24), len(decl_only26)))
    add("- Column-name differences on shared channels: %d grouped rows (section 2). %d are parquet columns that "
        "2026 has and no sampled 2024 file has (HIGH). Column presence also varies within each year (QA)." % (
            len(item2), n_missing_2026_cols))
    tier_counts = collections.Counter(tier(r[0]) for r in type_rows)
    add("- Shared parquet columns whose type set differs between years: %d. By risk: %s." % (
        len(type_rows), ", ".join("%s %d" % (t, tier_counts[t]) for t in
                                  ["HIGH (overflow)", "HIGH", "MEDIUM-HIGH", "MEDIUM", "LOW"])))
    add("- `__index_level_0__` (a pandas index column) appears in the 2024 parquet of %d of %d shared "
        "channels and in no 2026 parquet." % (len(index_24), len(both)))
    add("- 2024 parquet schema read failures in the sample: %d of %d files." % (
        len(failures["2024"]), nfiles["2024"]))
    add("")

    add("## 1. Channel list differences (ranked by risk)")
    add("")
    if item1:
        add("\n".join(table(["Risk score", "Risk", "Direction", "Channel", "2024 folders", "2026 folders", "Note"],
                            [r for r in item1])))
    else:
        add("None.")
    add("")
    add("Shared channels (both years): %d. Of these, declared in every sampled folder: 2024 %d of %d, "
        "2026 %d of %d." % (
            len(both), sum(1 for ch in both if decl24[ch] == n24), len(both),
            sum(1 for ch in both if decl26[ch] == n26), len(both)))
    add("")
    avail_rows = []
    for ch in sorted(set(avail["2024"]) | set(avail["2026"])):
        a24 = dict(avail["2024"].get(ch, {}))
        a26 = dict(avail["2026"].get(ch, {}))
        if set(a24) <= {"<absent>"} and set(a26) <= {"<absent>"}:
            continue
        avail_rows.append(["`%s`" % ch,
                           ", ".join("%s: %d" % kv for kv in sorted(a24.items())) if a24 else "-",
                           ", ".join("%s: %d" % kv for kv in sorted(a26.items())) if a26 else "-"])
    if avail_rows:
        add("Manifest `available` flags (folder counts; `<absent>` means the key is not set):")
        add("")
        add("\n".join(table(["Channel", "2024", "2026"], avail_rows)))
        add("")
    add("")

    add("## 2. Column names present in one year only (shared channels, ranked by risk)")
    add("")
    if item2:
        add("\n".join(table(["Risk score", "Risk", "Level", "Present in", "Column", "Channels", "Channel list",
                             "Note"], item2)))
    else:
        add("None.")
    add("")
    add("`__index_level_0__` is omitted from the table above. It is present in the 2024 parquet of %d shared "
        "channels (%s) and absent from every 2026 parquet in the sample. It is an index column written by "
        "pandas, so it is a schema artifact rather than csds data." % (
            len(index_24), code(sorted(index_24))))
    add("")
    add("Channels missing from the 2024 parquet entirely (no `__index_level_0__`): %s." % code(sorted(idx_missing_24)))
    add("")

    add("## 3. Parquet column types that differ between years (ranked by risk)")
    add("")
    add("Direction is 2024 data read into the 2026 declared type. The Pair column is the riskiest "
        "2024 type -> 2026 type combination. A column can be `int64` in 2024 but `int64, int8` in 2026: "
        "that means some 2026 matches store it as int8.")
    add("")
    if type_rows:
        rows = []
        for i, (sc, ch, col, t24, t26, pair, why) in enumerate(type_rows, 1):
            rows.append([i, sc, tier(sc), "`%s`" % ch, "`%s`" % col, "`%s`" % t24, "`%s`" % t26, pair, why])
        add("\n".join(table(["Rank", "Score", "Risk", "Channel", "Column", "2024 types", "2026 types",
                             "Riskiest pair", "Note"], rows)))
    else:
        add("None.")
    add("")
    add("Manifest-level check: %d columns declare a typed manifest `type` in both years with different type sets. "
        "Only %d of these also appear in the parquet table above. Manifest logical types and parquet physical "
        "types are not interchangeable (see the QA pair table: for example manifest `int32` sits on parquet "
        "`int64`). %d columns declare a type on one side only, which is structural: `merged` and `calculated` "
        "columns carry no `type` key in the 2024 manifests. The parquet table is the authoritative type "
        "comparison. The manifest-only differences are not listed individually." % (
            len(mdiff), len(mdiff) - len(mdiff_not_in_pq), mdiff_one_sided))
    add("")

    add("## 4. 2024 parquet files that fail to read their schema")
    add("")
    add("Checked %d 2024 parquet files across %d sampled folders." % (nfiles["2024"], n24))
    add("")
    if failures["2024"]:
        add("\n".join(table(["Day", "Match", "Channel", "Error"], failures["2024"])))
    else:
        add("None. Every 2024 parquet footer in the sample read cleanly.")
    add("")
    add("For reference, 2026 read failures in the sample: %d of %d." % (len(failures["2026"]), nfiles["2026"]))
    add("")

    add("## QA checks")
    add("")
    add("- Manifest column list vs parquet field list, per channel and folder: 2024 mismatches %d of %d, "
        "2026 mismatches %d of %d." % (qa_mismatch["2024"], nfiles["2024"], qa_mismatch["2026"], nfiles["2026"]))
    add("")
    if qa_examples:
        add("\n".join(table(["Year", "Match", "Channel", "In manifest only", "In parquet only"], qa_examples)))
        add("")
    add("- Manifest column list varies across folders within a year: 2024 %s; 2026 %s." % (
        code(mvar24), code(mvar26)))
    add("")
    add("- Manifest logical type vs parquet physical type for the same column in the same file (counts; "
        "a manifest `int` on an `int64` parquet is a naming mismatch, not a width change):")
    add("")
    pair_rows = []
    for year in ("2024", "2026"):
        for (mt, pt), cnt in sorted(mpair[year].items(), key=lambda kv: -kv[1]):
            pair_rows.append([year, "`%s`" % mt, "`%s`" % pt, cnt])
    add("\n".join(table(["Year", "Manifest type", "Parquet type", "Columns"], pair_rows)))
    add("")
    add("- Parquet column presence varies across folders within a year (present in some folders only). "
        "Each row is a column that some sampled folders of that year do not contain:")
    add("")
    if pvar:
        add("\n".join(table(["Year", "Channel", "Column", "Folders with column"], pvar)))
    else:
        add("None.")
    add("")
    add("- Within-year parquet type variance (same column, more than one type inside one year), ranked by risk:")
    add("")
    if wy_rows:
        add("\n".join(table(["Risk score", "Risk", "Year", "Channel", "Column", "Types", "Riskiest pair", "Note"],
                            [[sc, tier(sc), yr, "`%s`" % ch, "`%s`" % col, "`%s`" % ts, pr, why]
                             for sc, yr, ch, col, ts, pr, why in wy_rows])))
    else:
        add("None.")
    add("")

    text = "\n".join(lines) + "\n"
    with open(MD, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)

    print("wrote", MD)
    print("item1 rows:", len(item1), " item2 rows:", len(item2), " type rows:", len(type_rows))
    print("type risk: HIGH-overflow", n_hi, "MEDIUM-HIGH", n_mid_hi, "MEDIUM", n_medium, "LOW", n_low)
    print("manifest typed diffs:", len(mdiff), "not in parquet table:", len(mdiff_not_in_pq),
          "one-sided:", mdiff_one_sided)
    print("index col: 2024 channels", len(index_24), "2026 channels", len(index_26))
    print("failures 2024:", len(failures["2024"]), "of", nfiles["2024"], " 2026:", len(failures["2026"]))
    print("qa mismatch:", dict(qa_mismatch), " presence variance rows:", len(pvar),
          " within-year type rows:", len(wy_rows))
    print("mvar24", mvar24, "mvar26", mvar26)


if __name__ == "__main__":
    main()
