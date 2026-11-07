import os, re, csv, json, collections, time
import polars as pl

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
G = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data"
rows = list(csv.DictReader(open(SCR + "undetermined_matches.csv", encoding="utf-8")))
TARGET = {r["match_key"] for r in rows}
TUUID = {k.split("/")[4]: k for k in TARGET}
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
KEYCOLS = ["match_key", "match_id", "matchkey", "key", "uuid", "csds_match", "source_key"]
t0 = time.time()
groups = collections.defaultdict(lambda: {"files": 0, "covered": set(), "team_code_files": [], "key_cols": collections.Counter(), "distinct_all": set(), "errors": []})
scanned = 0
for root, dirs, fnames in os.walk(G):
    if re.search(r"holdout|frozen|test_set|heldout", root, re.I):
        dirs[:] = []
        continue
    for f in fnames:
        if not f.endswith(".parquet"):
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, G).replace("\\", "/")
        top = rel.split("/")[0] if "/" in rel else "(root)"
        g = groups[top]
        g["files"] += 1
        scanned += 1
        try:
            names = list(pl.scan_parquet(p).collect_schema().names())
        except Exception as e:
            g["errors"].append((rel, str(e)[:100])); continue
        if "team_code" in names:
            g["team_code_files"].append(rel)
        kc = [c for c in names if c.lower() in KEYCOLS]
        if not kc:
            kc = [c for c in names if "match" in c.lower()][:1]
        if not kc:
            continue
        c = kc[0]
        g["key_cols"][c] += 1
        try:
            vals = pl.scan_parquet(p).select(pl.col(c).cast(pl.Utf8)).unique().collect().get_column(c).drop_nulls().to_list()
        except Exception as e:
            g["errors"].append((rel, "read " + str(e)[:80])); continue
        for v in vals:
            if v in TARGET:
                g["covered"].add(v)
            else:
                m = UUID_RE.search(v)
                if m and m.group(0) in TUUID:
                    g["covered"].add(TUUID[m.group(0)])
        g["distinct_all"].update(vals[:0])  # placeholder to keep structure small
print("scanned", scanned, "elapsed", round(time.time() - t0, 1), flush=True)
out = {}
for top, g in sorted(groups.items()):
    out[top] = {"files": g["files"], "target_keys_covered": len(g["covered"]),
                "team_code_files": g["team_code_files"][:10], "team_code_file_count": len(g["team_code_files"]),
                "key_cols": dict(g["key_cols"]), "errors": g["errors"][:5], "error_count": len(g["errors"]),
                "sample": sorted(g["covered"])[:2]}
    print(top, json.dumps(out[top])[:400], flush=True)
json.dump(out, open(SCR + "g_group.json", "w"), indent=1)
print("WROTE", flush=True)
