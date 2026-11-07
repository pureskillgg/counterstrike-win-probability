import os, re, csv, json, collections, time
import polars as pl

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
G = "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data"
rows = list(csv.DictReader(open(SCR + "undetermined_matches.csv", encoding="utf-8")))
TARGET = {r["match_key"] for r in rows}
TARGET_UUID = {k.split("/")[4] for k in TARGET}
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
KEYCOLS = ("match_key", "match_id", "matchkey", "key", "uuid", "csds_match", "folder", "path", "source_key")

t0 = time.time()
files = []
skipped_holdout = 0
for root, dirs, fnames in os.walk(G):
    if re.search(r"holdout|frozen|test_set|heldout", root, re.I):
        skipped_holdout += len([f for f in fnames if f.endswith(".parquet")])
        dirs[:] = []
        continue
    for f in fnames:
        if f.endswith(".parquet"):
            files.append(os.path.join(root, f))
print("parquet files to scan", len(files), "skipped (holdout-like folders)", skipped_holdout, flush=True)

team_files = []
keycol_hits = []
errs = []
summary = collections.Counter()
for i, p in enumerate(files):
    rel = os.path.relpath(p, G).replace("\\", "/")
    try:
        names = pl.read_parquet_schema(p).keys() if hasattr(pl, "read_parquet_schema") else pl.scan_parquet(p).collect_schema().names()
        names = list(names)
    except Exception as e:
        try:
            names = list(pl.scan_parquet(p).collect_schema().names())
        except Exception as e2:
            errs.append((rel, str(e2)[:120])); continue
    has_team = "team_code" in names
    if has_team:
        team_files.append(rel)
        summary["team_code_files"] += 1
    kcols = [c for c in names if c.lower() in KEYCOLS or "match" in c.lower() or c.lower().endswith("_key")]
    if kcols:
        summary["has_key_like_col"] += 1
        covered = set()
        for c in kcols[:3]:
            try:
                vals = pl.scan_parquet(p).select(pl.col(c).cast(pl.Utf8)).unique().collect().get_column(c).to_list()
            except Exception:
                continue
            hit = 0
            for v in vals:
                if v is None: continue
                if v in TARGET:
                    covered.add(v)
                else:
                    m = UUID_RE.search(str(v))
                    if m and m.group(0) in TARGET_UUID:
                        covered.add(next(t for t in TARGET if t.split("/")[4] == m.group(0)))
            if covered: break
        if covered:
            keycol_hits.append({"file": rel, "key_cols": kcols[:3], "covered_targets": len(covered), "has_team_code": has_team, "sample": sorted(covered)[:2]})
    if i % 200 == 0:
        print("progress", i, round(time.time() - t0, 1), flush=True)
print("elapsed", round(time.time() - t0, 1), flush=True)
print("summary", dict(summary), "errors", len(errs), flush=True)
print("team_code files:", team_files[:50], flush=True)
print("files covering target keys:", len(keycol_hits), flush=True)
for h in keycol_hits[:40]:
    print("  ", h, flush=True)
json.dump({"scanned": len(files), "skipped_holdout_like": skipped_holdout, "summary": dict(summary),
           "team_code_files": team_files, "keycol_hits": keycol_hits[:200], "errors": errs[:50]},
          open(SCR + "g_scan.json", "w"), indent=1)
print("WROTE", flush=True)
