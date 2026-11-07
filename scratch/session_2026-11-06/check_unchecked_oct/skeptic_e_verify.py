import csv, os, json, collections
import pyarrow.parquet as pq

E = "E:/csds_final/csds"
W = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"

rows = list(csv.DictReader(open(W + "undetermined_matches.csv", encoding="utf-8")))
print("input rows", len(rows), flush=True)

c = collections.Counter()
tc_vals = collections.Counter()
schema_printed = False
bad = []
by_month = collections.Counter()
mk_mismatch = 0
mk_checked = 0
per_key = []
for r in rows:
    mk = r["match_key"]
    p = mk.split("/")
    y, m, d, u = p[1], p[2], p[3], p[4]
    by_month[f"{y}/{m}"] += 1
    folder = os.path.join(E, y, m, d, u)
    if os.path.isdir(folder):
        c["folder_exists"] += 1
    else:
        c["folder_missing"] += 1
        bad.append(("folder_missing", mk))
        per_key.append((mk, "missing"))
        continue
    pi = os.path.join(folder, "player_info")
    if not os.path.isfile(pi):
        c["player_info_missing"] += 1
        bad.append(("player_info_missing", mk))
        per_key.append((mk, "no_player_info"))
        continue
    try:
        t = pq.read_table(pi)
    except Exception as e:
        c["player_info_unreadable"] += 1
        bad.append(("player_info_unreadable", mk, str(e)[:120]))
        per_key.append((mk, "unreadable"))
        continue
    names = t.column_names
    if not schema_printed:
        print("SCHEMA", names, flush=True)
        print("ROWS first file", t.num_rows, flush=True)
        schema_printed = True
    if "team_code" not in names:
        c["no_team_code_col"] += 1
        bad.append(("no_team_code_col", mk))
        per_key.append((mk, "no_team_code"))
        continue
    c["team_code_col_present"] += 1
    vals = t.column("team_code").to_pylist()
    for v in vals:
        tc_vals[str(v)] += 1
    if "match_key" in names:
        mk_checked += 1
        mvals = set(t.column("match_key").to_pylist())
        if mk not in mvals:
            mk_mismatch += 1
            bad.append(("match_key_mismatch", mk))
    nonnull_tc = sum(1 for v in vals if v is not None)
    per_key.append((mk, f"ok rows={t.num_rows} tc_nonnull={nonnull_tc}"))

summary = {
    "input_rows": len(rows),
    "by_year_month": dict(sorted(by_month.items())),
    "counters": dict(c),
    "team_code_value_counts_all_rows": dict(tc_vals),
    "player_info_with_match_key_col_checked": mk_checked,
    "match_key_mismatches_in_player_info": mk_mismatch,
    "bad_examples": bad[:20],
    "bad_total": len(bad),
}
with open(W + "skeptic_e_verify.json", "w") as f:
    json.dump(summary, f, indent=1)
with open(W + "skeptic_e_verify_per_key.tsv", "w") as f:
    f.write("match_key\tstatus\n")
    for mk, s in per_key:
        f.write(f"{mk}\t{s}\n")
print(json.dumps(summary, indent=1)[:4000], flush=True)
print("DONE", flush=True)
