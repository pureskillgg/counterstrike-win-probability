import os, csv, json, collections, time
import polars as pl

SCR = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
ROOT = "E:/csds_final/csds/"
rows = list(csv.DictReader(open(SCR + "undetermined_matches.csv", encoding="utf-8")))
t0 = time.time()
stats = collections.Counter()
by_month = collections.defaultdict(collections.Counter)
samples = {}
per_key = []
team_codes = collections.Counter()
nfiles_hist = collections.Counter()
for r in rows:
    k = r["match_key"]
    _, y, m, d, u, _ = k.split("/")
    folder = f"{ROOT}{y}/{m}/{d}/{u}"
    rec = {"key": k, "month": f"{y}/{m}"}
    exact = os.path.isdir(folder)
    rec["exact_dir"] = exact
    stats["exact_dir"] += exact
    by_month[f"{y}/{m}"]["exact_dir"] += exact
    if not exact:
        per_key.append(rec); continue
    try:
        files = os.listdir(folder)
    except Exception as e:
        files = []
    nfiles_hist[min(len(files), 60)] += 1
    rec["n_files"] = len(files)
    pi = os.path.join(folder, "player_info")
    has_pi = os.path.isfile(pi)
    rec["player_info_file"] = has_pi
    stats["player_info_file"] += has_pi
    by_month[f"{y}/{m}"]["player_info_file"] += has_pi
    if has_pi:
        try:
            with open(pi, "rb") as fh:
                head = fh.read(4); fh.seek(-4, 2); tail = fh.read(4)
            is_parquet = head == b"PAR1" and tail == b"PAR1"
            rec["parquet_magic"] = is_parquet
            stats["parquet_magic"] += is_parquet
            df = pl.read_parquet(pi, columns=["team_code", "round", "player_id"])
            rec["rows"] = df.height
            rec["has_team_code"] = "team_code" in df.columns
            stats["team_code_present"] += 1 if rec["has_team_code"] else 0
            by_month[f"{y}/{m}"]["team_code_present"] += 1 if rec["has_team_code"] else 0
            rec["rounds"] = df["round"].n_unique()
            rec["players"] = df["player_id"].n_unique()
            rec["team_code_nonnull"] = df["team_code"].drop_nulls().len()
            stats["rows_nonzero"] += 1 if df.height > 0 else 0
            for v, c in df["team_code"].value_counts().iter_rows():
                team_codes[str(v)] += c
            if "sample" not in samples:
                samples["sample"] = {"key": k, "folder": folder, "rows": df.height, "rounds": rec["rounds"], "players": rec["players"]}
        except Exception as e:
            rec["read_error"] = f"{type(e).__name__}: {str(e)[:160]}"
            stats["read_error"] += 1
    per_key.append(rec)
print("elapsed", round(time.time() - t0, 1), "s", flush=True)
print("stats", dict(stats), flush=True)
for mth, c in sorted(by_month.items()):
    print("month", mth, dict(c), flush=True)
print("team_code value counts (all rows)", dict(team_codes.most_common(12)), flush=True)
print("files-per-folder histogram (capped 60)", sorted(nfiles_hist.items())[:5], "...", sorted(nfiles_hist.items())[-3:], flush=True)
out = {"targets": len(rows), "stats": dict(stats), "by_month": {k: dict(v) for k, v in by_month.items()},
       "team_code_values": dict(team_codes.most_common(20)), "sample": samples.get("sample"),
       "missing_exact_dir_sample": [p["key"] for p in per_key if not p.get("exact_dir")][:10],
       "read_errors_sample": [p for p in per_key if p.get("read_error")][:10]}
json.dump(out, open(SCR + "local_e_verify.json", "w"), indent=1)
with open(SCR + "local_e_verify_per_key.tsv", "w", encoding="utf-8") as f:
    f.write("key\tmonth\texact_dir\tplayer_info_file\tparquet_magic\thas_team_code\trows\trounds\tplayers\tn_files\tread_error\n")
    for p in per_key:
        f.write("\t".join(str(p.get(c, "")) for c in ["key", "month", "exact_dir", "player_info_file", "parquet_magic", "has_team_code", "rows", "rounds", "players", "n_files", "read_error"]) + "\n")
print("WROTE", flush=True)
