import os, sys, csv, shutil, time
from concurrent.futures import ThreadPoolExecutor
SRC = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"
DST = r"G:\pureskill\csds-cs2\source_2024\csds\2024\02"
base = r"G:\pureskill\csds-cs2"
want = set(l.strip() for l in open(os.path.join(base, "scripts", "channels_build_nobom.txt"), encoding="utf-8-sig") if l.strip()) | {"header", "csds"}
rows = [r for r in csv.DictReader(open(os.path.join(base, "logs", "prescan_2024.csv"), encoding="utf-8")) if r["status"] == "ok" and r["day"] != "01"]
t0 = time.time()
def copy_match(r):
    s = os.path.join(SRC, r["day"], r["match_folder"])
    d = os.path.join(DST, r["day"], r["match_folder"])
    os.makedirs(d, exist_ok=True)
    copied = skipped = 0; problems = []
    for e in os.scandir(s):
        if not e.is_file() or e.name not in want:
            continue
        dp = os.path.join(d, e.name)
        if os.path.exists(dp) and os.path.getsize(dp) == e.stat().st_size:
            skipped += 1; continue
        try:
            shutil.copyfile(e.path, dp)
            if os.path.getsize(dp) != e.stat().st_size:
                problems.append(f"size:{e.name}")
            copied += 1
        except OSError as ex:
            problems.append(f"{e.name}:{ex}")
    return (r["day"], r["match_folder"], copied, skipped, ";".join(problems))
done = 0; results = []
with ThreadPoolExecutor(16) as ex:
    for res in ex.map(copy_match, rows):
        results.append(res); done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(rows)} t={time.time()-t0:.0f}s", flush=True)
files = sum(r[2] for r in results); already = sum(r[3] for r in results); bad = [r for r in results if r[4]]
print(f"done matches={len(results)} files_copied={files} already_present={already} problem_matches={len(bad)} t={time.time()-t0:.0f}s", flush=True)
with open(os.path.join(base, "logs", "copy_2024_problems.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["day", "match", "copied", "skipped", "problems"]); w.writerows(bad)
