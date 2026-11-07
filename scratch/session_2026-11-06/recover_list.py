import csv, collections
OUT = "/home/sagemaker-user/adam/builder/fixes/fix332/"
s3f = set()
for line in open(OUT + "s3_folders_2026_10.txt"):
    dd, uu = line.split()
    s3f.add(("2026/10/" + dd, uu.rstrip("/")))
miss = [r for r in csv.DictReader(open(OUT + "missing_matches.csv"))]
rows = []
for r in miss:
    mk = r["match_key"]
    parts = mk.split("/")  # csds, YYYY, MM, DD, uuid, csds
    ymd = "/".join(parts[1:4])
    day = parts[1] + "/" + parts[2] + "/" + parts[3]
    uu = parts[4]
    present = (parts[1] + "/" + parts[2] + "/" + parts[3], uu) in s3f
    rows.append({"match_key": mk, "src": r["src"], "export_rounds": r["export_rounds"],
                 "month": parts[1] + "/" + parts[2], "day": parts[3], "uuid": uu,
                 "s3_folder_present": present})
cnt = collections.Counter((r["month"], r["s3_folder_present"]) for r in rows)
print("missing by month/present", sorted(cnt.items()))
rec = [r for r in rows if r["s3_folder_present"]]
print("recoverable matches", len(rec), "rounds", sum(int(r["export_rounds"]) for r in rec))
print("not recoverable matches", len(rows) - len(rec), "rounds", sum(int(r["export_rounds"]) for r in rows if not r["s3_folder_present"]))
with open(OUT + "recover_list.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
