import os, re, sys, collections
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = r"G:\pureskill\csds-cs2"
for rel in [r"logs\swarm\E1\cache\player_info.parquet", r"logs\swarm\E1\cache\item_pickup.parquet", r"evac_2026-10-13\data\state_sample\players_1hz.parquet", r"data\combined\deaths.parquet"]:
    sch = pq.ParquetFile(os.path.join(ROOT, rel)).schema_arrow
    print("SCHEMA", rel, "rows", pq.ParquetFile(os.path.join(ROOT, rel)).metadata.num_rows, [f"{n}:{t}" for n, t in zip(sch.names, sch.types)])
# which files hold username-like path segments other than the own account (file names only, counts)
USERNAME_RE = re.compile(rb"(?i)(?:[A-Za-z]:[\/]+Users[\/]+|/home/|/Users/)([^\/\s\"'<>:|*?]{1,40})")
per_file = collections.Counter(); lens = collections.Counter(); forms = collections.Counter()
for sub in ["logs", "notes", "deliverables"]:
    for dp, dn, fn in os.walk(os.path.join(ROOT, sub)):
        for f in fn:
            ext = os.path.splitext(f)[1].lower()
            if ext in (".parquet", ".png", ".zip", ".pptx", ".docx", ".pdf", ".npy", ".npz", ".pkl", ".pyc"):
                continue
            try: data = open(os.path.join(dp, f), "rb").read()
            except Exception: continue
            for m in USERNAME_RE.finditer(data):
                n = m.group(1)
                if n.lower() in (b"<user>",): continue
                per_file[os.path.relpath(os.path.join(dp, f), ROOT)] += 1
                lens[len(n)] += 1
                forms["has_dot" if b"." in n else ("has_digit" if re.search(rb"\d", n) else "alpha_only")] += 1
print("OTHER_USERNAME_SEGMENT_FILES", len(per_file), "total", sum(per_file.values()))
for f, c in per_file.most_common(12): print("   ", c, f)
print("   length_hist", dict(lens))
print("   forms", dict(forms))
