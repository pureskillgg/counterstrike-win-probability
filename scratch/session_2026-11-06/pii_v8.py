import os, re, sys, zipfile, collections
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = r"G:\pureskill\csds-cs2\deliverables"
USER_RE = re.compile(rb"(?i)users[\/]+<user>")
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
c = collections.Counter()
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        if f.lower().endswith(".zip"):
            c["zip_archives"] += 1
            with zipfile.ZipFile(os.path.join(dp, f)) as zf:
                for info in zf.infolist():
                    if info.is_dir(): continue
                    c["members"] += 1
                    d = zf.read(info.filename)
                    if USER_RE.search(d): c["members_with_user_path"] += 1
                    if EMAIL_RE.search(d): c["members_with_email"] += 1
        elif os.path.splitext(f)[1].lower() in (".json", ".html", ".md", ".txt", ".csv", ".sh", ".py", ".sha256", ".svg"):
            c["text_files"] += 1
            if USER_RE.search(open(os.path.join(dp, f), "rb").read()): c["text_files_with_user_path"] += 1
print(dict(c))
