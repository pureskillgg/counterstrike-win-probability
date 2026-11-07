import os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = r"G:\pureskill\csds-cs2"
USERNAME_RE = re.compile(rb"(?i)(?:[A-Za-z]:[\/]+Users[\/]+|/home/|/Users/)([^\/\s\"'<>:|*?]{1,40})")
GENERIC = {b"root", b"user", b"default", b"public", b"ubuntu", b"runner", b"jovyan", b"admin", b"guest", b"system", b"administrator", b"ec2-user", b"sagemaker-user", b"node", b"vscode", b"all users", b"default user", b"localappdata", b"appdata"}
cls = collections.Counter()
for sub in ["logs", "notes", "deliverables"]:
    for dp, dn, fn in os.walk(os.path.join(ROOT, sub)):
        for f in fn:
            ext = os.path.splitext(f)[1].lower()
            if ext in (".parquet", ".png", ".zip", ".pptx", ".docx", ".pdf", ".npy", ".npz", ".pkl", ".pyc"): continue
            try: data = open(os.path.join(dp, f), "rb").read()
            except Exception: continue
            for m in USERNAME_RE.finditer(data):
                n = m.group(1).lower()
                if n == b"13144": continue
                if n in GENERIC: cls["generic_system_or_tool_name"] += 1
                elif re.fullmatch(rb"\d{4}-\d{2}(-\d{2})?", n): cls["date_like_segment"] += 1
                else: cls["unclassified_segment"] += 1
print("OTHER_SEGMENT_CLASSES", dict(cls))
