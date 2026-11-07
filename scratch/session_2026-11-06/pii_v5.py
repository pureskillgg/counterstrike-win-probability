import os
import re
import sys
import collections
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = r"G:\pureskill\csds-cs2"
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
USERNAME_RE = re.compile(rb"(?i)(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)([^\\/\s\"'<>:|*?]{1,40})")
IDLIKE = re.compile(r"(?i)(player|steam|user|name|email|phone|addr|ip|account|uid|person|nick|handle|ident|sid)")

# 1. Where do the email-shaped byte matches sit in the 5 parquet files: footer or data region?
parq = [
    r"logs\swarm\E1\cache\item_pickup.parquet",
    r"logs\swarm\E1\cache\round_state.parquet",
    r"logs\swarm2\verify_D7_8_round_state_slim.parquet",
    r"notes\provenance_2026-11-06\logloss_check\gru_seed0_final_val_preds.parquet",
    r"notes\provenance_2026-11-06\logloss_check\gru_seed0_val_preds.parquet",
]
for rel in parq:
    p = os.path.join(ROOT, rel)
    data = open(p, "rb").read()
    flen = int.from_bytes(data[-8:-4], "little")
    footer_start = len(data) - 8 - flen
    where = collections.Counter()
    for m in EMAIL_RE.finditer(data):
        where["footer" if m.start() >= footer_start else "data_region"] += 1
    print("EMAIL_LOCATION", rel, "file_bytes", len(data), dict(where))

# 2. Identifier-like column names across all local parquet files (schema only, footer read)
print("== identifier-like parquet columns (schema only)")
hits = []
total = 0
for sub in ["logs", "notes", "deliverables", "evac_2026-10-13", "data"]:
    base = os.path.join(ROOT, sub)
    if not os.path.isdir(base):
        continue
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d not in (".pixi", ".git")]
        for f in fn:
            if not f.lower().endswith(".parquet"):
                continue
            total += 1
            p = os.path.join(dp, f)
            try:
                sch = pq.ParquetFile(p).schema_arrow
                names = sch.names
                idc = [n for n in names if IDLIKE.search(n)]
                if idc:
                    rel = os.path.relpath(p, ROOT)
                    hits.append((rel, idc))
            except Exception as e:
                print("  schema_err", os.path.relpath(p, ROOT), type(e).__name__)
print("parquet_files_checked", total, "files_with_id_like_columns", len(hits))
for rel, idc in hits:
    print("  ", rel, idc)

# 3. Non-user usernames in path strings: classify, print categories only
cats = collections.Counter()
SYSTEM = {b"13144", b"public", b"default", b"default user", b"all users", b"sagemaker-user", b"root", b"runner", b"ubuntu", b"jovyan", b"node", b"vscode", b"user"}
for sub in ["logs", "notes", "deliverables"]:
    base = os.path.join(ROOT, sub)
    for dp, dn, fn in os.walk(base):
        for f in fn:
            p = os.path.join(dp, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in (".parquet", ".png", ".zip", ".pptx", ".docx", ".pdf", ".npy", ".npz", ".pkl", ".pyc"):
                continue
            try:
                data = open(p, "rb").read()
            except Exception:
                continue
            for m in USERNAME_RE.finditer(data):
                name = m.group(1).lower()
                if name == b"13144":
                    cats["own_account_13144"] += 1
                elif name in SYSTEM:
                    cats["system_generic"] += 1
                elif name.startswith((b"%", b"<", b"{", b"$")) or b"%" in name or b"{" in name:
                    cats["placeholder_template"] += 1
                else:
                    cats["other_named_user_segment"] += 1
print("== username segments in path strings (text files, count only):", dict(cats))
