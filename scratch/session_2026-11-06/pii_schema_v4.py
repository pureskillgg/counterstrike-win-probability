import os
import re
import sys
import zipfile
import collections

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = r"G:\pureskill\csds-cs2"
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
COMMON_TLD = {b"com", b"org", b"net", b"edu", b"io", b"ai", b"uk", b"de", b"fr", b"gov", b"ac", b"int", b"info", b"dev", b"eu", b"ca", b"au", b"nl"}
IP_RE = re.compile(rb"(?<![\d.])((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])")

# 1. Parquet footers only: schema and row counts, no value reads
import pyarrow.parquet as pq
parq = [
    r"logs\swarm\E1\cache\item_pickup.parquet",
    r"logs\swarm\E1\cache\round_state.parquet",
    r"logs\swarm2\verify_D7_8_round_state_slim.parquet",
    r"notes\provenance_2026-11-06\logloss_check\gru_seed0_final_val_preds.parquet",
    r"notes\provenance_2026-11-06\logloss_check\gru_seed0_val_preds.parquet",
]
for rel in parq:
    p = os.path.join(ROOT, rel)
    try:
        md = pq.ParquetFile(p).metadata
        sch = pq.ParquetFile(p).schema_arrow
        cols = []
        for i in range(md.num_columns):
            c = md.row_group(0).column(i) if md.num_row_groups else None
            cols.append((sch.names[i], str(sch.types[i]), c.compression if c else None))
        string_cols = [c for c in cols if "string" in c[1] or "binary" in c[1]]
        print("PARQUET", rel, "rows", md.num_rows, "cols", md.num_columns, "string_or_binary_cols", len(string_cols))
        for name, typ, comp in cols:
            print("     col", name, typ, comp)
    except Exception as e:
        print("PARQUET_ERR", rel, type(e).__name__, str(e)[:120])

# 2. Key context for emails in text toml files (report key name and TLD class only)
def toml_context(label, data):
    lines = data.decode("utf-8", "replace").splitlines()
    section = ""
    for ln in lines:
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s
        for m in EMAIL_RE.finditer(ln.encode("utf-8", "replace")):
            tld = m.group(0).rsplit(b".", 1)[1].lower()
            key = ln.split("=", 1)[0].strip() if "=" in ln else "(no_key_on_line)"
            print("TOML_EMAIL", label, "section=", section, "key=", key[:40], "tld_common=", tld in COMMON_TLD)

toml_path = os.path.join(ROOT, r"counterstrike-win-probability\pixi.toml")
if os.path.exists(toml_path):
    toml_context("local_pixi_toml_(other_tree)", open(toml_path, "rb").read())
# zip member
zp = os.path.join(ROOT, r"deliverables\checkpoints\CHECKPOINT_2026-10-20_0017.zip")
with zipfile.ZipFile(zp) as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        if os.path.splitext(info.filename)[1].lower() == ".toml":
            data = zf.read(info.filename)
            if EMAIL_RE.search(data):
                print("ZIP_TOML_MEMBER_NAME", info.filename, "size", info.file_size)
                toml_context("zip_member", data)

# 3. IP key context for other-public-shaped hits in JSON/MD/log text (key token only)
def ip_key_context(label, text):
    out = collections.Counter()
    for ln in text.splitlines():
        for m in IP_RE.finditer(ln.encode("utf-8", "replace")):
            a, b = [int(x) for x in m.group(0).split(b".")][:2]
            o = [int(x) for x in m.group(0).split(b".")]
            if a in (127, 0, 255, 10) or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168) or max(o) <= 30:
                continue
            left = ln.encode("utf-8", "replace")[:m.start()]
            toks = re.findall(rb"[A-Za-z_][A-Za-z_]{1,}", left[-40:])
            out[(label, toks[-1].decode() if toks else "(no_token)")] += 1
    return out

agg = collections.Counter()
for root_sub in ["logs", "notes", "deliverables"]:
    for dp, dn, fn in os.walk(os.path.join(ROOT, root_sub)):
        for f in fn:
            ext = os.path.splitext(f)[1].lower()
            if ext not in (".json", ".md", ".txt", ".log", ".yaml", ".csv", ".py", ".html"):
                continue
            try:
                txt = open(os.path.join(dp, f), "rb").read().decode("utf-8", "replace")
            except Exception:
                continue
            agg += ip_key_context(ext, txt)
print("IP_OTHER_PUBLIC_KEY_TOKENS (ext, preceding token):")
for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    print("   ", k, v)
