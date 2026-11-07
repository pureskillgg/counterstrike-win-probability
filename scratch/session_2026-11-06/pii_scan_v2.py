import os
import re
import sys
import zipfile
import collections

ROOT = r"G:\pureskill\csds-cs2"
SUBS = ["logs", "notes", "deliverables"]
SKIP_DIRS = {".pixi", ".git", "node_modules", "__pycache__"}
BIN_EXT = {".parquet", ".zip", ".png", ".pyc", ".pptx", ".pkl", ".joblib", ".cbm", ".npy", ".xlsx", ".docx", ".pdf", ".svg"}

USER_RE = re.compile(rb"(?i)users[\\/]+13144")
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
IP_RE = re.compile(rb"(?<![\d.])((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])")
GENERIC_PROVIDERS = {b"gmail.com", b"outlook.com", b"hotmail.com", b"yahoo.com", b"icloud.com", b"proton.me", b"protonmail.com", b"live.com", b"msn.com"}
PLACEHOLDER = (b"example.", b"test.", b"localhost", b"domain.", b"email.", b"your")


def ip_class(s):
    o = [int(x) for x in s.split(b".")]
    a, b = o[0], o[1]
    if a == 127:
        return "loopback"
    if o == [0, 0, 0, 0]:
        return "unspecified"
    if a == 255 or a == 0:
        return "netmask_or_zero"
    if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
        return "private_rfc1918"
    if a == 169 and b == 254:
        return "link_local"
    if 224 <= a <= 239:
        return "multicast"
    if (a, b, o[2]) in ((192, 0, 2), (198, 51, 100), (203, 0, 113)):
        return "documentation"
    if max(o) <= 30:
        return "small_octets_version_like"
    return "other_public_shaped"


def email_class(m):
    dom = m.split(b"@", 1)[1].lower()
    if dom.startswith(PLACEHOLDER) or dom.endswith((b".example", b".test", b".local")):
        return "placeholder_or_test"
    if dom in GENERIC_PROVIDERS:
        return "consumer_provider"
    if dom.endswith((b".py", b".md", b".txt", b".json", b".csv", b".log", b".html", b".png", b".svg", b".pdf", b".docx", b".zip")):
        return "filename_like_token"
    return "other_domain_shape"


def scan_bytes(data):
    res = {}
    res["user"] = len(USER_RE.findall(data))
    res["email"] = [email_class(m) for m in EMAIL_RE.findall(data)]
    ips = [m.group(0) for m in IP_RE.finditer(data)]
    res["ip"] = [ip_class(s) for s in ips]
    # keys preceding other_public_shaped IPs (alphabetic key tokens only, never the value)
    keys = []
    for m in IP_RE.finditer(data):
        if ip_class(m.group(0)) == "other_public_shaped":
            left = data[max(0, m.start() - 24):m.start()]
            tok = re.findall(rb"[A-Za-z_]{2,}", left)
            keys.append(tok[-1].decode("ascii", "replace") if tok else "(no_key)")
    res["ip_keys"] = keys
    return res


def is_binary(data):
    return b"\x00" in data[:8192]


summary = collections.defaultdict(lambda: collections.Counter())
per_sub_files = collections.Counter()
per_file_hits = []

for sub in SUBS:
    base = os.path.join(ROOT, sub)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            ext = os.path.splitext(fn)[1].lower()
            per_sub_files[sub] += 1
            s = summary[sub]
            try:
                size = os.path.getsize(full)
                if ext == ".zip":
                    s["zip_files"] += 1
                    try:
                        with zipfile.ZipFile(full) as zf:
                            for info in zf.infolist():
                                if info.is_dir():
                                    continue
                                if info.file_size > 200 * 1024 * 1024:
                                    s["zip_member_too_big_skipped"] += 1
                                    continue
                                data = zf.read(info.filename)
                                mext = os.path.splitext(info.filename)[1].lower()
                                s["zip_members"] += 1
                                s["zip_member_ext_" + (mext or "none")] += 1
                                r = scan_bytes(data)
                                if r["user"]:
                                    s["zip_member_user_path"] += 1
                                if r["email"]:
                                    s["zip_member_email"] += 1
                                    s["zip_email_matches"] += len(r["email"])
                                    for c in r["email"]:
                                        s["zip_email_cls_" + c] += 1
                                if r["ip"]:
                                    s["zip_member_ip"] += 1
                                    for c in r["ip"]:
                                        s["zip_ip_cls_" + c] += 1
                                    for k in r["ip_keys"]:
                                        s["zip_ip_key_" + k] += 1
                    except Exception as e:
                        s["zip_unreadable"] += 1
                    continue
                with open(full, "rb") as fh:
                    data = fh.read()
            except Exception:
                s["unreadable"] += 1
                continue
            binary = ext in BIN_EXT or is_binary(data)
            kind = "binary" if binary else "text"
            s["files_" + kind] += 1
            r = scan_bytes(data)
            if r["user"]:
                s["user_path_files_" + kind] += 1
            if r["email"]:
                s["email_files_" + kind] += 1
                s["email_matches"] += len(r["email"])
                for c in r["email"]:
                    s["email_cls_" + c] += 1
                per_file_hits.append(("EMAIL", kind, ext or "(none)", len(r["email"]), rel))
            if r["ip"]:
                s["ip_files_" + kind] += 1
                s["ip_matches"] += len(r["ip"])
                for c in r["ip"]:
                    s["ip_cls_" + kind + "_" + c] += 1
                for k in r["ip_keys"]:
                    s["ip_key_" + kind + "_" + k] += 1
                per_file_hits.append(("IP", kind, ext or "(none)", len(r["ip"]), rel))

print("per_sub_files", dict(per_sub_files))
for sub in SUBS:
    print("==", sub)
    for k, v in sorted(summary[sub].items()):
        print("   ", k, v)
print("PER_FILE_HIT_LIST (type, kind, ext, count, relpath)")
for row in per_file_hits:
    print("  ", row)
