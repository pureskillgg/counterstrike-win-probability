import os
import re
import sys
import zipfile
import collections

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = r"G:\pureskill\csds-cs2"
SUBS = ["logs", "notes", "deliverables"]
SKIP_DIRS = {".pixi", ".git", "node_modules", "__pycache__"}

USER_RE = re.compile(rb"(?i)users[\\/]+13144")
USERNAME_RE = re.compile(rb"(?i)(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)([^\\/\s\"'<>:|*?]{1,40})")
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
IP_RE = re.compile(rb"(?<![\d.])((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])")
COMMON_TLD = {b"com", b"org", b"net", b"edu", b"io", b"ai", b"uk", b"de", b"fr", b"gov", b"ac", b"int", b"info", b"dev", b"eu", b"ca", b"au", b"nl"}
SYSTEM_NAMES = {b"13144", b"sagemaker-user", b"public", b"default", b"root", b"ubuntu", b"all users", b"default user"}

VERSION_HINTS = (b"cu1", b"cu2", b"libprotobuf", b"nvidia", b"==", b"version", b"torch", b"so.", b".so", b"lib", b"pip", b"wheel", b"sha", b"v1.", b"v2.", b"v0.")
INFRA_HINTS = (b"ssh", b"root@", b"pod", b"runpod", b"host", b"server", b"endpoint", b"s3", b"ec2", b"sagemaker", b"publicip", b"public_ip", b"public ip", b"ip ", b"ip:", b"ip=", b"address")
PERSONAL_HINTS = (b"home", b"isp", b"residential", b"laptop", b"my ip", b"router", b"wifi")


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
    if max(o) <= 30:
        return "small_octets_version_like"
    return "other_public_shaped"


def context_class(line_lc, ip_bytes):
    if any(h in line_lc for h in VERSION_HINTS):
        return "version_or_library_context"
    if any(h in line_lc for h in PERSONAL_HINTS):
        return "personal_hint_context"
    if any(h in line_lc for h in INFRA_HINTS):
        return "infra_hint_context"
    return "no_hint_context"


def is_binary(data):
    return b"\x00" in data[:8192]


def magic(data):
    if data[:4] == b"PAR1":
        return "parquet_magic"
    if data[:2] == b"PK":
        return "zip_magic"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png_magic"
    if data[:2] == b"\x1f\x8b":
        return "gzip_magic"
    if data[:6] == b"\x93NUMPY":
        return "npy_magic"
    return "other"


email_by_kind = collections.Counter()
email_tld = collections.Counter()
email_magic = collections.Counter()
email_list = []
ip_ctx = collections.Counter()
ip_other_list = []
user_names = collections.Counter()
user_path_files = collections.Counter()
user_path_hits = []


def scan_obj(rel, data, kind, ext_label):
    # email
    ems = EMAIL_RE.findall(data)
    if ems:
        email_by_kind[(kind, ext_label)] += len(ems)
        email_magic[(kind, magic(data))] += len(ems)
        for m in ems:
            tld = m.rsplit(b".", 1)[1].lower()
            email_tld["common_tld" if tld in COMMON_TLD else "uncommon_tld_or_binary_noise"] += 1
        email_list.append((rel, kind, ext_label, len(ems)))
    # ip
    if kind == "text":
        text = data.decode("utf-8", "replace")
        for line in text.splitlines():
            lb = line.encode("utf-8", "replace")
            for m in IP_RE.finditer(lb):
                ipb = m.group(0)
                cls = ip_class(ipb)
                ctx = context_class(lb.lower(), ipb)
                ip_ctx[(cls, ctx)] += 1
                if cls == "other_public_shaped":
                    ip_other_list.append((rel, ext_label, ctx))
    else:
        for m in IP_RE.finditer(data):
            cls = ip_class(m.group(0))
            ip_ctx[(cls, "binary_no_context")] += 1
    # user-profile paths
    if USER_RE.search(data):
        user_path_files[kind] += 1
    for m in USERNAME_RE.finditer(data):
        name = m.group(1).lower()
        if name in SYSTEM_NAMES:
            user_names["system_or_own_" + name.decode("ascii", "replace")] += 1
        else:
            user_names["other_username_hits"] += 1


for sub in SUBS:
    base = os.path.join(ROOT, sub)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = sub + "/" + os.path.relpath(full, base).replace("\\", "/")
            ext = os.path.splitext(fn)[1].lower() or "(none)"
            if ext == ".zip":
                try:
                    with zipfile.ZipFile(full) as zf:
                        for info in zf.infolist():
                            if info.is_dir() or info.file_size > 200 * 1024 * 1024:
                                continue
                            data = zf.read(info.filename)
                            mext = os.path.splitext(info.filename)[1].lower() or "(none)"
                            kind = "binary" if is_binary(data) else "text"
                            scan_obj(rel + "::member", data, kind, "zipmember" + mext)
                except Exception as e:
                    print("zip_error", rel, type(e).__name__)
                continue
            try:
                with open(full, "rb") as fh:
                    data = fh.read()
            except Exception:
                continue
            kind = "binary" if (ext in {".parquet", ".png", ".pptx", ".pdf", ".docx", ".pyc", ".pkl", ".npy", ".npz", ".gz"} or is_binary(data)) else "text"
            scan_obj(rel, data, kind, ext)

print("== EMAIL matches by (kind, ext):", dict(email_by_kind))
print("== EMAIL matches by magic:", dict(email_magic))
print("== EMAIL TLD class:", dict(email_tld))
print("== EMAIL-hit containers (rel, kind, ext, n):")
for row in email_list:
    print("   ", row)
print("== IPv4 by (class, context):")
for k, v in sorted(ip_ctx.items()):
    print("   ", k, v)
print("== IPv4 other_public_shaped hits (rel, ext, context):")
for row in ip_other_list:
    print("   ", row)
print("== user-profile files by kind:", dict(user_path_files))
print("== username tokens in path strings:", dict(user_names))
