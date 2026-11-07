import os
import re
import collections

ROOT = r"C:\Users\<user>\Documents\pureskill"
SKIP_DIRS = {"envs", ".git", "node_modules", "__pycache__", "catboost_info", "AutogluonModels"}
BIN_EXT = {".parquet", ".zip", ".png", ".pyc", ".pptx", ".pkl", ".joblib", ".cbm", ".npy", ".xlsx", ".docx", ".stackdump"}

USER_RE = re.compile(rb"(?i)users[\\/]+<user>")
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
IP_RE = re.compile(rb"(?<![\d.])((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])")
PLACEHOLDER_DOMAINS = {"example.com", "example.org", "example.net", "test.com", "localhost", "domain.com", "email.com"}


def ip_class(s):
    o = [int(x) for x in s.split(b".")] if isinstance(s, bytes) else [int(x) for x in s.split(".")]
    a, b = o[0], o[1]
    if a == 127:
        return "loopback"
    if o == [0, 0, 0, 0]:
        return "unspecified"
    if a == 255 or (a == 0):
        return "netmask_or_zero"
    if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
        return "private_rfc1918"
    if a == 169 and b == 254:
        return "link_local"
    if 224 <= a <= 239:
        return "multicast"
    if (a, b) == (192, 0) and o[2] == 2 or (a, b, o[2]) == (198, 51, 100) or (a, b, o[2]) == (203, 0, 113):
        return "documentation"
    if max(o) <= 30:
        return "small_octets_version_like"
    return "other_public_shaped"


def is_binary(data):
    return b"\x00" in data[:8192]


counts = collections.Counter()
user_files = collections.Counter()
email_files = collections.Counter()
email_domain_cls = collections.Counter()
ip_files = collections.Counter()
ip_cls = collections.Counter()
files_seen = 0
ext_seen = collections.Counter()
email_hits_detail = []
ip_hits_detail = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, ROOT)
        ext = os.path.splitext(fn)[1].lower()
        files_seen += 1
        ext_seen[ext or "(none)"] += 1
        try:
            with open(full, "rb") as fh:
                data = fh.read()
        except Exception as e:
            counts["unreadable"] += 1
            continue
        binary = ext in BIN_EXT or is_binary(data)
        kind = "binary" if binary else "text"
        if USER_RE.search(data):
            user_files[(kind, ext or "(none)")] += 1
        em = EMAIL_RE.findall(data)
        if em:
            email_files[(kind, ext or "(none)")] += 1
            counts["email_matches_total"] += len(em)
            for m in em:
                dom = m.split(b"@", 1)[1].lower().decode("ascii", "replace")
                if dom in PLACEHOLDER_DOMAINS or dom.endswith(".example") or dom.endswith(".test"):
                    email_domain_cls["placeholder_or_test_domain"] += 1
                elif dom.endswith((".py", ".md", ".txt", ".json", ".csv", ".log")):
                    email_domain_cls["filename_like_token"] += 1
                else:
                    email_domain_cls["other_domain_shape"] += 1
            email_hits_detail.append((rel, kind, len(em)))
        ips = IP_RE.findall(data)
        if ips:
            ip_files[(kind, ext or "(none)")] += 1
            full_ips = [m.group(0) for m in IP_RE.finditer(data)]
            counts["ip_matches_total"] += len(full_ips)
            for s in full_ips:
                ip_cls[(kind, ip_class(s))] += 1
            ip_hits_detail.append((rel, kind, len(full_ips)))

print("files_seen", files_seen)
print("ext_top", ext_seen.most_common(30))
print("user_profile_string_files", sum(user_files.values()), dict(user_files))
print("email_shaped_files", sum(email_files.values()), dict(email_files))
print("email_matches_total", counts["email_matches_total"], dict(email_domain_cls))
print("ipv4_shaped_files", sum(ip_files.values()), dict(ip_files))
print("ipv4_matches_total", counts["ip_matches_total"])
print("ipv4_by_class", dict(ip_cls))
print("EMAIL_FILES_LIST")
for rel, kind, n in email_hits_detail:
    print("  ", kind, n, rel)
print("IP_FILES_LIST")
for rel, kind, n in ip_hits_detail:
    print("  ", kind, n, rel)
