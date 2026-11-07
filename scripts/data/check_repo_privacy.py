#!/usr/bin/env python3
"""Scan a folder (default: the repo root) for identifiers that must not be committed.

    py -3 scripts/data/check_repo_privacy.py [folder]

It looks at every file, skipping .git, and reports:
  * raw match keys: csds/YYYY/MM/DD/<uuid>/csds. Pseudonymised keys (csds/YYYY/MM/DD/m_<24 hex>/csds) pass.
  * email addresses (git@github.com style hosts and the noreply addresses are allowed)
  * Windows or Linux user-name paths: Users\\<name>\\ with any name except the <user> placeholder, and /home/<name>/ except `sagemaker-user`, `root` and the <user> placeholder
  * IPv4 addresses that are not private, loopback or version-like (a pod address, for example)
  * AWS account numbers inside ARNs
Office files (.docx, .pptx, .xlsx) are checked member by member. Other binary files are skipped.

Exit code 0 means nothing was found; 1 means at least one finding. Findings print as `rule  file:count`; the matched text is never printed, so the report is safe to paste.
"""
import io
import os
import re
import sys
import zipfile
from collections import Counter

UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
RULES = {
    "raw-match-key": re.compile(r"csds/\d{4}/\d{2}/\d{2}/" + UUID + r"/csds"),
    "email": re.compile(
        r"(?<![\w.+-])[\w.+-]+@(?!users\.noreply\.github\.com|anthropic\.com|github\.com)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:edu|com|org|net|io|gg|dev|co)\b"
    ),
    "windows-user-path": re.compile(
        r"(?i)Users[\\/]+(?!<user>|Public\b|Default\b|All Users\b|username\b|you\b|your|name\b|\*|%|\{|\$)[A-Za-z0-9_.-]{2,32}[\\/]"
        r"|C--Users-(?!<user>-)[A-Za-z0-9_.]{2,32}-"
    ),
    "linux-home-path": re.compile(r"/home/(?!<user>|sagemaker-user\b|root\b|user\b|username\b|you\b|\$|\{)[A-Za-z0-9_.-]{2,32}/"),
    "public-ipv4": re.compile(
        r"(?<![\d.])(?!127\.|0\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|255\.|169\.254\.|8\.8\.|1\.1\.1\.)"
        r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.]*\d)(?!\.\d)"
    ),
    "aws-account-in-arn": re.compile(r"arn:aws:[a-z0-9-]+:[a-z0-9-]*:(?!<account-id>)\d{12}(?!\d)"),
}
# a version string such as 12.1.3.1 has four numbers like an address; require the first number to be 11 or more
# and ignore the well-known version-like prefixes that appeared in this repo's CUDA and driver notes
VERSION_LIKE = re.compile(r"^(?:[0-9]|1[0-2])\.\d{1,2}\.\d{1,3}\.\d{1,3}$")


def text_findings(text, counter):
    for name, rx in RULES.items():
        for m in rx.finditer(text):
            if name == "public-ipv4" and VERSION_LIKE.match(m.group(0)):
                continue
            counter[name] += 1


def scan_bytes(data, counter):
    if data.startswith(b"PK\x03\x04"):
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            return
        with z:
            names = z.namelist()
            if not any(n.startswith(("ppt/", "word/", "xl/")) for n in names):
                return
            for n in names:
                if n.endswith((".xml", ".rels")):
                    text_findings(z.read(n).decode("utf-8", "replace"), counter)
        return
    if b"\x00" in data[:8192]:
        return
    text_findings(data.decode("utf-8", "replace"), counter)


def main(argv):
    root = argv[1] if len(argv) > 1 else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    total = Counter()
    per_file = {}
    self_path = os.path.abspath(__file__)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.abspath(path) == self_path:  # its own patterns would match themselves
                continue
            c = Counter()
            try:
                with open(path, "rb") as f:
                    scan_bytes(f.read(), c)
            except OSError:
                continue
            if c:
                per_file[os.path.relpath(path, root).replace("\\", "/")] = c
                total.update(c)
    if not total:
        print("clean: no raw match keys, emails, user-name paths, public IPs or AWS account numbers found")
        return 0
    print("FINDINGS (matched text is not printed):")
    for rule, n in total.most_common():
        print(f"  {rule}: {n} in {sum(1 for c in per_file.values() if rule in c)} files")
    for rel, c in sorted(per_file.items())[:40]:
        print(f"    {rel}: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
