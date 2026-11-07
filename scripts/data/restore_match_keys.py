#!/usr/bin/env python3
"""Restore real match keys in files of the public repo.

Why this exists
---------------
The public repo holds pseudonymised match keys. Every match key

    csds/YYYY/MM/DD/<match uuid>/csds

appears as

    csds/YYYY/MM/DD/m_<24 hex>/csds

where the 24 hex characters are the first 24 characters of
sha256(lowercase match uuid). A bare match uuid (one that is in the
known-match set) appears as the bare token m_<24 hex>. The date stays visible.
The mapping cannot be reversed without the real keys, so the pseudonyms are
not secret on their own: only someone who already holds the real keys can
restore the original text.

To restore, give this script the real keys (a text or JSON file that contains
keys in the form csds/YYYY/MM/DD/<uuid>/csds, or a parquet file with a
`match_key` column). It builds the map pseudonym -> uuid from those keys and
rewrites every m_<24 hex> token it can resolve back to the lowercase uuid.
Pseudonyms that the given keys do not cover are left alone and counted.

Usage
-----
    py -3 restore_match_keys.py --keys <file> [--keys <file2> ...] <files or dirs>
    py -3 restore_match_keys.py --keys real_keys.txt --check-sha256 <prefix> <one file>

Files and directories are rewritten in place (.git is skipped). Office files
(.docx/.pptx/.xlsx, zip with word/ or ppt/ members) have their .xml and .rels
members rewritten; other binary data is left unchanged.

Limits
------
* Restoration writes lowercase uuids. A key that was uppercase in the
  original would come back lowercase.
* Only m_<24 hex> tokens are touched. Other text is never modified.
* Requires polars only for parquet keys files.
"""
import argparse
import hashlib
import io
import os
import re
import sys
import zipfile

UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
KEY_RE = re.compile(r"csds/\d{4}/\d{2}/\d{2}/(" + UUID + r")/csds")
PSEUDO_RE = re.compile(r"(?<!\w)m_([0-9a-f]{24})(?!\w)")


def anon(uuid):
    return hashlib.sha256(uuid.lower().encode()).hexdigest()[:24]


def read_keys(path):
    """Return the set of lowercase match uuids found in a keys file."""
    uuids = set()
    if path.lower().endswith(".parquet"):
        try:
            import polars as pl
        except ImportError:
            sys.exit("parquet keys need polars: py -3 -m pip install polars")
        values = pl.read_parquet(path, columns=["match_key"])["match_key"].to_list()
        for v in values:
            for m in KEY_RE.finditer(str(v)):
                uuids.add(m.group(1).lower())
        return uuids
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="surrogateescape")
    for m in KEY_RE.finditer(text):
        uuids.add(m.group(1).lower())
    return uuids


def transform_text(text, table, found, restored):
    def repl(m):
        hexpart = m.group(1)
        found.add(hexpart)
        uuid = table.get(hexpart)
        if uuid is None:
            return m.group(0)
        restored.add(hexpart)
        return uuid
    return PSEUDO_RE.sub(repl, text)


def transform_bytes(data, table, found, restored):
    def text_bytes(b):
        t = b.decode("utf-8", errors="surrogateescape")
        return transform_text(t, table, found, restored).encode("utf-8", errors="surrogateescape")

    if data.startswith(b"PK\x03\x04"):
        try:
            zin = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            zin = None
        if zin is not None:
            with zin:
                names = zin.namelist()
                if any(n.startswith("ppt/") or n.startswith("word/") for n in names):
                    out = io.BytesIO()
                    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                        for info in zin.infolist():
                            payload = zin.read(info.filename)
                            if info.filename.endswith((".xml", ".rels")):
                                payload = text_bytes(payload)
                            new = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                            new.external_attr = info.external_attr
                            new.compress_type = zipfile.ZIP_DEFLATED
                            zout.writestr(new, payload)
                    return out.getvalue()
    if b"\x00" in data[:8192]:
        return data
    return text_bytes(data)


def iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames if d != ".git"]
                for name in filenames:
                    yield os.path.join(dirpath, name)
        elif os.path.isfile(p):
            yield p
        else:
            sys.exit(f"not found: {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Restore real match keys in pseudonymised repo files.")
    ap.add_argument("--keys", action="append", required=True,
                    help="file of real keys (text/json or parquet with match_key); repeatable")
    ap.add_argument("--check-sha256", metavar="PREFIX",
                    help="after restoring, print whether the sha256 of the single given file starts with PREFIX")
    ap.add_argument("targets", nargs="+", help="files or directories to restore in place")
    args = ap.parse_args(argv)

    uuids = set()
    for k in args.keys:
        uuids |= read_keys(k)
    table = {anon(u): u for u in uuids}
    print(f"keys loaded: {len(uuids)} distinct match uuids")

    found, restored = set(), set()
    files_changed = files_seen = 0
    single_target = None
    for path in iter_files(args.targets):
        files_seen += 1
        single_target = path
        with open(path, "rb") as f:
            data = f.read()
        new = transform_bytes(data, table, found, restored)
        if new != data:
            with open(path, "wb") as f:
                f.write(new)
            files_changed += 1

    remaining = found - restored
    print(f"files scanned={files_seen} changed={files_changed}")
    print(f"distinct pseudonyms restored={len(restored)}")
    print(f"distinct pseudonyms remaining unrestored={len(remaining)}")

    if args.check_sha256:
        if files_seen != 1:
            sys.exit("--check-sha256 needs exactly one file target")
        with open(single_target, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        ok = digest.startswith(args.check_sha256.lower())
        print(f"sha256={digest}")
        print(f"starts with {args.check_sha256}: {'YES' if ok else 'NO'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
