"""verify_E3_7_envdiff.py -- classify file-level differences between the two env copies.

Read-only. Stdlib only. Run with: python.exe -I -X utf8 -B verify_E3_7_envdiff.py
Purpose: the G: copy and the C: copy must have the same content before any load-time
A/B comparison means anything. This script groups size mismatches and one-sided files
by category (pycache, extension, top-level package).
"""
import collections
import os
import sys

ENV_ON_G = r"G:\pureskill\csds-cs2\.pixi\envs\default"
ENV_ON_C = r"C:\Users\<user>\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default"
FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def walk(root):
    out = {}
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                children = list(it)
        except OSError:
            continue
        for e in children:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                    continue
                st = os.stat(e.path, follow_symlinks=False)
                out[os.path.relpath(e.path, root).replace("\\", "/")] = (st.st_size, st.st_nlink, st.st_mtime)
            except OSError:
                pass
    return out


def category(rel):
    parts = rel.split("/")
    top = parts[0]
    if "__pycache__" in parts:
        kind = "__pycache__"
    else:
        kind = os.path.splitext(rel)[1].lower() or "(none)"
    if top in ("Lib", "Library", "Scripts", "DLLs", "Tools"):
        sub = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:2])
    else:
        sub = top
    return kind, sub


def main():
    eg = walk(ENV_ON_G)
    ec = walk(ENV_ON_C)
    print(f"G files={len(eg)} C files={len(ec)}")
    common = set(eg) & set(ec)
    mism = [k for k in common if eg[k][0] != ec[k][0]]
    only_g = sorted(set(eg) - set(ec))
    only_c = sorted(set(ec) - set(eg))
    print(f"common={len(common)} size_mismatch={len(mism)} only_G={len(only_g)} only_C={len(only_c)}")

    print("-- size mismatches by kind")
    kinds = collections.Counter(category(k)[0] for k in mism)
    for k, n in kinds.most_common(15):
        print(f"   {k:14s} {n}")
    print("-- size mismatches by top subtree")
    subs = collections.Counter(category(k)[1] for k in mism)
    for k, n in subs.most_common(15):
        print(f"   {k:45s} {n}")
    print("-- sample size mismatches (rel, G size, C size, G mtime, C mtime)")
    for k in sorted(mism)[:8]:
        print(f"   {k}  G={eg[k][0]}  C={ec[k][0]}  Gmt={eg[k][2]:.0f}  Cmt={ec[k][2]:.0f}")

    print("-- only_on_G by kind")
    kinds_g = collections.Counter(category(k)[0] for k in only_g)
    for k, n in kinds_g.most_common(10):
        print(f"   {k:14s} {n}")
    print("-- only_on_G sample")
    for k in only_g[:8]:
        print(f"   {k}")
    print("-- only_on_C")
    for k in only_c[:5]:
        print(f"   {k}")

    pyc_g = sum(1 for k in eg if k.endswith(".pyc"))
    pyc_c = sum(1 for k in ec if k.endswith(".pyc"))
    print(f"-- .pyc files: G={pyc_g} C={pyc_c}")
    # mismatches that are not pycache: are any of them binaries the import path loads?
    non_pyc = [k for k in mism if not k.endswith(".pyc")]
    print(f"-- size mismatches excluding .pyc: {len(non_pyc)}")
    for k in sorted(non_pyc)[:12]:
        print(f"   {k}  G={eg[k][0]}  C={ec[k][0]}")


if __name__ == "__main__":
    sys.exit(main())
