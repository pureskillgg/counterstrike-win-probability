# -*- coding: utf-8 -*-
"""verify_E3_3_sizediff.py

Explains the same-relpath/different-size files between the claimed C: env and
the G: project env (found by verify_E3_3_recount.py). Read-only. Appends its
section to logs\\swarm2\\verify_E3_3.log.

For each differing file: category by extension, whether the C: copy is
hard-linked (st_nlink>1, i.e. shared with the rattler cache), size delta, and,
for files up to 5 MB, whether the two contents are identical once each env's
own install prefix is replaced by one placeholder (prefix-rewrite hypothesis).
"""
import collections
import os
import stat
import time
import traceback

LOG = r"G:\pureskill\csds-cs2\logs\swarm2\verify_E3_3.log"
CLAIM = r"C:\Users\13144\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default"
G_ENV = r"G:\pureskill\csds-cs2\.pixi\envs\default"
REPARSE = 0x400
MAX_CMP = 5 * 1024 * 1024
LINES = []


def out(*parts):
    s = " ".join(str(p) for p in parts)
    LINES.append(s)
    print(s, flush=True)


def lp(p):
    p = os.path.normpath(p)
    if p.startswith("\\\\?\\"):
        return p
    return "\\\\?\\" + p


def walk(root):
    recs = {}
    stack = [root]
    while stack:
        d = stack.pop()
        with os.scandir(lp(d)) as it:
            entries = list(it)
        for ent in entries:
            full = os.path.join(d, ent.name)
            st = os.stat(lp(full), follow_symlinks=False)
            if getattr(st, "st_file_attributes", 0) & REPARSE:
                continue
            if stat.S_ISDIR(st.st_mode):
                stack.append(full)
            elif stat.S_ISREG(st.st_mode):
                rel = os.path.relpath(full, root)
                recs[rel] = (full, st.st_size, st.st_nlink)
    return recs


def variants(p):
    v = {p, p.replace("\\", "/"), p.replace("\\", "\\\\")}
    return sorted(v, key=len, reverse=True)


C_PREFIXES = variants(CLAIM)
G_PREFIXES = variants(G_ENV)


def normalise(data, prefixes, placeholder):
    for p in prefixes:
        data = data.replace(p.encode("utf-8"), placeholder)
    return data


def main():
    out("")
    out("== SIZE-DIFFERENCE ANALYSIS (claimed C: env vs G: env, same relpath)", time.strftime("%Y-%m-%d %H:%M:%S"))
    rc = walk(CLAIM)
    rg = walk(G_ENV)
    common = sorted(set(rc) & set(rg))
    diff = [r for r in common if rc[r][1] != rg[r][1]]
    out("same relpath: %d ; different size: %d" % (len(common), len(diff)))
    cat = collections.Counter()
    hl = collections.Counter()
    delta = collections.Counter()
    for r in diff:
        cat[os.path.splitext(r)[1].lower() or "(none)"] += 1
        hl["C hard-linked (cache)" if rc[r][2] > 1 else "C own copy (nlink 1)"] += 1
        delta[rg[r][1] - rc[r][1]] += 1
    out("by extension (top 12): %s" % cat.most_common(12))
    out("C copy link status: %s" % dict(hl))
    out("G minus C size delta (top 8): %s" % delta.most_common(8))
    out("prefix lengths: C=%d chars, G=%d chars, difference=%d"
        % (len(CLAIM), len(G_ENV), len(CLAIM) - len(G_ENV)))

    identical = 0
    differs = []
    too_big = []
    for r in diff:
        cf, cs, _ = rc[r]
        gf, gs, _ = rg[r]
        if max(cs, gs) > MAX_CMP:
            too_big.append(r)
            continue
        with open(lp(cf), "rb") as fh:
            cdata = fh.read()
        with open(lp(gf), "rb") as fh:
            gdata = fh.read()
        cn = normalise(cdata, C_PREFIXES, b"<INSTALL_PREFIX>")
        gn = normalise(gdata, G_PREFIXES, b"<INSTALL_PREFIX>")
        if cn == gn:
            identical += 1
        else:
            differs.append(r)
    out("files <= 5 MB compared after prefix normalisation: %d" % (len(diff) - len(too_big)))
    out("  identical after normalising each env's install prefix: %d" % identical)
    out("  NOT identical after normalisation: %d" % len(differs))
    out("  files > 5 MB not compared: %d" % len(too_big))
    for r in differs[:10]:
        out("    differs: %s (C %d B, G %d B, C nlink %d)" % (r, rc[r][1], rg[r][1], rc[r][2]))
    for r in too_big[:5]:
        out("    too big: %s (C %d B, G %d B)" % (r, rc[r][1], rg[r][1]))
    out("")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        out("EXCEPTION:")
        out(traceback.format_exc())
    finally:
        with open(LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(LINES) + "\n")
