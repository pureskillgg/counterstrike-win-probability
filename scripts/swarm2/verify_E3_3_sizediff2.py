# -*- coding: utf-8 -*-
"""verify_E3_3_sizediff2.py

Breaks down the non-.pyc size-different files between the claimed C: env and the
G: env after install-prefix normalisation. Read-only; appends to the log.
"""
import collections
import os
import sys
import time
import traceback

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "verify_E3_3_sizediff", r"G:\pureskill\csds-cs2\scripts\swarm2\verify_E3_3_sizediff.py")
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

LINES = []


def out(*parts):
    s = " ".join(str(p) for p in parts)
    LINES.append(s)
    print(s, flush=True)


def main():
    out("")
    out("== NON-PYC BREAKDOWN after prefix normalisation", time.strftime("%Y-%m-%d %H:%M:%S"))
    rc = sd.walk(sd.CLAIM)
    rg = sd.walk(sd.G_ENV)
    common = sorted(set(rc) & set(rg))
    diff = [r for r in common if rc[r][1] != rg[r][1]]
    nonpyc = [r for r in diff if not r.lower().endswith(".pyc")]
    ident = collections.Counter()
    bad = []
    for r in nonpyc:
        cf, cs, cn_ = rc[r]
        gf, gs, _ = rg[r]
        with open(sd.lp(cf), "rb") as fh:
            cd = fh.read()
        with open(sd.lp(gf), "rb") as fh:
            gd = fh.read()
        cn = sd.normalise(cd, sd.C_PREFIXES, b"<P>")
        gn = sd.normalise(gd, sd.G_PREFIXES, b"<P>")
        ext = os.path.splitext(r)[1].lower()
        if cn == gn:
            ident[(ext, "identical")] += 1
        else:
            ident[(ext, "DIFFERENT")] += 1
            bad.append((r, cn, gn, cs, gs))
    out("non-pyc size-different files: %d" % len(nonpyc))
    for k, v in sorted(ident.items()):
        out("  %-8s %-10s %d" % (k[0] or "(none)", k[1], v))
    out("non-pyc, non-identical after normalisation: %d" % len(bad))
    for r, cn, gn, cs, gs in bad[:12]:
        # first differing offset and short context from each side
        i = 0
        m = min(len(cn), len(gn))
        while i < m and cn[i] == gn[i]:
            i += 1
        out("  %s  C=%dB G=%dB normalised C=%dB G=%dB firstdiff@%d" % (r, cs, gs, len(cn), len(gn), i))
        out("     C: %r" % cn[max(0, i - 40):i + 60])
        out("     G: %r" % gn[max(0, i - 40):i + 60])
    out("")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        out("EXCEPTION:")
        out(traceback.format_exc())
    finally:
        with open(sd.LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(LINES) + "\n")
