# -*- coding: utf-8 -*-
"""verify_E3_3_others.py

Re-derives the removal-scenario A (delete this env alone) numbers for the two
other candidates in the E3-3 claim, cs2gpu and cs2gpu_alt, with the same
FindFirstFileNameW enumeration used by verify_E3_3_recount.py. Read-only.
Appends its section to logs\\swarm2\\verify_E3_3.log.
"""
import collections
import os
import sys
import time
import traceback

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_E3_3_recount as rc  # noqa: E402  (module guard: main() not run on import)

WS = rc.WS
CACHE = rc.CACHE
CASES = [
    ("cs2gpu", WS + r"\cs2gpu-15282324412032665829\envs\default", "claim cs2gpu: 2.097 GiB apparent, 0.043 unique"),
    ("cs2gpu_alt", WS + r"\cs2gpu_alt-16300978804608365045\envs\default", "claim cs2gpu_alt: 1.294 GiB apparent, 0.034 unique"),
]


def classify(full, self_root):
    f = full.lower()
    sl = self_root.lower()
    if f == sl or f.startswith(sl + "\\"):
        return "self"
    if f.startswith(CACHE.lower() + "\\"):
        return "cache"
    if f.startswith(WS.lower() + "\\"):
        return "ws"
    return "oth"


def run(label, root, claim_text):
    t0 = time.time()
    recs, reparse, errs = rc.walk(root)
    seen = set()
    rows = []
    mism = 0
    for full, rel, size, nlink, dev, ino in recs:
        if (dev, ino) in seen:
            continue
        seen.add((dev, ino))
        names = rc.link_names(full)
        if len(names) != nlink:
            mism += 1
        alloc, ninfo = rc.std_info(full)
        cnt = collections.Counter(classify(n, root) for n in names)
        rows.append((size, alloc, nlink, cnt))
    appar = sum(r[2] for r in recs)
    a_rows = [r for r in rows if r[3]["cache"] == 0 and r[3]["ws"] == 0 and r[3]["oth"] == 0]
    b_rows = [r for r in rows if r[3]["ws"] == 0 and r[3]["oth"] == 0]
    rc.out("== %s (%s)" % (label, claim_text))
    rc.out("  files=%d inodes=%d reparse=%d walk_errors=%d apparent=%s"
           % (len(recs), len(rows), len(reparse), len(errs), rc.gib(appar)))
    rc.out("  enumerated-name count != st_nlink: %d" % mism)
    rc.out("  A delete this env alone : inodes=%d logical=%s on-disk=%s"
           % (len(a_rows), rc.gib(sum(r[0] for r in a_rows)), rc.gib(sum(r[1] for r in a_rows))))
    rc.out("  B cache copy removed    : inodes=%d logical=%s on-disk=%s"
           % (len(b_rows), rc.gib(sum(r[0] for r in b_rows)), rc.gib(sum(r[1] for r in b_rows))))
    rc.out("  inodes with a cache name: %d ; with other-ws names: %d ; elsewhere names: %d"
           % (sum(1 for r in rows if r[3]["cache"]), sum(1 for r in rows if r[3]["ws"]),
              sum(1 for r in rows if r[3]["oth"])))
    rc.out("  (%.1fs)" % (time.time() - t0))
    rc.out("")


def main():
    rc.out("")
    rc.out("== OTHER CANDIDATES, scenario A recount (FindFirstFileNameW names, on-disk = AllocationSize)",
           time.strftime("%Y-%m-%d %H:%M:%S"))
    for label, root, text in CASES:
        run(label, root, text)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        rc.out("EXCEPTION:")
        rc.out(traceback.format_exc())
    finally:
        with open(rc.LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(rc.LINES) + "\n")
