"""E3 unique-disk estimate for a directory tree (stratified by file size).

Read-only. Walks the tree with os.scandir (exact apparent bytes from DirEntry.stat, no extra call).
st_nlink is read with os.stat:
  - every file of at least BIG bytes is checked exactly (large files dominate the bytes);
  - files smaller than BIG are checked on every SMALL_EVERY-th small file, and weighted by SMALL_EVERY.

A file with st_nlink == 1 is single-link, so its bytes are unique on the volume.
A file with st_nlink > 1 shares data with another path (the pixi/rattler cache), so it is not unique here.

Estimator:
  unique_est = sum(size for big files checked, if nlink == 1)
             + SMALL_EVERY * sum(size for sampled small files, if nlink == 1)

Usage:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -B -X utf8 E3_size_sample2.py <root> [<small_every>] [<big_bytes>]
"""
import os
import sys
import time

root = sys.argv[1]
SMALL_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 20
BIG = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 20  # 1 MiB

t0 = time.time()
try:
    children = [c for c in os.scandir(root) if not c.is_symlink()]
except OSError as exc:
    print("ERROR cannot list %s: %s" % (root, exc))
    sys.exit(1)

keys = ("files", "bytes", "big_checked", "big_bytes", "big_unique", "small_sampled", "small_unique", "errors")
grand = {k: 0 for k in keys}
print("ROOT %s small_every=%d big_bytes=%d children=%d" % (root, SMALL_EVERY, BIG, len(children)))
for c in sorted(children, key=lambda x: x.name):
    s = {k: 0 for k in keys}
    small_counter = 0
    if c.is_dir(follow_symlinks=False):
        stack = [c.path]
    else:
        stack = []
        try:
            s["files"] = 1
            s["bytes"] = c.stat(follow_symlinks=False).st_size
        except OSError:
            s["errors"] += 1
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            s["errors"] += 1
            continue
        with it:
            for e in it:
                try:
                    if e.is_symlink():
                        continue
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                        continue
                    sz = e.stat(follow_symlinks=False).st_size
                except OSError:
                    s["errors"] += 1
                    continue
                s["files"] += 1
                s["bytes"] += sz
                if sz >= BIG:
                    s["big_checked"] += 1
                    s["big_bytes"] += sz
                    try:
                        nl = os.stat(e.path, follow_symlinks=False).st_nlink
                    except OSError:
                        s["errors"] += 1
                        continue
                    if nl == 1:
                        s["big_unique"] += sz
                else:
                    small_counter += 1
                    if small_counter % SMALL_EVERY == 0:
                        s["small_sampled"] += 1
                        try:
                            nl = os.stat(e.path, follow_symlinks=False).st_nlink
                        except OSError:
                            s["errors"] += 1
                            continue
                        if nl == 1:
                            s["small_unique"] += sz
    unique_est = s["big_unique"] + SMALL_EVERY * s["small_unique"]
    for k in keys:
        grand[k] += s[k]
    print("CHILD %-58s exact_GiB=%8.3f unique_est_GiB=%8.3f files=%-8d big_checked=%-6d small_sampled=%-6d errors=%d"
          % (c.name, s["bytes"] / 2**30, unique_est / 2**30, s["files"], s["big_checked"], s["small_sampled"], s["errors"]))
    sys.stdout.flush()

grand_unique = grand["big_unique"] + SMALL_EVERY * grand["small_unique"]
print("GRAND exact_GiB=%.3f unique_est_GiB=%.3f unique_fraction=%.4f files=%d big_checked=%d small_sampled=%d errors=%d secs=%.1f"
      % (grand["bytes"] / 2**30, grand_unique / 2**30,
         (grand_unique / grand["bytes"]) if grand["bytes"] else 0.0,
         grand["files"], grand["big_checked"], grand["small_sampled"], grand["errors"], time.time() - t0))
sys.stdout.flush()
