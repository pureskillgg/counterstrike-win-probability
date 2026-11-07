"""E3 sampled unique-disk estimate for a directory tree.

Read-only. Walks the tree with os.scandir (sizes from DirEntry.stat, no extra call).
Every SAMPLE_EVERY-th file gets an os.stat call to read st_nlink. A file with
st_nlink == 1 is single-link, so its bytes are unique on the volume. A file with
st_nlink > 1 shares its data with another path (for example the pixi/rattler package
cache), so its bytes are not counted as unique here.

Estimator (Horvitz-Thompson style, systematic sample of every k-th file):
  est_total_bytes  = (sum of sampled sizes) * k        (should be near the exact total)
  est_unique_bytes = (sum of sampled sizes where nlink == 1) * k
The exact apparent total is also printed, from a full pass with no stat calls.

Usage:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -B -X utf8 E3_size_sample.py <root> [<sample_every>]
"""
import os
import sys
import time

root = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 50

t0 = time.time()
children = []
try:
    for c in os.scandir(root):
        if c.is_symlink():
            continue
        children.append(c)
except OSError as exc:
    print("ERROR cannot list %s: %s" % (root, exc))
    sys.exit(1)

grand = {"files": 0, "bytes": 0, "sampled": 0, "samp_bytes": 0, "samp_unique": 0, "errors": 0}
print("ROOT %s sample_every=%d children=%d" % (root, k, len(children)))
for c in sorted(children, key=lambda x: x.name):
    stats = {"files": 0, "bytes": 0, "sampled": 0, "samp_bytes": 0, "samp_unique": 0, "errors": 0}
    if c.is_dir(follow_symlinks=False):
        stack = [c.path]
    else:
        stack = []
        try:
            sz = c.stat(follow_symlinks=False).st_size
            stats["files"] = 1
            stats["bytes"] = sz
        except OSError:
            stats["errors"] += 1
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            stats["errors"] += 1
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
                    stats["errors"] += 1
                    continue
                stats["files"] += 1
                stats["bytes"] += sz
                if stats["files"] % k == 0:
                    stats["sampled"] += 1
                    stats["samp_bytes"] += sz
                    try:
                        nl = os.stat(e.path, follow_symlinks=False).st_nlink
                    except OSError:
                        stats["errors"] += 1
                        continue
                    if nl == 1:
                        stats["samp_unique"] += sz
    for key in grand:
        grand[key] += stats[key]
    est_total = stats["samp_bytes"] * k
    est_unique = stats["samp_unique"] * k
    print("CHILD %-58s exact_GiB=%8.3f est_GiB=%8.3f est_unique_GiB=%8.3f files=%-8d sampled=%-6d errors=%d"
          % (c.name, stats["bytes"] / 2**30, est_total / 2**30, est_unique / 2**30,
             stats["files"], stats["sampled"], stats["errors"]))
    sys.stdout.flush()

print("GRAND exact_GiB=%.3f est_GiB=%.3f est_unique_GiB=%.3f files=%d sampled=%d errors=%d secs=%.1f"
      % (grand["bytes"] / 2**30, grand["samp_bytes"] * k / 2**30, grand["samp_unique"] * k / 2**30,
         grand["files"], grand["sampled"], grand["errors"], time.time() - t0))
sys.stdout.flush()
