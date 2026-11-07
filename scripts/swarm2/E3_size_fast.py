"""E3 fast apparent-size walker, one line per immediate child of each root.

Read-only. Uses DirEntry.stat(), which on Windows comes from the directory enumeration
and needs no extra system call per file, so it is much faster than os.stat per file.
Reports apparent (not unique) bytes. Hard-link dedup is measured separately by E3_size_probe.py.

Usage:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 E3_size_fast.py <root> [<root> ...]
"""
import os
import sys
import time


def tree_size(path):
    total = 0
    files = 0
    errors = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            errors += 1
            continue
        with it:
            for e in it:
                try:
                    if e.is_symlink():
                        continue
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    else:
                        total += e.stat(follow_symlinks=False).st_size
                        files += 1
                except OSError:
                    errors += 1
    return total, files, errors


if __name__ == "__main__":
    grand_total = 0
    for root in sys.argv[1:]:
        t0 = time.time()
        print("ROOT %s" % root)
        rows = []
        try:
            children = list(os.scandir(root))
        except OSError as exc:
            print("  ERROR cannot list: %s" % exc)
            continue
        root_total = 0
        for c in children:
            if c.is_symlink():
                continue
            if c.is_dir(follow_symlinks=False):
                size, nfiles, err = tree_size(c.path)
            else:
                size, nfiles, err = c.stat(follow_symlinks=False).st_size, 1, 0
            root_total += size
            rows.append((size, nfiles, err, c.name))
        rows.sort(reverse=True)
        for size, nfiles, err, name in rows:
            print("  CHILD %-60s GiB=%8.3f files=%-8d errors=%d" % (name, size / 2**30, nfiles, err))
        print("  ROOT_TOTAL GiB=%.3f children=%d secs=%.1f" % (root_total / 2**30, len(rows), time.time() - t0))
        grand_total += root_total
        sys.stdout.flush()
    print("GRAND_TOTAL GiB=%.3f" % (grand_total / 2**30))
