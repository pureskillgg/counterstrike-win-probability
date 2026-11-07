"""E3 size probe: apparent size, file count, and hard-link status per directory tree.

Read-only. Walks each given directory with os.scandir, sums st_size, and uses
os.stat(...).st_nlink (Windows reports the hard-link count) to tell how many files
are hard links shared with another path (for example the pixi/rattler package cache).

Usage:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 E3_size_probe.py <dir> [<dir> ...]
"""
import os
import sys
import time


def probe(root):
    t0 = time.time()
    files = 0
    total = 0
    linked_files = 0  # st_nlink > 1
    linked_bytes = 0
    unlinked_bytes = 0  # st_nlink == 1
    errors = 0
    top = {}  # first-level child -> [files, bytes]
    stack = [(root, None)]
    while stack:
        d, tag = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            errors += 1
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append((e.path, tag if tag is not None else e.name))
                        continue
                    if not e.is_file(follow_symlinks=False):
                        continue
                    st = os.stat(e.path, follow_symlinks=False)
                except OSError:
                    errors += 1
                    continue
                files += 1
                total += st.st_size
                bucket = tag if tag is not None else "(files at root)"
                rec = top.setdefault(bucket, [0, 0])
                rec[0] += 1
                rec[1] += st.st_size
                if st.st_nlink > 1:
                    linked_files += 1
                    linked_bytes += st.st_size
                else:
                    unlinked_bytes += st.st_size
    return {
        "root": root,
        "files": files,
        "apparent_gib": round(total / 2**30, 3),
        "hardlinked_files": linked_files,
        "hardlinked_gib": round(linked_bytes / 2**30, 3),
        "single_link_gib": round(unlinked_bytes / 2**30, 3),
        "errors": errors,
        "seconds": round(time.time() - t0, 1),
        "top_level": {k: [v[0], round(v[1] / 2**30, 3)] for k, v in sorted(top.items(), key=lambda kv: -kv[1][1])},
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        r = probe(arg)
        print("ROOT", r["root"])
        print("  files=%d apparent_GiB=%.3f hardlinked_files=%d hardlinked_GiB=%.3f single_link_GiB=%.3f errors=%d secs=%.1f"
              % (r["files"], r["apparent_gib"], r["hardlinked_files"], r["hardlinked_gib"],
                 r["single_link_gib"], r["errors"], r["seconds"]))
        for k, v in list(r["top_level"].items())[:25]:
            print("  TOP %-40s files=%-7d GiB=%.3f" % (k, v[0], v[1]))
        sys.stdout.flush()
