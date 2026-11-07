import os, sys, json
RP = 0x400
def walk(root):
    files = dirs = nbytes = reparse = 0
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            it = os.scandir(cur)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if st.st_file_attributes & RP:
                    reparse += 1
                    continue
                if e.is_dir(follow_symlinks=False):
                    dirs += 1
                    stack.append(e.path)
                else:
                    files += 1
                    nbytes += st.st_size
    return dict(files=files, dirs=dirs, bytes=nbytes, reparse_skipped=reparse)
pairs = {}
src = "/g/pureskill/csds-cs2"
dst = "/c/Users/13144/Backups/csds-cs2_2026-11-06"
out = {}
for name in sorted(os.listdir(src)):
    p = os.path.join(src, name)
    if os.path.isdir(p) or os.path.islink(p):
        out[name] = {"src": None, "dst": None}
results = {}
# top-level entries in source, measured without following reparse points
for name in sorted(os.listdir(src)):
    p = os.path.join(src, name)
    try:
        st = os.lstat(p)
    except OSError:
        continue
    if st.st_file_attributes & RP if hasattr(st, 'st_file_attributes') else False:
        results[name] = {"note": "reparse point at top level, skipped in source walk"}
        continue
    if os.path.isdir(p):
        s = walk(p)
        q = os.path.join(dst, name)
        d = walk(q) if os.path.isdir(q) else None
        results[name] = {"src": s, "dst": d}
    else:
        s_st = os.stat(p)
        q = os.path.join(dst, name)
        d_st = os.stat(q) if os.path.exists(q) else None
        results[name] = {"src": {"files": 1, "bytes": s_st.st_size}, "dst": ({"files": 1, "bytes": d_st.st_size} if d_st else None)}
with open(sys.argv[1], "w") as f:
    json.dump(results, f, indent=1)
print("written", sys.argv[1], len(results), "top-level entries")
