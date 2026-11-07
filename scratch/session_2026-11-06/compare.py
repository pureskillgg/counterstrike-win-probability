import os, sys
sys.path.insert(0, sys.argv[1])
from walkmod import walk
SRC = "G:/pureskill/csds-cs2"
DST = "C:/Users/<user>/Backups/csds-cs2_2026-11-06"
RP = 0x400
tot_s = tot_d = None
rows = []
for name in sorted(os.listdir(SRC)):
    p = os.path.join(SRC, name)
    st = os.lstat(p)
    if st.st_file_attributes & RP:
        rows.append((name, "REPARSE (junction) - excluded from copy", None, None))
        continue
    if os.path.isdir(p):
        s = walk(p)
        q = os.path.join(DST, name)
        d = walk(q) if os.path.isdir(q) else dict(files=-1, bytes=-1)
        rows.append((name, s, d, None))
    else:
        q = os.path.join(DST, name)
        s_b = st.st_size
        d_b = os.path.getsize(q) if os.path.exists(q) else -1
        rows.append((name, dict(files=1, bytes=s_b), dict(files=1 if d_b >= 0 else 0, bytes=d_b), None))
for name, s, d, _ in rows:
    if isinstance(s, str):
        print(f"{name:45s} {s}")
        continue
    ok = (s["files"] == d["files"] and s["bytes"] == d["bytes"])
    print(f"{name:45s} src_files={s['files']:>7} dst_files={d['files']:>7} src_GiB={s['bytes']/1024**3:9.3f} dst_GiB={d['bytes']/1024**3:9.3f} {'OK' if ok else 'MISMATCH'}")
