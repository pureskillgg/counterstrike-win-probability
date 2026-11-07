import os, sys
sys.path.insert(0, sys.argv[1])
from walkmod import walk
SRC = "G:/pureskill/csds-cs2"
DST = "C:/Users/<user>/Backups/csds-cs2_2026-11-06"
RP = 0x400
TS = dict(files=0, bytes=0); TD = dict(files=0, bytes=0)
mism = []
for name in sorted(os.listdir(SRC)):
    p = os.path.join(SRC, name)
    st = os.lstat(p)
    if st.st_file_attributes & RP:
        continue
    if os.path.isdir(p):
        s = walk(p)
        q = os.path.join(DST, name)
        d = walk(q) if os.path.isdir(q) else dict(files=-1, bytes=-1)
    else:
        s = dict(files=1, bytes=st.st_size)
        q = os.path.join(DST, name)
        d = dict(files=1, bytes=os.path.getsize(q)) if os.path.exists(q) else dict(files=-1, bytes=-1)
    TS["files"] += s["files"]; TS["bytes"] += s["bytes"]
    TD["files"] += d["files"]; TD["bytes"] += d["bytes"]
    if s["files"] != d["files"] or s["bytes"] != d["bytes"]:
        mism.append((name, s, d))
print("TOTAL src files=%d bytes=%d" % (TS["files"], TS["bytes"]))
print("TOTAL dst files=%d bytes=%d" % (TD["files"], TD["bytes"]))
print("top-level entries with mismatch (files or exact bytes):", len(mism))
for m in mism:
    print(m)
