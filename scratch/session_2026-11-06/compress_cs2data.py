"""Compress the cs2 sequence data into a staging folder before upload (gzip level 1, streaming).

Sources (laptop G:): data/round_state (all files), data/combined/cache_pos_2026_trainctr.parquet,
data/combined/cache_snapshot_2026.parquet, data/diagnostics/allmaps_geo_trainsites (all files).
Output: staging_cs2data_gz/<same relative path>.gz  plus manifest.json with sha256 of the ORIGINAL file (checked on the pod).
"""
import gzip
import hashlib
import json
import os
import shutil
import time

ROOT = "G:/pureskill/csds-cs2/"
OUT = ROOT + "staging_cs2data_gz/"
SOURCES = [
    "data/round_state",
    "data/combined/cache_pos_2026_trainctr.parquet",
    "data/combined/cache_snapshot_2026.parquet",
    "data/diagnostics/allmaps_geo_trainsites",
]
SKIP = {".gitignore"}
os.makedirs(OUT, exist_ok=True)
manifest = []
t0 = time.time()


def files_under(p):
    full = ROOT + p
    if os.path.isfile(full):
        yield p
        return
    for dp, dn, fn in os.walk(full):
        dn[:] = [d for d in dn if d not in (".meridian-outputs-cache",)]
        for f in fn:
            if f in SKIP:
                continue
            yield os.path.relpath(os.path.join(dp, f), ROOT).replace("\\", "/")


total_in = total_out = 0
for src in SOURCES:
    for rel in files_under(src):
        inp = ROOT + rel
        outp = OUT + rel + ".gz"
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        h = hashlib.sha256()
        with open(inp, "rb") as fi, gzip.open(outp, "wb", compresslevel=1) as fo:
            for chunk in iter(lambda: fi.read(64 * 1024 * 1024), b""):
                h.update(chunk)
                fo.write(chunk)
        n_in = os.path.getsize(inp)
        n_out = os.path.getsize(outp)
        total_in += n_in
        total_out += n_out
        manifest.append({"path": rel, "bytes": n_in, "gz_bytes": n_out, "sha256": h.hexdigest()})
        print(f"{rel}: {n_in/1e6:.1f} MB -> {n_out/1e6:.1f} MB  [{time.time()-t0:.0f}s]", flush=True)

json.dump({"files": manifest, "total_bytes": total_in, "total_gz_bytes": total_out}, open(OUT + "manifest.json", "w"), indent=1)
print(f"TOTAL {total_in/1e9:.2f} GB -> {total_out/1e9:.2f} GB, {len(manifest)} files, {time.time()-t0:.0f}s", flush=True)
