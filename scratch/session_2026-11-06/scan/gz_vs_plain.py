import gzip, hashlib, os, sys
def sha_plain(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(16 << 20), b''):
            h.update(c)
    return h.hexdigest(), os.path.getsize(p)
def sha_gz(p):
    h = hashlib.sha256(); n = 0
    with gzip.open(p, 'rb') as f:
        for c in iter(lambda: f.read(16 << 20), b''):
            h.update(c); n += len(c)
    return h.hexdigest(), n
pairs = [
    ('G:/pureskill/csds-cs2/staging_cs2data_gz/data/combined/cache_pos_2026_trainctr.parquet.gz', 'G:/pureskill/csds-cs2/data/combined/cache_pos_2026_trainctr.parquet'),
    ('G:/pureskill/csds-cs2/staging_cs2data_gz/data/combined/cache_snapshot_2026.parquet.gz', 'G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026.parquet'),
]
for gzp, plain in pairs:
    if not (os.path.exists(gzp) and os.path.exists(plain)):
        print('missing', gzp if not os.path.exists(gzp) else plain, flush=True); continue
    a = sha_gz(gzp); b = sha_plain(plain)
    print(os.path.basename(plain), 'gz-decompressed bytes', a[1], 'plain bytes', b[1], 'sha equal', a[0] == b[0], flush=True)
print('DONE', flush=True)
