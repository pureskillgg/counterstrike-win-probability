import os, sys, collections
roots = sys.argv[1:]
ext_cnt = collections.Counter(); ext_bytes = collections.Counter(); top = collections.Counter(); top_bytes = collections.Counter()
big = []
for root in roots:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != '.git']
        for f in fn:
            p = os.path.join(dp, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            ext = os.path.splitext(f)[1].lower() or '(none)'
            if f.lower().startswith('.env'): ext = '.env-SKIPPED'
            ext_cnt[ext] += 1; ext_bytes[ext] += st.st_size
            rel = os.path.relpath(dp, root).split(os.sep)[0]
            key = os.path.join(root, rel)
            top[key] += 1; top_bytes[key] += st.st_size
            if st.st_size > 200_000_000: big.append((st.st_size, p))
print('== by extension (count, MB)')
for e, c in ext_cnt.most_common(45): print(f'{c:8d} {ext_bytes[e]/1e6:12.1f} {e}')
print('== by top-level folder (count, MB)')
for k in sorted(top, key=lambda k: -top_bytes[k])[:70]: print(f'{top[k]:8d} {top_bytes[k]/1e6:12.1f} {k}')
print('== files > 200MB')
for s, p in sorted(big, reverse=True)[:40]: print(f'{s/1e6:10.1f} {p}')
