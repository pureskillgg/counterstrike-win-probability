import os, sys
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
root = sys.argv[1]
for entry in sorted(os.scandir(root), key=lambda e: e.name):
    n = 0; b = 0
    if entry.is_dir(follow_symlinks=False):
        for dp, dn, fn in os.walk(entry.path, followlinks=False):
            for f in fn:
                try:
                    b += os.path.getsize(os.path.join(dp, f)); n += 1
                except OSError:
                    pass
            if n > 3_000_000: break
    else:
        try: b = entry.stat().st_size; n = 1
        except OSError: pass
    print(f'{n:9d} files {b/1e6:12.1f} MB  {entry.name}', flush=True)
