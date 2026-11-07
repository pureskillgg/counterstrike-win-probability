import sys, os, tarfile, json
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
sys.path.insert(0, r'C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\scan')
from pii_scan_fast import Acc
TEXT = ('.py', '.sh', '.log', '.json', '.csv', '.txt', '.md', '.toml', '.yaml', '.yml', '.cfg', '.ini', '.ipynb', '.stackdump', '.html')
SKIPNAME = ('frozen', 'holdout', 'oct26', 'oct_26', '10-26', '10_26')
root = sys.argv[1]
summary = {}
for dp, dn, fn in os.walk(root):
    for f in fn:
        if not (f.endswith('.tar') or f.endswith('.tar.gz') or f.endswith('.tgz')):
            continue
        p = os.path.join(dp, f)
        mode = 'r:gz' if f.endswith(('.gz', '.tgz')) else 'r:'
        n_members = 0; scanned = 0; names_nontext = 0; hits = {}
        try:
            with tarfile.open(p, mode) as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    n_members += 1
                    low = m.name.lower()
                    if not low.endswith(TEXT) or any(s in low for s in SKIPNAME):
                        names_nontext += 1
                        continue
                    if m.size > 60 * 1024 * 1024:
                        names_nontext += 1
                        continue
                    fh = tf.extractfile(m)
                    acc = Acc()
                    acc.feed(fh.read(), 0)
                    scanned += 1
                    for k, v in acc.counts.items():
                        if v:
                            hits.setdefault(m.name, {})[k] = v
                    if acc.examples:
                        hits.setdefault(m.name, {})['_masked'] = {k: v for k, v in acc.examples.items() if v}
        except Exception as e:
            print('ERR', f, type(e).__name__, str(e)[:120], flush=True); continue
        print(json.dumps({'tar': f, 'members_files': n_members, 'text_scanned': scanned, 'non_text_or_skipped_names_only': names_nontext, 'hit_members': len(hits)}), flush=True)
        for mname, hh in list(hits.items())[:40]:
            cats = {k: v for k, v in hh.items() if k != '_masked'}
            print('   HIT', f, mname, cats, hh.get('_masked', ''), flush=True)
print('DONE tar', flush=True)
