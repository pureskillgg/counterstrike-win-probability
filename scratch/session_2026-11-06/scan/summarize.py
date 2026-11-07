import sys, json
from collections import Counter
rep = Counter(); skipped = Counter(); errors = []
hits_rows = []
cats_all = Counter()
for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
    line = line.strip()
    if not line.startswith('{'):
        continue
    d = json.loads(line)
    if 'skipped' in d:
        skipped[d['skipped']] += 1; continue
    if 'error' in d:
        errors.append((d.get('obj'), d.get('error'), d.get('msg', d.get('err', ''))[:80])); continue
    rep['objects'] += 1
    h = d.get('hits', {})
    for c, names in h.items():
        for n, v in names.items():
            cats_all[(c, n)] += v
    if any(c in ('id', 'pii', 'name', 'chat', 'secret', 'linkage') for c in h):
        hits_rows.append(d)
print('scanned objects', rep['objects'], '| errors', len(errors), '| skipped', dict(skipped))
print('hit totals by pattern:', {f'{c}:{n}': v for (c, n), v in sorted(cats_all.items())})
for e in errors[:10]: print('ERR', e)
mode = sys.argv[2] if len(sys.argv) > 2 else 'brief'
for d in hits_rows:
    h = {c: v for c, v in d.get('hits', {}).items() if c in ('id', 'pii', 'name', 'chat', 'secret', 'linkage', 'holdout_rule')}
    ex = d.get('masked_examples', {})
    print('HIT', d['obj'][-150:], h, ex if ex else '')
