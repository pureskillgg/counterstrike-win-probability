import csv, re, sys, collections
UUID = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
STEAM = re.compile(r'7656119\d{10}')
PATH = re.compile(r'^csds/\d{4}/\d{2}/\d{2}/' )
def cls(s):
    return re.sub(r'[a-z]', 'a', re.sub(r'[A-Z]', 'A', re.sub(r'\d', '9', s)))
targets = {}
for p in sys.argv[1:]:
    with open(p, encoding='utf-8', newline='') as fh:
        r = csv.DictReader(fh)
        cols = [c for c in (r.fieldnames or []) if c and re.search(r'match|player|steam|name|user|account|sid|team|demo|id$|key', c, re.I)]
        counts = {c: collections.Counter() for c in cols}
        n = 0; steam_hits = 0; uuid_hits = 0; path_hits = 0; total_cells=0
        for row in r:
            n += 1
            for c in cols:
                v = row.get(c) or ''
                if not v: counts[c]['<empty>'] += 1; continue
                if STEAM.search(v): steam_hits += 1
                if UUID.search(v): uuid_hits += 1
                if PATH.match(v): path_hits += 1
                counts[c][('uuid' if UUID.search(v) else cls(v)[:40])] += 1
        print('==', p, 'rows=', n)
        print('   candidate id cols:', cols)
        for c in cols:
            top = counts[c].most_common(4)
            print('   ', c, 'distinct_shapes_top4=', [(k if k in ('uuid','<empty>') else k, v) for k, v in top], 'n_distinct=', len(counts[c]))
        print('   steam-id-pattern cell hits:', steam_hits, ' uuid-pattern hits:', uuid_hits, ' csds-path hits:', path_hits)
