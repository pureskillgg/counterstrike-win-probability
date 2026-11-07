import subprocess, re, json, csv, io, sys, collections
REPO = r'C:\Users\<user>\Documents\pureskill\counterstrike-win-probability'
S = r'C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad'
STEAM = re.compile(rb'7656119\d{10}')
UUID = re.compile(rb'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
IDCOL = re.compile(r'steam|player|name|user|account|sid|nick|avatar|ip|email|phone|address|uid|pid|guid', re.I)
rows = [l.rstrip('\n').split('\t') for l in open(S + r'\hist_blob_list.txt', encoding='utf-8') if l.strip()]
sigs = collections.Counter(); examples = {}
steam_total = 0; uuid_total = 0; idcols = collections.defaultdict(set); errs = []
for commit, path in rows:
    try:
        blob = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=REPO, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        errs.append((commit[:8], path, 'show-fail')); continue
    st = len(STEAM.findall(blob)); ut = len(UUID.findall(blob))
    steam_total += st; uuid_total += ut
    kind = 'other'; schema = ''
    if path.endswith('.csv'):
        kind = 'csv'
        try:
            header = next(csv.reader(io.StringIO(blob.decode('utf-8', 'replace'))))
            schema = ','.join(header)
            cols = [c for c in header if IDCOL.search(c)]
            for c in cols: idcols[path].add(c)
        except Exception as e:
            schema = f'ERR {e}'
    elif path.endswith('.json'):
        kind = 'json'
        try:
            d = json.loads(blob.decode('utf-8', 'replace'))
            if isinstance(d, dict): schema = 'dict:' + ','.join(sorted(d.keys()))[:300]
            elif isinstance(d, list): schema = f'list[{len(d)}] of {type(d[0]).__name__ if d else "empty"}'
            else: schema = type(d).__name__
        except Exception as e:
            schema = f'ERR {e}'
    else:
        schema = 'non-csv/json'
    sigs[(kind, schema[:200])] += 1
    examples.setdefault((kind, schema[:200]), path)
    if st: print(f'steam-id-pattern hits={st} commit={commit[:8]} path={path}')
print('blobs screened:', len(rows), 'errors:', errs[:5])
print('total steam-id-pattern cell hits across all committed blobs:', steam_total)
print('total uuid-pattern hits across all committed blobs:', uuid_total)
print('identifier-like column names found in CSV blobs (path -> cols):')
for p, c in idcols.items(): print('  ', p, sorted(c))
print('distinct schema signatures:', len(sigs))
for (k, s), n in sigs.most_common(60):
    print(f'  x{n} [{k}] {s[:160]}')
