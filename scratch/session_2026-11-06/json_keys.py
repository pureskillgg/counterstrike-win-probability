import subprocess, re, json, collections
REPO = r'C:\Users\13144\Documents\pureskill\counterstrike-win-probability'
S = r'C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad'
IDKEY = re.compile(r'steam|player|name|user|account|sid|nick|avatar|\bip\b|email|phone|address|uid|pid|guid|team|sender|author|owner', re.I)
rows = [l.rstrip('\n').split('\t') for l in open(S + r'\hist_blob_list.txt', encoding='utf-8') if l.strip()]
allkeys = collections.Counter(); idhits = collections.defaultdict(set)
def walk(o, path, prefix=''):
    if isinstance(o, dict):
        for k, v in o.items():
            allkeys[k] += 1
            if IDKEY.search(k): idhits[path].add(prefix + '.' + k if prefix else k)
            walk(v, path, (prefix + '.' + k) if prefix else k)
    elif isinstance(o, list):
        for x in o[:50]: walk(x, path, prefix + '[]')
seen = set()
for commit, path in rows:
    if not path.endswith('.json'): continue
    blob = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=REPO, capture_output=True, check=True).stdout
    try: d = json.loads(blob.decode('utf-8', 'replace'))
    except Exception as e: print('ERR', path, e); continue
    walk(d, path)
print('json blobs walked:', sum(1 for _, p in rows if p.endswith('.json')))
print('identifier-like JSON key paths (path: key paths):')
for p, ks in idhits.items(): print('  ', p, sorted(ks)[:12])
print('distinct JSON key names:', len(allkeys))
print(sorted(allkeys.keys())[:400])
