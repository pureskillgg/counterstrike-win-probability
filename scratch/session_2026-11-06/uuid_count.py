import subprocess, re
REPO = r'C:\Users\<user>\Documents\pureskill\counterstrike-win-probability'
S = r'C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad'
PAT = re.compile(rb'csds/\d{4}/\d{2}/\d{2}/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/csds')
rows = [l.rstrip('\n').split('\t') for l in open(S + r'\public_data_paths.txt', encoding='utf-8') if False]
pub = subprocess.run(['git', 'log', '51a21d4a', '--format=%H'], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
seen = set(); per_file = {}
for c in pub:
    files = subprocess.run(['git', 'ls-tree', '-r', '--name-only', c], cwd=REPO, capture_output=True, check=True).stdout.decode().split('\n')
    for f in files:
        if not re.search(r'\.(csv|json|txt)$', f): continue
        blob = subprocess.run(['git', 'show', f'{c}:{f}'], cwd=REPO, capture_output=True).stdout
        ids = set(m.group(1) for m in PAT.finditer(blob))
        seen |= ids
        per_file.setdefault(f, set()).update(ids)
print('public-branch commits:', len(pub), ' distinct match-key UUIDs across committed data:', len(seen))
print('files containing match-key UUIDs:', sum(1 for v in per_file.values() if v))
