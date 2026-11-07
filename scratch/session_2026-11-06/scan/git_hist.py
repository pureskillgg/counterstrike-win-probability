#!/usr/bin/env python3
"""Scan every blob reachable from any ref (including deleted history) with the PII pattern set.
Frozen / holdout / Oct 26-31 named blobs are skipped (name only). Counts and masked examples only."""
import sys, subprocess, re
from collections import Counter, defaultdict
SCANDIR = r'C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\scan'
sys.path.insert(0, SCANDIR)
from pii_scan import Acc, COMPILED

REPO = sys.argv[1]
SKIP = re.compile(r'(?i)(frozen|holdout|oct(ober)?_?2[6-9]|oct_?3[01]|10_?2[6-9]|10_?3[01])')


def git(args, inp=None):
    return subprocess.run(['git', '-C', REPO] + args, input=inp, capture_output=True).stdout


objs = git(['rev-list', '--objects', '--all']).decode('utf-8', 'replace').splitlines()
paths = {}
order = []
for line in objs:
    sha, _, path = line.partition(' ')
    order.append(sha)
    if path and sha not in paths:
        paths[sha] = path
types = git(['cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
            ('\n'.join(order) + '\n').encode()).decode().splitlines()
blobs = []
for t in types:
    sha, typ, size = t.split()
    if typ == 'blob':
        blobs.append((sha, int(size)))
print('objects', len(order), 'blobs', len(blobs))

scanned = skipped = 0
hit_paths = {}
for sha, size in blobs:
    path = paths.get(sha, '(no-path)')
    base = path.rsplit('/', 1)[-1]
    if base.lower().startswith('.env'):
        skipped += 1
        continue
    if SKIP.search(path) and not base.endswith(('.py', '.sh', '.md')):
        skipped += 1
        continue
    content = git(['cat-file', 'blob', sha])
    acc = Acc()
    acc.feed(content, 0)
    scanned += 1
    hits = {k: v for k, v in acc.counts.items() if v}
    if hits:
        hit_paths[path] = (hits, dict(acc.examples))

print('blobs scanned', scanned, 'skipped (name-only or .env)', skipped)
cnt = Counter()
for p, (h, ex) in hit_paths.items():
    for k, v in h.items():
        cnt[k] += v
print('total hits by pattern:', dict(cnt))
print('paths with any hit:', len(hit_paths))
for p in sorted(hit_paths)[:120]:
    h, ex = hit_paths[p]
    print(p, h, ex if ex else '')

# commit author / committer identities (masked)
emails = git(['log', '--all', '--format=%ae|%ce']).decode('utf-8', 'replace').splitlines()
uniq = Counter()
for line in emails:
    for e in line.split('|'):
        if '@' in e:
            uniq[e] += 1
print('distinct author/committer emails in history:', len(uniq))
for e, c in uniq.most_common(10):
    local, _, dom = e.partition('@')
    print('  masked author email: %s***@%s  (commits: %d)' % (local[:3], dom, c))
