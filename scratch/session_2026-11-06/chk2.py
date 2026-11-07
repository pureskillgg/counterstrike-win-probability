import csv, json, collections
B = '/home/sagemaker-user/adam/builder'
F = B + '/fixes/fix332'
ov = list(csv.DictReader(open(B + '/fixes/freeze_fix/override_rounds.csv')))
print('status', collections.Counter(r['status'] for r in ov))
print('reason', collections.Counter(r['reason'] for r in ov))
print('override col', collections.Counter(r['override'] for r in ov).most_common(5))
odd = [r for r in ov if r['status'] != ov[0]['status']]
for r in odd[:3]:
    print('ODD', {k: (v[:120] if isinstance(v, str) else v) for k, v in r.items()})
print('sample row', {k: (v[:160] if isinstance(v, str) else v) for k, v in ov[0].items()})
r18 = [r for r in ov if '2d963086' in r['match_key'] and r['round'] == '18']
print('2d963086 r18 rows', len(r18))
for r in r18:
    print({k: (v[:160] if isinstance(v, str) else v) for k, v in r.items()})

cands = list(csv.DictReader(open(F + '/candidates.csv')))
cm = {(r['match_key'], int(r['round'])): r for r in cands}
print('r18 in candidates', cm.get(('csds/2026/06/28/2d963086-4d4f-4ff0-8f94-183cb0e15f2c/csds', 18)))
