import csv, json, collections, os
B = '/home/sagemaker-user/adam/builder'
F = B + '/fixes/fix332'
cands = list(csv.DictReader(open(F + '/candidates.csv')))
print('candidates rows', len(cands))
kinds = collections.Counter(r['kind'] for r in cands)
print('kinds', dict(kinds))

# fixed set from override map
ov = list(csv.DictReader(open(B + '/fixes/freeze_fix/override_rounds.csv')))
print('override rows', len(ov), 'cols', list(ov[0].keys())[:12])
ovkeys = set()
for r in ov:
    ovkeys.add((r.get('match_key'), int(r.get('round'))))
print('override distinct (match,round)', len(ovkeys))

cand_keys = {(r['match_key'], int(r['round'])): r for r in cands}
fixed_c = [k for k in cand_keys if k in ovkeys]
print('candidates in override', len(fixed_c))
print('kinds of override-covered candidates', dict(collections.Counter(cand_keys[k]['kind'] for k in fixed_c)))
rem = [k for k in cand_keys if k not in ovkeys]
print('candidates not in override', len(rem), dict(collections.Counter(cand_keys[k]['kind'] for k in rem)))
print('override keys not in candidates', len(ovkeys - set(cand_keys)))

# matches
fm = set(k[0] for k in fixed_c)
rm = set(k[0] for k in rem)
am = set(cand_keys_m[0] for cand_keys_m in cand_keys)
print('affected matches', len(am), 'fixed matches', len(fm), 'remaining matches', len(rm), 'both', len(fm & rm), 'remaining only', len(rm - fm))
