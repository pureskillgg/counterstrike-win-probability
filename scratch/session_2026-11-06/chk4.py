import pyarrow.parquet as pq, csv, collections
import pyarrow.compute as pc
B = '/home/sagemaker-user/adam/builder'
F = B + '/fixes/fix332'
P = B + '/final/pos_all_seconds/'
cands = list(csv.DictReader(open(F + '/candidates.csv')))
cm = {(r['match_key'], int(r['round'])): r for r in cands}
fixed = set((r['match_key'], int(r['round'])) for r in cands) - set(
    (r['match_key'], int(r['round'])) for r in cands if False)
ov = list(csv.DictReader(open(B + '/fixes/freeze_fix/override_rounds.csv')))
fixed = set((r['match_key'], int(r['round'])) for r in ov if r['status'] == 'fixed')
affected_m = set(k[0] for k in cm)
print('fixed', len(fixed), 'affected matches', len(affected_m))
match_chunk = collections.defaultdict(set)
for c in ['c029', 'c043', 'c044', 'c045', 'c046']:
    t = pq.read_table(P + 'bucket_%s.parquet' % c, columns=['match_key', 'round', 'second'])
    mk = t.column('match_key').to_pylist()
    rd = t.column('round').to_pylist()
    sc = t.column('second').to_pylist()
    keys = collections.Counter(zip(mk, rd, sc))
    dups = sum(v - 1 for v in keys.values() if v > 1)
    chunk_matches = set(mk)
    aff_here = chunk_matches & affected_m
    for m in aff_here:
        match_chunk[m].add(c)
    rounds_here = set(zip(mk, rd))
    aff_rounds = set(k for k in rounds_here if k in cm)
    fx = set(k for k in aff_rounds if k in fixed)
    rem = aff_rounds - fx
    fixed_matches_here = set(k[0] for k in fx)
    print(c, 'rows', len(mk), 'dup keys', dups, 'matches', len(chunk_matches),
          'affected matches', len(aff_here), 'affected rounds', len(aff_rounds),
          'fixed rounds', len(fx), 'rem rounds', len(rem), 'fixed matches', len(fixed_matches_here))
    # affected rounds that are in candidates but absent in chunk
print('affected matches with chunk multi', sum(1 for m in match_chunk if len(match_chunk[m]) > 1))
print('affected matches with no chunk', len(affected_m - set(match_chunk)))
allfixed_in_chunks = set()
for c in ['c029', 'c043', 'c044', 'c045', 'c046']:
    pass
