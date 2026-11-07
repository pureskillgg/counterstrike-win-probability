import json, os, glob
import pyarrow.parquet as pq
import pyarrow.compute as pc

D = '/home/sagemaker-user/adam/builder/data'
F = '/home/sagemaker-user/adam/builder/final'
E = F + '/export_clean_v2'

export_keys = set(json.load(open(E + '/match_keys.json')))

print('== prep reports c000..c067')
rep = {}
missing = []
for c in range(68):
    p = f'{D}/c{c:03d}/prep_report.json'
    if not os.path.exists(p):
        missing.append(c)
        continue
    rep[c] = json.load(open(p))
print('chunks with prep_report', len(rep), 'missing', missing)
for k in ['matches', 'use_matches', 'use_rounds', 'rounds_total', 'round_end_rows', 'dup_matches_removed']:
    tot = sum(rep[c].get(k, 0) for c in rep)
    a = sum(rep[c].get(k, 0) for c in rep if c <= 46)
    b = sum(rep[c].get(k, 0) for c in rep if c >= 47)
    print(f'{k:20s} total={tot} c000-c046={a} c047-c067={b}')
print('store field matches chunk id', all(rep[c]['store'] == f'c{c:03d}' for c in rep))

print('\n== chunk matches.parquet keys')
chunk_keys = {}
for c in range(68):
    p = f'{D}/c{c:03d}/matches.parquet'
    t = pq.read_table(p, columns=['match_key'])
    chunk_keys[c] = t.column('match_key').to_pylist()
rows_total = sum(len(v) for v in chunk_keys.values())
uniq_chunk = set(k for v in chunk_keys.values() for k in v)
print('matches.parquet rows total', rows_total, 'unique keys', len(uniq_chunk))
bm = pq.read_table(F + '/bucket_matches.parquet', columns=['match_key']).column('match_key').to_pylist()
print('bucket_matches.parquet rows', len(bm), 'unique', len(set(bm)))
print('bucket_matches == union of chunk matches', set(bm) == uniq_chunk)
print('chunk rows == prep matches per chunk for all 68', all(len(chunk_keys[c]) == rep[c]['matches'] for c in rep))

print('\n== pos_parts coverage')
pp = sorted(glob.glob(F + '/pos_parts/bucket_c*.parquet'))
print('pos_parts files', len(pp), 'first', os.path.basename(pp[0]), 'last', os.path.basename(pp[-1]))
pp_nums = sorted(int(os.path.basename(p)[8:11]) for p in pp)
print('pos_parts chunk numbers contiguous 0..46', pp_nums == list(range(47)))
pp_rows = 0
pp_keys = set()
per_chunk_rows = {}
for p in pp:
    t = pq.read_table(p, columns=['match_key'])
    ks = t.column('match_key').to_pylist()
    pp_rows += len(ks)
    pp_keys |= set(ks)
    c = int(os.path.basename(p)[8:11])
    per_chunk_rows[c] = (len(ks), len(set(ks)))
print('pos_parts total rows', pp_rows, 'unique match_keys', len(pp_keys))
print('pos_parts keys in export', len(pp_keys & export_keys), 'pos_parts keys not in export', len(pp_keys - export_keys))
print('pos_parts keys subset of bucket chunk universe', len(pp_keys & uniq_chunk), 'of', len(pp_keys))
pp_chunks_covered = {c for c in range(68) if set(chunk_keys[c]) & pp_keys}
print('chunks with any key in pos_parts', len(pp_chunks_covered), 'min', min(pp_chunks_covered), 'max', max(pp_chunks_covered))
chunks_fully_in_pos = [c for c in range(68) if set(chunk_keys[c]) <= pp_keys]
print('chunks whose every match is in pos_parts', len(chunks_fully_in_pos))

print('\n== export coverage by source')
cm = pq.read_table(F + '/clean_matches.parquet', columns=['match_key', 'src', 'in_scope', 'dup_keeper', 'n_ok_rounds'])
cmd = cm.to_pydict()
src_keys = {'bucket': set(), 'laptop': set()}
for k, s in zip(cmd['match_key'], cmd['src']):
    src_keys.setdefault(s, set()).add(k)
print('clean_matches src counts', {s: len(v) for s, v in src_keys.items()})
print('export keys in bucket src', len(export_keys & src_keys.get('bucket', set())), '| in laptop src', len(export_keys & src_keys.get('laptop', set())), '| in neither', len(export_keys - src_keys.get('bucket', set()) - src_keys.get('laptop', set())))
exp_bucket = export_keys & uniq_chunk
print('export keys that are in bucket chunk universe', len(exp_bucket))
exp_not_pos = export_keys - pp_keys
print('export keys NOT in pos_parts', len(exp_not_pos))
print('  of which in bucket chunk universe', len(exp_not_pos & uniq_chunk), '| laptop src', len(exp_not_pos & src_keys.get('laptop', set())))
exp_chunk_not_pos = (export_keys & uniq_chunk) - pp_keys
print('export bucket keys NOT in pos_parts', len(exp_chunk_not_pos))
ch_of = {}
for c, v in chunk_keys.items():
    for k in v:
        ch_of.setdefault(k, c)
miss_by_chunk = {}
for k in exp_chunk_not_pos:
    miss_by_chunk[ch_of[k]] = miss_by_chunk.get(ch_of[k], 0) + 1
print('export bucket keys missing from pos_parts, by chunk', dict(sorted(miss_by_chunk.items())))
print('per-pos-part rows (chunk: rows,unique) sample', {c: per_chunk_rows[c] for c in list(per_chunk_rows)[:3]}, '...', {c: per_chunk_rows[c] for c in list(per_chunk_rows)[-3:]})
