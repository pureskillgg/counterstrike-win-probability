import pyarrow.parquet as pq, json, os
B = '/home/sagemaker-user/adam/builder'
P = B + '/final/pos_all_seconds/'
for c in ['c000', 'c029', 'c043', 'c044', 'c045', 'c046']:
    f = P + 'bucket_%s.parquet' % c
    md = pq.ParquetFile(f).metadata
    names = pq.ParquetFile(f).schema_arrow.names
    print(c, 'rows', md.num_rows, 'ncols', len(names))
    if c in ('c000', 'c043'):
        print('  cols', names)
        print('  types', [str(t) for t in pq.ParquetFile(f).schema_arrow.types])
        print('  side/player cols', [n for n in names if 'side' in n.lower() or 'player' in n.lower() or 'pid' in n.lower() or 'team' in n.lower()])
print('dir entries', len(os.listdir(P)), sorted(os.listdir(P))[-4:])
print('README', open(B + '/final/export_clean_v4/pos_v4/README.json').read()[:300] if os.path.exists(B + '/final/export_clean_v4/pos_v4/README.json') else 'no pos_v4 README')
