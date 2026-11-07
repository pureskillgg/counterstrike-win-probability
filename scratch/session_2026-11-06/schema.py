import pyarrow.parquet as pq, glob
F='/home/sagemaker-user/adam/builder/final'
for p in [F+'/pos_parts/bucket_c000.parquet', F+'/pos_parts/bucket_c046.parquet', '/home/sagemaker-user/adam/builder/data/c000/matches.parquet', '/home/sagemaker-user/adam/builder/data/c067/matches.parquet', F+'/bucket_matches.parquet', F+'/clean_matches.parquet']:
    md = pq.ParquetFile(p).metadata
    print(p, 'rows', md.num_rows, 'groups', md.num_row_groups)
    print('  cols', pq.ParquetFile(p).schema_arrow.names[:40])
