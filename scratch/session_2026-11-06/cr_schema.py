import pyarrow.parquet as pq
p='/home/sagemaker-user/adam/builder/final/clean_rounds.parquet'
pf=pq.ParquetFile(p)
print('rows', pf.metadata.num_rows)
print([ (f.name,str(f.type)) for f in pf.schema_arrow])
