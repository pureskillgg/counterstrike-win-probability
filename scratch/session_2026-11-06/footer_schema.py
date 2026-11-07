import subprocess, struct, os, sys, json
import pyarrow.parquet as pq
BUCKET = 's3://z7yvc40ykx/'
AWS = ['aws', '--profile', 'runpod', '--endpoint-url', 'https://s3api-eu-ro-1.runpod.io', '--region', 'eu-ro-1']
S = r'C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\footers'
def s3api(*args):
    return subprocess.run(AWS + ['s3api'] + list(args), capture_output=True, text=True, check=True)
def fetch_range(key, rng, outpath):
    s3api('get-object', '--bucket', 'z7yvc40ykx', '--key', key, '--range', rng, outpath)
    return open(outpath, 'rb').read()
keys = sys.argv[1:]
for key in keys:
    size = int(json.loads(s3api('head-object', '--bucket', 'z7yvc40ykx', '--key', key).stdout)['ContentLength'])
    tail8 = fetch_range(key, 'bytes=-8', os.path.join(S, 'tail8.bin'))
    flen = struct.unpack('<I', tail8[:4])[0]
    assert tail8[4:] == b'PAR1', 'not parquet'
    footer = fetch_range(key, f'bytes={-(flen+8)}', os.path.join(S, 'footer.bin'))
    stub = os.path.join(S, 'stub.parquet')
    with open(stub, 'wb') as fh:
        fh.truncate(size)
        fh.seek(0); fh.write(b'PAR1')
        fh.seek(size - (flen + 8)); fh.write(footer)
    md = pq.read_metadata(stub)
    sch = md.schema.to_arrow_schema()
    print(f'== {key}  bytes={size}  rows={md.num_rows}  row_groups={md.num_row_groups}  ncols={len(sch)}')
    print('   columns:', ', '.join(f'{f.name}:{f.type}' for f in sch)[:3000])
    os.remove(stub)
