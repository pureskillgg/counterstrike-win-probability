import sys, os, subprocess, json
import pyarrow.parquet as pq
BUCKET = "z7yvc40ykx"
COMMON = ["--profile", "runpod", "--endpoint-url", "https://s3api-eu-ro-1.runpod.io", "--region", "eu-ro-1"]
OUT = sys.argv[1]
keys = sys.argv[2:]
def get_range(key, rng, outpath):
    cmd = ["aws", "s3api", "get-object", "--bucket", BUCKET, "--key", key, "--range", rng, outpath] + COMMON
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("get-object failed for range %s: rc=%d" % (rng, r.returncode))
for key in keys:
    name = key.replace("/", "__")
    try:
        head = subprocess.run(["aws", "s3api", "head-object", "--bucket", BUCKET, "--key", key] + COMMON, capture_output=True, text=True)
        size = json.loads(head.stdout)["ContentLength"]
        tail = os.path.join(OUT, name + ".tail8")
        get_range(key, "bytes=-8", tail)
        b8 = open(tail, "rb").read()
        if b8[4:] != b"PAR1":
            print("NOT PARQUET (no PAR1 footer marker):", key)
            continue
        flen = int.from_bytes(b8[:4], "little")
        fpath = os.path.join(OUT, name + ".footer")
        get_range(key, "bytes=-%d" % (flen + 8), fpath)
        foot = open(fpath, "rb").read()
        sparse = os.path.join(OUT, name + ".sparse")
        with open(sparse, "wb") as f:
            f.truncate(size)
            f.seek(size - len(foot))
            f.write(foot)
        md = pq.read_metadata(sparse)
        sch = pq.read_schema(sparse)
        print("=== %s | size=%d | rows=%d | row_groups=%d | cols=%d" % (key, size, md.num_rows, md.num_row_groups, md.num_columns))
        print("   columns:", ", ".join("%s:%s" % (f.name, f.type) for f in sch))
    except Exception as ex:
        print("ERROR", key, type(ex).__name__, str(ex)[:200])
