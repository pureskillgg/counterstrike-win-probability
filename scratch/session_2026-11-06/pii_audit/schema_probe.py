import io, subprocess, json, os, sys, uuid
import pyarrow.parquet as pq

BUCKET = "z7yvc40ykx"
EP = "https://s3api-eu-ro-1.runpod.io"
REG = "eu-ro-1"
PROF = "runpod"
TMPDIR = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/pii_rng"
BASE = ["--profile", PROF, "--endpoint-url", EP, "--region", REG]

def head_size(key):
    r = subprocess.run(["aws", "s3api", "head-object", "--bucket", BUCKET, "--key", key] + BASE,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("head failed for " + key)
    return json.loads(r.stdout)["ContentLength"]

def s3_range(key, start, end_incl):
    out = os.path.join(TMPDIR, uuid.uuid4().hex + ".bin")
    cmd = ["aws", "s3api", "get-object", "--bucket", BUCKET, "--key", key,
           "--range", f"bytes={start}-{end_incl}", out] + BASE
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("range read failed for " + key)
    with open(out, "rb") as f:
        data = f.read()
    os.remove(out)
    return data

class RangeIO(io.RawIOBase):
    def __init__(self, key, size):
        self.key = key
        self.size = size
        self.pos = 0
    def readable(self):
        return True
    def seekable(self):
        return True
    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos
    def tell(self):
        return self.pos
    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        data = s3_range(self.key, self.pos, end)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

def schema_of(key):
    size = head_size(key)
    f = RangeIO(key, size)
    pf = pq.ParquetFile(f)
    md = pf.metadata
    sch = pf.schema_arrow
    print("KEY", key, "bytes", size, "rows", md.num_rows, "row_groups", md.num_row_groups)
    for fld in sch:
        print("   col:", fld.name, "|", str(fld.type))

if __name__ == "__main__":
    for k in sys.argv[1:]:
        try:
            schema_of(k)
        except Exception as e:
            print("ERR", k, type(e).__name__, str(e)[:200])
