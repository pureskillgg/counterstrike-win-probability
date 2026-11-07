import subprocess, json, struct, io, sys
import pyarrow.parquet as pq

BUCKET = "z7yvc40ykx"
COMMON = ["--profile", "runpod", "--endpoint-url", "https://s3api-eu-ro-1.runpod.io", "--region", "eu-ro-1"]

def head_size(key):
    out = subprocess.run(["aws", *COMMON, "s3api", "head-object", "--bucket", BUCKET, "--key", key,
                          "--query", "ContentLength", "--output", "text"], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())

def get_range(key, start, end):
    tmp = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/match_key_audit/range.bin"
    subprocess.run(["aws", *COMMON, "s3api", "get-object", "--bucket", BUCKET, "--key", key,
                    "--range", f"bytes={start}-{end}", tmp], capture_output=True, check=True)
    with open(tmp, "rb") as f:
        return f.read()

class RangeFile(io.RawIOBase):
    def __init__(self, key):
        self.key = key
        self.size = head_size(key)
        self.pos = 0
        self.cache = {}
    def readable(self): return True
    def seekable(self): return True
    def seek(self, off, whence=0):
        if whence == 0: self.pos = off
        elif whence == 1: self.pos += off
        else: self.pos = self.size + off
        return self.pos
    def tell(self): return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        end = min(self.pos + n, self.size) - 1
        if end < self.pos: return b""
        data = get_range(self.key, self.pos, end)
        self.pos += len(data)
        return data
    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

for key in sys.argv[1:]:
    try:
        rf = RangeFile(key)
        md = pq.ParquetFile(rf).metadata
        sch = pq.ParquetFile(RangeFile(key)).schema_arrow
        print("==", key, "size_MiB=%.1f" % (rf.size / 2**20), "rows(metadata)=", md.num_rows, "row_groups=", md.num_row_groups)
        print("   cols:", [(f.name, str(f.type)) for f in sch])
    except Exception as e:
        print("==", key, "ERROR", type(e).__name__, str(e)[:200])
