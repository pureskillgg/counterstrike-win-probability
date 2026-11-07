import subprocess
import json
import sys
import io
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

BUCKET = "z7yvc40ykx"
COMMON = ["--profile", "runpod", "--endpoint-url", "https://s3api-eu-ro-1.runpod.io", "--region", "eu-ro-1"]
TMP = r"C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\range.bin"


def head_size(key):
    out = subprocess.run(["aws", "s3api", "head-object", "--bucket", BUCKET, "--key", key,
                          "--query", "ContentLength", "--output", "text"] + COMMON,
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def get_range(key, start, end):
    # inclusive byte range, read into a scratch file, returned as bytes
    subprocess.run(["aws", "s3api", "get-object", "--bucket", BUCKET, "--key", key,
                    "--range", "bytes=%d-%d" % (start, end), TMP] + COMMON,
                   capture_output=True, check=True)
    with open(TMP, "rb") as f:
        return f.read()


class S3RangeFile(io.RawIOBase):
    """Read-only seekable file over S3 range GETs (footer reads only)."""

    def __init__(self, key, size):
        self.key = key
        self.size = size
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = get_range(self.key, self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


KEYS = [
    "cs2data/combined/cache_snapshot_2026.parquet",
    "cs2data/combined/cache_snapshot_2026_ext.parquet",
    "cs2data/combined/cache_pos_2026_trainctr.parquet",
    "cs2data/combined/cache_pos_2026_allmatch_EXPLORATORY.parquet",
    "cs2data/combined/rounds.parquet",
]

PII_HINTS = ["client_name", "server_name", "name", "steam", "player", "clan", "ip", "email", "header", "sharecode", "demo_id", "match_id", "unique_steamids"]

for key in KEYS:
    size = head_size(key)
    f = S3RangeFile(key, size)
    pf = pq.ParquetFile(f)
    md = pf.metadata
    names = pf.schema_arrow.names
    print("==", key.split("/")[-1], "| bytes:", size, "| rows:", md.num_rows, "| cols:", len(names))
    hits = [n for n in names if any(h in n.lower() for h in PII_HINTS)]
    print("   PII-hint column names:", hits)
    print("   all column names:", names)
