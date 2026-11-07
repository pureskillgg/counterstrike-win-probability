"""Schema-only inspection of objects in the runpod S3 bucket.

Reads byte ranges only (parquet footers, npz/npy headers). No row data is read
and nothing is written except small scratch range blocks. Prints column names,
types and counts, never cell values.
"""
import io
import os
import sys
import subprocess
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

BUCKET = "z7yvc40ykx"
AWS = ["aws", "--profile", "runpod", "--endpoint-url", "https://s3api-eu-ro-1.runpod.io", "--region", "eu-ro-1"]
SCR = r"C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad"
BLK = 1 << 20
TMP = os.path.join(SCR, "s3_blk.bin")


def run(args):
    return subprocess.run(AWS + args, capture_output=True, text=True, check=True).stdout


def size_of(key):
    return int(run(["s3api", "head-object", "--bucket", BUCKET, "--key", key, "--query", "ContentLength", "--output", "text"]).strip())


class S3File(io.RawIOBase):
    def __init__(self, key):
        self.key = key
        self.size = size_of(key)
        self.pos = 0
        self.blocks = {}

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

    def _block(self, i):
        if i not in self.blocks:
            start = i * BLK
            end = min(start + BLK, self.size) - 1
            run(["s3api", "get-object", "--bucket", BUCKET, "--key", self.key, "--range", f"bytes={start}-{end}", TMP])
            with open(TMP, "rb") as fh:
                self.blocks[i] = fh.read()
        return self.blocks[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size)
        out = bytearray()
        p = self.pos
        while p < end:
            i = p // BLK
            b = self._block(i)
            off = p - i * BLK
            take = min(len(b) - off, end - p)
            out += b[off:off + take]
            p += take
        self.pos = p
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def parquet(key):
    f = S3File(key)
    pf = pq.ParquetFile(pa.PythonFile(f, mode="r"))
    md = pf.metadata
    print(f"PARQUET {key} bytes={f.size:,} rows={md.num_rows:,} row_groups={md.num_row_groups} cols={md.num_columns}")
    for fld in pf.schema_arrow:
        print(f"    {fld.name}: {fld.type}")
    kv = pf.schema_arrow.metadata or {}
    print("    metadata keys:", sorted(k.decode("utf-8", "replace") for k in kv.keys()))


def npz(key):
    f = S3File(key)
    zf = zipfile.ZipFile(f)
    print(f"NPZ {key} bytes={f.size:,} members={len(zf.infolist())}")
    for info in zf.infolist():
        with zf.open(info.filename) as m:
            ver = np.lib.format.read_magic(m)
            if ver == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(m)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(m)
        print(f"    {info.filename}: dtype={dtype} shape={shape}")


if __name__ == "__main__":
    mode = sys.argv[1]
    for key in sys.argv[2:]:
        try:
            if mode == "parquet":
                parquet(key)
            elif mode == "npz":
                npz(key)
        except Exception as e:  # report and continue
            print(f"ERROR {key}: {type(e).__name__}: {e}")
