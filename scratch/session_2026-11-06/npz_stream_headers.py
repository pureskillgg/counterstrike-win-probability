"""Read zip member names and npy headers from a gzip'd npz streamed on stdin.

Only headers are parsed; member data is discarded as it streams past.
Prints member name, dtype and shape. Never prints array values.
"""
import gzip
import io
import sys
import zlib

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
src = gzip.GzipFile(fileobj=sys.stdin.buffer, mode="rb")


def readn(n):
    out = b""
    while len(out) < n:
        chunk = src.read(n - len(out))
        if not chunk:
            break
        out += chunk
    return out


count = 0
while True:
    sig = readn(4)
    if sig != b"PK\x03\x04":
        break  # central directory or end of archive
    hdr = readn(26)
    (_ver, flags, method, _t, _d, _crc, comp, uncomp, nlen, xlen) = __import__("struct").unpack("<HHHHHIIIHH", hdr)
    name = readn(nlen).decode("utf-8", "replace")
    extra = readn(xlen)
    if comp == 0xFFFFFFFF or uncomp == 0xFFFFFFFF:
        # zip64 extra field (id 0x0001): uncompressed then compressed, each only if the header holds the sentinel
        p = 0
        while p + 4 <= len(extra):
            fid, fsz = __import__("struct").unpack("<HH", extra[p:p + 4])
            if fid == 0x0001:
                body = extra[p + 4:p + 4 + fsz]
                q = 0
                if uncomp == 0xFFFFFFFF:
                    uncomp = __import__("struct").unpack("<Q", body[q:q + 8])[0]
                    q += 8
                if comp == 0xFFFFFFFF:
                    comp = __import__("struct").unpack("<Q", body[q:q + 8])[0]
                break
            p += 4 + fsz
    if flags & 0x08:
        print(f"MEMBER {name}: data descriptor used, sizes not in header; stopping")
        break
    # read enough of the member to parse the npy header
    head_raw = readn(min(comp, 4096))
    if method == 8:
        head = zlib.decompressobj(-zlib.MAX_WBITS).decompress(head_raw, 4096)
    else:
        head = head_raw
    try:
        buf = io.BytesIO(head)
        ver = np.lib.format.read_magic(buf)
        if ver == (1, 0):
            shape, _f, dtype = np.lib.format.read_array_header_1_0(buf)
        else:
            shape, _f, dtype = np.lib.format.read_array_header_2_0(buf)
        print(f"MEMBER {name}: dtype={dtype} shape={shape} compressed_bytes={comp:,} uncompressed_bytes={uncomp:,}")
    except Exception as e:
        print(f"MEMBER {name}: header not parsed ({type(e).__name__}) compressed_bytes={comp:,}")
    # discard the rest of this member
    remaining = comp - len(head_raw)
    while remaining > 0:
        chunk = src.read(min(remaining, 1 << 24))
        if not chunk:
            break
        remaining -= len(chunk)
    count += 1

print(f"members seen: {count}")
