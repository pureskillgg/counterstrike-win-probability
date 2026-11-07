import io, subprocess, json, os, sys, uuid, collections
import pyarrow.parquet as pq

BUCKET = "z7yvc40ykx"
EP = "https://s3api-eu-ro-1.runpod.io"
REG = "eu-ro-1"
PROF = "runpod"
TMPDIR = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/pii_rng"
BASE = ["--profile", PROF, "--endpoint-url", EP, "--region", REG]
WIN = 262144
BS = chr(92)
os.makedirs(TMPDIR, exist_ok=True)


def s3_get(key, rng):
    out = os.path.join(TMPDIR, uuid.uuid4().hex + ".bin")
    cmd = ["aws", "s3api", "get-object", "--bucket", BUCKET, "--key", key, "--range", rng, out] + BASE
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("get failed")
    meta = json.loads(r.stdout) if r.stdout.strip() else {}
    with open(out, "rb") as f:
        data = f.read()
    os.remove(out)
    return data, meta


class TailIO(io.RawIOBase):
    def __init__(self, key, size, wstart, window):
        self.key, self.size, self.pos, self.wstart, self.window = key, size, 0, wstart, window

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
        end = min(self.pos + n, self.size)
        if self.pos >= self.wstart and end <= self.wstart + len(self.window):
            data = self.window[self.pos - self.wstart:end - self.wstart]
        else:
            data, _ = s3_get(self.key, "bytes=%d-%d" % (self.pos, end - 1))
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


MARKERS = [("http", "http"), ("win_users", BS + "users" + BS), ("unix_users", "/users/"), ("at_sign", "@"), ("steam", "steam")]


def scan(key):
    win, meta = s3_get(key, "bytes=-%d" % WIN)
    cr = meta.get("ContentRange")
    total = int(cr.split("/")[-1])
    start = total - len(win)
    pf = pq.ParquetFile(TailIO(key, total, start, win))
    md = pf.metadata
    sch = pf.schema_arrow
    cols = tuple((fld.name, str(fld.type)) for fld in sch)
    kv = sch.metadata or {}
    kvinfo = []
    for k, v in kv.items():
        kname = k.decode() if isinstance(k, bytes) else str(k)
        vs = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
        low = vs.lower()
        flags = tuple(name for name, s in MARKERS if s in low)
        kvinfo.append((kname, len(vs), flags))
    return dict(key=key, size=total, rows=md.num_rows, rg=md.num_row_groups, cols=cols, kv=tuple(kvinfo))


if __name__ == "__main__":
    keys = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    groups = collections.OrderedDict()
    for k in keys:
        try:
            r = scan(k)
        except Exception as e:
            print("ERR", k, type(e).__name__, str(e)[:120])
            continue
        groups.setdefault((r["cols"], r["kv"]), []).append(r)
    print("TOTAL_FILES_SCANNED", sum(len(v) for v in groups.values()), "DISTINCT_SCHEMAS", len(groups))
    for gi, ((cols, kv), items) in enumerate(groups.items()):
        dirs = collections.Counter("/".join(x["key"].split("/")[:-1]) for x in items)
        rows = [x["rows"] for x in items]
        print("\n=== SCHEMA %d: %d files | ncols %d | rows min %d max %d" % (gi, len(items), len(cols), min(rows), max(rows)))
        print("  folders:", dict(dirs))
        print("  files_example:", [x["key"].split("/")[-1] for x in items[:6]])
        print("  columns:", ", ".join("%s:%s" % (n, t) for n, t in cols))
        print("  kv:", [(a, b, c) for a, b, c in kv] if kv else "none")
