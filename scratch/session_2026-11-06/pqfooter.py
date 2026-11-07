"""Read-only Parquet footer inspector (stdlib only). Prints schema, row counts, codecs.
Reads only the file tail (footer). Never reads data pages."""
import struct, sys, json

CODEC = {0: 'UNCOMPRESSED', 1: 'SNAPPY', 2: 'GZIP', 3: 'LZO', 4: 'BROTLI', 5: 'LZ4', 6: 'ZSTD', 7: 'LZ4_RAW'}
PTYPE = {0: 'BOOLEAN', 1: 'INT32', 2: 'INT64', 3: 'INT96', 4: 'FLOAT', 5: 'DOUBLE', 6: 'BYTE_ARRAY', 7: 'FIXED_LEN_BYTE_ARRAY'}
REP = {0: 'REQUIRED', 1: 'OPTIONAL', 2: 'REPEATED'}


def varint(buf, pos):
    r = 0
    s = 0
    while True:
        b = buf[pos]
        pos += 1
        r |= (b & 0x7F) << s
        if not (b & 0x80):
            return r, pos
        s += 7


def zz(n):
    return (n >> 1) ^ -(n & 1)


class TR:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def byte(self):
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def vi(self):
        r, self.pos = varint(self.buf, self.pos)
        return r

    def binary(self):
        n = self.vi()
        v = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return v

    def struct(self):
        out = {}
        last = 0
        while True:
            h = self.byte()
            if h == 0:
                break
            delta = h >> 4
            t = h & 0x0F
            fid = zz(self.vi()) if delta == 0 else last + delta
            last = fid
            if t == 1:
                out[fid] = True
            elif t == 2:
                out[fid] = False
            else:
                out[fid] = self.value(t)
        return out

    def value(self, t):
        if t == 3:
            b = self.byte()
            return b - 256 if b > 127 else b
        if t in (4, 5, 6):
            return zz(self.vi())
        if t == 7:
            v = struct.unpack('<d', self.buf[self.pos:self.pos + 8])[0]
            self.pos += 8
            return v
        if t == 8:
            return self.binary()
        if t in (9, 10):
            h = self.byte()
            size = h >> 4
            et = h & 0x0F
            if size == 15:
                size = self.vi()
            if et in (1, 2):
                return [self.byte() == 1 for _ in range(size)]
            return [self.value(et) for _ in range(size)]
        if t == 11:
            size = self.vi()
            if size == 0:
                return {}
            kv = self.byte()
            kt, vt = kv >> 4, kv & 0x0F
            return {self.value(kt): self.value(vt) for _ in range(size)}
        if t == 12:
            return self.struct()
        raise ValueError('unknown thrift type %s at %s' % (t, self.pos))


def footer(path):
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(size - 8)
        tail = f.read(8)
        assert tail[4:] == b'PAR1', 'not a parquet file'
        mlen = struct.unpack('<I', tail[:4])[0]
        f.seek(size - 8 - mlen)
        meta = f.read(mlen)
    return TR(meta).struct(), size


def s(v):
    if isinstance(v, bytes):
        return v.decode('utf-8', 'replace')
    return v


def inspect(path):
    md, size = footer(path)
    schema = md[2]
    root = schema[0]
    leaves = schema[1:]
    cols = []
    for el in leaves:
        cols.append({
            'name': s(el.get(4)),
            'ptype': PTYPE.get(el.get(1), el.get(1)),
            'rep': REP.get(el.get(3), el.get(3)),
            'converted': el.get(6),
            'num_children': el.get(5),
        })
    rgs = md.get(4, [])
    rg_rows = [rg.get(3) for rg in rgs]
    codecs = {}
    chunk_info = {}
    if rgs:
        for ch in rgs[0].get(1, []):
            m = ch.get(3, {})
            path_name = '.'.join(s(p) for p in m.get(3, []))
            codecs[path_name] = CODEC.get(m.get(4), m.get(4))
            chunk_info[path_name] = {'codec': CODEC.get(m.get(4), m.get(4)), 'enc': m.get(2)}
    kv = {}
    for item in md.get(5, []) or []:
        kv[s(item.get(1))] = s(item.get(2))
    out = {
        'path': path,
        'file_bytes': size,
        'num_rows': md.get(3),
        'row_groups': len(rgs),
        'row_group_rows_sample': rg_rows[:5],
        'n_columns': len(cols),
        'columns': cols,
        'codecs_rg0': codecs,
        'created_by': s(md.get(6)),
        'kv_keys': list(kv.keys()),
    }
    if 'pandas' in kv:
        try:
            out['pandas_meta_columns'] = [c.get('name') for c in json.loads(kv['pandas'])['columns']][:200]
        except Exception as e:  # noqa
            out['pandas_meta_err'] = str(e)
    return out


if __name__ == '__main__':
    for p in sys.argv[1:]:
        r = inspect(p)
        print(json.dumps(r, default=str, indent=1))
        print('=' * 60)
