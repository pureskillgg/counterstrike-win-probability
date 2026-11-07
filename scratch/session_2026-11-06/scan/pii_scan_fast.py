#!/usr/bin/env python3
"""Count-only pattern scanner for PII / secret / linkage markers.

Modes:
  s3   <listfile>   listfile lines: "<size> <key>" ; streams each object from S3 (no local copy)
  local <path>...   walks local paths (skips .git and any .env file)

Never prints matched secret values. Identifier-type matches print only a masked
prefix (first 3 chars + '***'). Parquet / parquet.gz objects also report their column
names (schema only, no values).
"""
import sys, os, re, subprocess, gzip, io, shutil, json
from collections import Counter, defaultdict

BUCKET = 'z7yvc40ykx'
EP = 'https://s3api-eu-ro-1.runpod.io'
REGION = 'eu-ro-1'
AWS = r'C:\Program Files\Amazon\AWSCLIV2\aws.exe'
if not os.path.exists(AWS):
    AWS = 'aws'
MAX_BYTES = 1_300_000_000  # streamed; gz objects are decompressed on the fly
CHUNK = 16 * 1024 * 1024
OVERLAP = 2048

# name, regex(bytes), category, show_masked_example
PATTERNS = [
    ('steam64', rb'(?<!\d)7656119\d{10}(?!\d)', 'id', True),
    ('steam2', rb'STEAM_[0-5]:[01]:\d{4,}', 'id', True),
    ('steam3', rb'\[U:1:\d{4,}\]', 'id', True),
    ('steam_url', rb'steamcommunity\.com/(?:id|profiles)/[^\s"\'<>)]+', 'id', False),
    ('email', rb'[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}', 'pii', True),
    ('win_user_path', rb'[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b|Default\b|All Users\b)[^\\/\s"\'<>:|*?]+', 'pii', True),
    ('posix_home', rb'/(?:home|Users)/[A-Za-z0-9._-]{2,}', 'pii', True),
    ('ipv4', rb'(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])', 'ipv4', False),
    ('url', rb'https?://[^\s"\'<>)\]]+', 'url', False),
    ('sharecode', rb'CSGO-[A-Za-z0-9]{5}(?:-[A-Za-z0-9]{5}){4}', 'pii', False),
    ('match_key', rb'csds/\d{4}/\d{2}/\d{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/csds', 'linkage', False),
    ('oct26_31_key', rb'csds/2026/10/(?:2[6-9]|3[01])/', 'holdout_rule', False),
    ('name_field', rb'(?i)["\']?(?:player_?name|nick(?:name)?|steam_?name|client_name|server_name|display_?name|real_?name|user_?name|account_?name|clan_?tag)["\']?\s*[:=]', 'name', False),
    ('chat_field', rb'(?i)player_?chat|chat_?text|say_?text', 'chat', False),
    ('secret_aws_akid', rb'AKIA[0-9A-Z]{16}', 'secret', False),
    ('secret_runpod', rb'rpa_[A-Za-z0-9]{20,}', 'secret', False),
    ('secret_github', rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}', 'secret', False),
    ('secret_hf', rb'hf_[A-Za-z0-9]{30,}', 'secret', False),
    ('secret_privkey', rb'-----BEGIN [A-Z ]*PRIVATE KEY-----', 'secret', False),
    ('secret_kv', rb'(?i)(?:aws_secret_access_key|aws_access_key_id|api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)["\']?\s*[:=]\s*["\']?[^\s"\',;]{6,}', 'secret', False),
    ('url_userinfo', rb'[A-Za-z]+://[^\s/@:]+:[^\s/@]+@', 'secret', False),
]
COMPILED = [(n, re.compile(p), c, s) for n, p, c, s in PATTERNS]
REPORT_CATS = {'id', 'pii', 'name', 'chat', 'secret', 'linkage', 'holdout_rule', 'ipv4', 'url'}

TEXT_EXT = {'.log', '.json', '.csv', '.txt', '.md', '.py', '.sh', '.toml', '.yaml', '.yml', '.ini', '.cfg',
            '.js', '.ipynb', '.html', '.xml', '.stackdump', '.pyc', '.lock', '.jsonl', '.docx', '.bat', '.ps1', '.r', '.ts', '.tsx', '.cpp', '.h', '.c'}
BIN_EXT = {'.pt', '.npy', '.npz', '.parquet', '.gz', '.tar', '.zip', '.rar', '.pkl', '.joblib', '.ubj', '.bin', '.png', '.svg', '.pdf', '.safetensors', '.onnx'}


def mask(b):
    s = b.decode('utf-8', 'replace')
    return s[:3] + '***(len %d)' % len(s)


class Acc:
    def __init__(self):
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.bytes = 0
        self.lines = 0

    def feed(self, window, tail_len):
        """Scan window; count matches that end after the first tail_len bytes (already-scanned overlap)."""
        for name, rx, cat, show in COMPILED:
            for m in rx.finditer(window):
                if m.end() <= tail_len:
                    continue
                self.counts[name] += 1
                if show and len(self.examples[name]) < 2:
                    self.examples[name].append(mask(m.group(0)))


def scan_stream(fh, acc):
    tail = b''
    while True:
        chunk = fh.read(CHUNK)
        if not chunk:
            break
        acc.bytes += len(chunk)
        acc.lines += chunk.count(b'\n')
        window = tail + chunk
        acc.feed(window, len(tail))
        tail = window[-OVERLAP:]


def parquet_schema_from_tail(tail_bytes):
    """Return column names from a parquet footer given the last bytes of the file."""
    import pyarrow.parquet as pq
    if len(tail_bytes) < 8 or tail_bytes[-4:] != b'PAR1':
        return None
    flen = int.from_bytes(tail_bytes[-8:-4], 'little')
    if len(tail_bytes) < flen + 8:
        return 'footer-larger-than-tail:%d' % flen
    buf = b'PAR1' + tail_bytes[-(flen + 8):]
    try:
        sch = pq.read_schema(io.BytesIO(buf))
        return sch.names
    except Exception as e:
        return 'schema-error:%s' % type(e).__name__


def report(label, size, acc, extra=None):
    hits = {k: v for k, v in acc.counts.items() if v}
    cats = {}
    for name, rx, cat, show in COMPILED:
        if name in hits:
            cats.setdefault(cat, {})[name] = hits[name]
    interesting = {c: v for c, v in cats.items() if c in REPORT_CATS}
    out = {'obj': label, 'size': size, 'scanned_bytes': acc.bytes, 'lines': acc.lines, 'hits': cats}
    ex = {k: v for k, v in acc.examples.items() if v}
    if ex:
        out['masked_examples'] = ex
    if extra is not None:
        out['extra'] = extra
    flag = 'HIT' if any(c in ('id', 'pii', 'name', 'chat', 'secret', 'linkage') for c in cats) else 'clean'
    print(json.dumps(out, default=str), flush=True)
    return flag


def s3_list(listfile):
    rows = []
    for line in open(listfile, encoding='utf-8', errors='replace'):
        line = line.rstrip('\n')
        # list format written by the caller: "<size> <key>"
        head, _, key = line.partition(' ')
        try:
            size = int(head)
        except ValueError:
            continue
        key = key.strip()
        if key:
            rows.append((size, key))
    return rows


def s3_scan(size, key):
    label = 's3://%s/%s' % (BUCKET, key)
    if size > MAX_BYTES:
        print(json.dumps({'obj': label, 'size': size, 'skipped': 'over-size-limit'}), flush=True)
        return
    low = key.lower()
    cmd = [AWS, '--profile', 'runpod', '--endpoint-url', EP, '--region', REGION, 's3', 'cp', 's3://%s/%s' % (BUCKET, key), '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    acc = Acc()
    extra = None
    try:
        if low.endswith('.gz') and not low.endswith('.tar.gz'):
            base = low[:-3]
            with gzip.GzipFile(fileobj=proc.stdout) as gz:
                if base.endswith('.parquet'):
                    tail = bytearray()
                    tail_ = b''
                    while True:
                        chunk = gz.read(CHUNK)
                        if not chunk:
                            break
                        acc.bytes += len(chunk)
                        acc.lines += chunk.count(b'\n')
                        window = tail_ + chunk
                        acc.feed(window, len(tail_))
                        tail_ = window[-OVERLAP:]
                        tail.extend(chunk)
                        if len(tail) > 64 * 1024 * 1024:
                            del tail[:len(tail) - 16 * 1024 * 1024]
                    extra = {'parquet_columns': parquet_schema_from_tail(bytes(tail[-(16 * 1024 * 1024):]))}
                else:
                    scan_stream(gz, acc)
        else:
            if low.endswith('.parquet') or low.endswith('.npy') or low.endswith('.npz'):
                # binary numeric: raw pattern scan plus footer schema for parquet
                data = proc.stdout.read()
                acc.bytes = len(data)
                acc.feed(data, 0)
                if low.endswith('.parquet'):
                    extra = {'parquet_columns': parquet_schema_from_tail(data[-(16 * 1024 * 1024):])}
            else:
                scan_stream(proc.stdout, acc)
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read().decode('utf-8', 'replace')[:200]
            print(json.dumps({'obj': label, 'error': 'aws rc %d' % rc, 'err': err}), flush=True)
            return
    except Exception as e:
        print(json.dumps({'obj': label, 'error': type(e).__name__, 'msg': str(e)[:200]}), flush=True)
        try:
            proc.kill()
        except Exception:
            pass
        return
    report(label, size, acc, extra)


def local_scan(path, size):
    acc = Acc()
    extra = None
    low = path.lower()
    try:
        if low.endswith('.gz') and not low.endswith('.tar.gz'):
            if size > 50 * 1024 * 1024:
                print(json.dumps({'obj': path, 'size': size, 'skipped': 'large-gz-name-only'}), flush=True)
                return
            with gzip.open(path, 'rb') as fh:
                scan_stream(fh, acc)
        elif low.endswith('.parquet'):
            # footer-only: schema names; values were covered by the audit's value screen
            with open(path, 'rb') as fh:
                fh.seek(max(0, size - 16 * 1024 * 1024))
                tailb = fh.read()
            acc.bytes = len(tailb)
            extra = {'parquet_columns': parquet_schema_from_tail(tailb), 'mode': 'footer-only'}
        else:
            with open(path, 'rb') as fh:
                scan_stream(fh, acc)
    except Exception as e:
        print(json.dumps({'obj': path, 'error': type(e).__name__, 'msg': str(e)[:200]}), flush=True)
        return
    report(path, size, acc, extra)


def walk_local(roots, max_size):
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d != '.git' and d != '__pycache__' and d != 'envs' and d != '.pixi']
            for f in fn:
                if f.lower().startswith('.env'):
                    continue
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size > max_size:
                    print(json.dumps({'obj': p, 'size': st.st_size, 'skipped': 'over-size-limit'}), flush=True)
                    continue
                ext = os.path.splitext(f)[1].lower()
                low = f.lower()
                # data-named frozen / holdout / Oct 26-31 files: names only, never read
                if re.search(r'(?i)(frozen|holdout|oct(ober)?_?2[6-9]|oct_?3[01]|10_?2[6-9]|10_?3[01])', f) and ext not in ('.py', '.sh', '.md'):
                    print(json.dumps({'obj': p, 'size': st.st_size, 'skipped': 'holdout-or-frozen-named data file (name only)'}), flush=True)
                    continue
                if low.endswith('.tar') or low.endswith('.tar.gz') or low.endswith('.tgz') or ext in ('.npy', '.npz', '.zip', '.rar', '.7z', '.docx', '.xlsx', '.pptx', '.pdf'):
                    print(json.dumps({'obj': p, 'size': st.st_size, 'skipped': 'binary-archive-not-raw-scanned'}), flush=True)
                    continue
                if ext in BIN_EXT and ext not in ('.gz', '.parquet', '.pt', '.pkl', '.joblib', '.ubj', '.safetensors', '.bin'):
                    continue
                local_scan(p, st.st_size)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 's3':
        listfile = sys.argv[2]
        only = sys.argv[3] if len(sys.argv) > 3 else None
        for size, key in s3_list(listfile):
            if only and not re.search(only, key):
                continue
            s3_scan(size, key)
    elif mode == 'local':
        max_size = int(sys.argv[2])
        walk_local(sys.argv[3:], max_size)
    elif mode == 'one':
        size = int(sys.argv[2])
        s3_scan(size, sys.argv[3])
    print('DONE', mode, flush=True)
