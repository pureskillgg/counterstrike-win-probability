import sys, re, io
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
import pyarrow.parquet as pq
M = sys.argv[1]
targets = sys.argv[2:]
URL_UI = re.compile(rb'([A-Za-z]+)://([^\s/@:]+):([^\s/@]+)@([^\s"\'<>/]{0,80})')
EM = re.compile(rb'([A-Za-z0-9._%+-]{2,})@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})')
def cls(s):
    return ''.join('a' if c.isalpha() else ('d' if c.isdigit() else c) for c in s.decode('utf-8', 'replace'))
for t in targets:
    data = open(t, 'rb').read()
    name = t.replace(M, '').lstrip('/' + chr(92))
    for m in URL_UI.finditer(data):
        scheme, user, pwd, host = m.groups()
        print(name, '| URL-with-userinfo | scheme', scheme.decode(), '| user len', len(user), '| password len', len(pwd), '| host', host.decode('utf-8','replace')[:40], '| (user and password not printed)')
    emails = list(EM.finditer(data))
    if emails:
        sigs = [(len(m.group(1)), len(m.group(2)), 'dot' if b'.' in m.group(2) else 'nodot') for m in emails[:5]]
        print(name, '| email-shaped count', len(emails), '| local/domain lengths and dot presence (no content)', sigs)
    try:
        sch = pq.read_schema(t)
        print('   schema', [(f.name, str(f.type)) for f in sch][:40])
    except Exception as e:
        print('   schema error', type(e).__name__)
