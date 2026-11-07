import re, sys, collections
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1], columns=['match_key'])
vals = t.column('match_key').to_pylist()
n = len(vals)
print("rows read (match_key only):", n)
steam = re.compile(r'(?<!\d)7656119\d{10}(?!\d)')
email = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
ipv4 = re.compile(r'(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)')
winuser = re.compile(r'<user>')
def shape(s):
    s2 = re.sub(r'[0-9]', '9', s)
    s2 = re.sub(r'[A-Za-z]', 'a', s2)
    return re.sub(r'(9+)', '9+', re.sub(r'(a+)', 'a+', s2))
shapes = collections.Counter(shape(v) for v in vals if v is not None)
print("distinct shapes:", len(shapes))
for sh, c in shapes.most_common(5):
    print("  shape", repr(sh)[:80], "count", c)
print("null match_key:", sum(v is None for v in vals))
print("steamid-shaped rows:", sum(bool(steam.search(v)) for v in vals if v))
print("email-shaped rows:", sum(bool(email.search(v)) for v in vals if v))
print("ipv4-shaped rows:", sum(bool(ipv4.search(v)) for v in vals if v))
print("account-token rows:", sum(bool(winuser.search(v)) for v in vals if v))
print("mean key length:", round(sum(len(v) for v in vals if v)/max(1,n),1))
