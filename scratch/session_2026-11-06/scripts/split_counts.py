import sys, json, collections
p = sys.argv[1]
d = json.load(open(p, encoding='utf-8'))
print("top-level keys:", list(d.keys()))
a = d.get("assign", {})
print("assign entries:", len(a))
print("label counts:", dict(collections.Counter(a.values())))
# year-month-day buckets of assigned keys (dates only, no ids)
c = collections.Counter()
for k, v in a.items():
    parts = k.split('/')
    c[(v, parts[1] if len(parts)>1 else '?')] += 1
print("label by year:", dict(c))
oct = collections.Counter()
for k, v in a.items():
    parts = k.split('/')
    if len(parts)>3 and parts[1]=='2026' and parts[2]=='10':
        oct[(v, parts[3])] += 1
print("2026-10 by label/day:", dict(sorted(oct.items())))
for kk in d:
    if kk != 'assign':
        val = d[kk]
        print(kk, type(val).__name__, (len(val) if hasattr(val,'__len__') else val) if not isinstance(val,(dict,list)) or True else '')
