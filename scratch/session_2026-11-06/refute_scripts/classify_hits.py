import os, re, sys, collections
T = sys.argv[1]
users = collections.Counter(); ips = []; steam_locs = []
GENERIC = {"root","runpod","ubuntu","user","users","workspace","ec2-user","vscode","node","app","work","home"}
for root, _, files in os.walk(T):
    for fn in files:
        p = os.path.join(root, fn)
        s = open(p, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(p, T).replace(os.sep, "/")
        for m in re.finditer(r"/(?:home|Users)/([^/\s\"']+)", s):
            users[m.group(1).lower()] += 1
        for m in re.finditer(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b", s):
            ips.append((rel, m.group(0)))
        for m in re.finditer(r"7656119\d{10}", s):
            steam_locs.append((rel, m.start()))
print("unix home-path usernames: distinct=%d total_hits=%d" % (len(users), sum(users.values())))
print("  generic/system names among them:", sorted(u for u in users if u in GENERIC))
print("  non-generic distinct count:", len([u for u in users if u not in GENERIC]))
print("  username length histogram:", dict(collections.Counter(len(u) for u in users)))
cats = collections.Counter()
for rel, ip in ips:
    o = [int(x) for x in ip.split(".")]
    if o[0] in (0, 127, 255) or ip in ("8.8.8.8", "1.1.1.1"): cat = "reserved/public-dns/placeholder"
    elif o[0] == 10 or (o[0] == 192 and o[1] == 168) or (o[0] == 172 and 16 <= o[1] <= 31): cat = "private"
    elif o[0] in (1,2,3,4,5,6,7,8,9) or re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip) and o[3] in (0,1,255): cat = "looks-like-version-or-placeholder"
    else: cat = "other-public-looking"
    cats[cat] += 1
print("IP-like hits by class:", dict(cats), "| files:", sorted(set(r for r, _ in ips)))
print("steamid64-shaped literals: count=%d | files=%s | offsets=%s" % (len(steam_locs), sorted(set(r for r,_ in steam_locs)), [o for _,o in steam_locs]))
