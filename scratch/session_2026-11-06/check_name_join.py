import json, re, sys, glob, collections
import pyarrow.parquet as pq
import pyarrow.compute as pc
sys.stdout.reconfigure(encoding="utf-8")
exec(open(r"C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\check_name_link.py", encoding="utf-8").read().split("# 2. player name tables")[0])

tkeys = set(targets.keys())
B = sorted(set(v[1].split("\\tome\\csds")[0] for v in targets.values()))[0]
pn_files = sorted(glob.glob(B + r"\tome\csds\basic_player_name.*\dataframe_*"))
print("player_name files:", len(pn_files))
names_by_match = collections.defaultdict(set)
nrows = 0
for f in pn_files:
    t = pq.read_table(f, columns=["match_key", "name_new", "name_old"])
    mk = t.column("match_key").to_pylist()
    nn = t.column("name_new").to_pylist()
    no = t.column("name_old").to_pylist()
    for k, a, b in zip(mk, nn, no):
        if k in tkeys:
            nrows += 1
            for x in (a, b):
                if x and len(x) >= 3:
                    names_by_match[k].add(x.lower())
print("player-name rows for target matches:", nrows, "| matches with any name:", len(names_by_match))

# 3. server-name tokens vs player names of the same match (counts only)
exact_hits = 0
substr_hits = 0
names_with_hit = 0
tok_counts = collections.Counter()
for k, (s, _) in targets.items():
    toks = set(t_.lower() for t_ in re.findall(r"[A-Za-z]{3,}", s))
    pns = names_by_match.get(k, set())
    e = toks & pns
    sub = {t_ for t_ in toks if any(t_ in n for n in pns)}
    exact_hits += len(e)
    substr_hits += len(sub)
    if e or sub:
        names_with_hit += 1
    tok_counts[len(toks)] += 1
print("alphabetic tokens (3+) across the 14 names: distribution by count per name:", dict(tok_counts))
print("exact token == player-name hits:", exact_hits)
print("token substring-of-player-name hits:", substr_hits)
print("server names with any hit:", names_with_hit, "of", len(targets))
