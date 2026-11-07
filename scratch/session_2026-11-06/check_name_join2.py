import re, sys, glob, collections
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
exec(open("check_name_link.py", encoding="utf-8").read().split("# 2. player name tables")[0])
B = r"G:\pureskill\csds-cs2\tomes\combined_failed_22_25_20261027"
tkeys = set(targets.keys())
fs = sorted(glob.glob(B + r"\tome\csds\basic_player_personal.*\dataframe_*"))
pn = collections.defaultdict(set)
rows_hit = 0
for f in fs:
    t = pq.read_table(f, columns=["match_key", "name", "clan_tag"])
    for mk, nm, ct in zip(t.column("match_key").to_pylist(), t.column("name").to_pylist(), t.column("clan_tag").to_pylist()):
        if mk in tkeys:
            rows_hit += 1
            for x in (nm, ct):
                if x and len(x.strip()) >= 2:
                    pn[mk].add(x.strip().lower())
print("personal rows for the 14 targets:", rows_hit, "| targets with any name/clan:", len(pn))
exact = sub = names_hit = 0
per_name_token_counts = collections.Counter()
for k, (s, _) in targets.items():
    toks = set(x.lower() for x in re.findall(r"[A-Za-z]{3,}", s))
    names = pn.get(k, set())
    e = toks & names
    sb = {x for x in toks if any(x in n for n in names if len(x) >= 3)}
    exact += len(e); sub += len(sb)
    if e or sb: names_hit += 1
    per_name_token_counts[len(toks)] += 1
print("alphabetic tokens per name:", dict(per_name_token_counts))
print("exact token == player name or clan tag, hits:", exact)
print("token contained in a player name or clan tag, hits:", sub)
print("server names with any hit:", names_hit, "of", len(targets))
