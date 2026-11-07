import re, sys, os

ROOT = "C:/Users/13144/Documents/pureskill/counterstrike-win-probability"
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
files = [l.strip() for l in open(S + "/include_list.txt") if l.strip()]
EXT = r"\.(json|md|csv|parquet|npz|npy|txt|pkl|joblib|ubj|json\.gz|png|pt|pth|xlsx|tsv)\b"
pat = re.compile(r"""(["'])((?:[A-Za-z]:)?[^"'\n]{0,200}?""" + EXT + r""")\1""")
pat_dir = re.compile(r"""(["'])((?:data|notes|docs|models|results|out|outputs|reports|runs|preds|experiments)[/\\][^"'\n]{0,160})\1""")
for f in files:
    src = open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
    hits = []
    for m in pat.finditer(src):
        s = m.group(2)
        if len(s) > 3 and not s.startswith("http"):
            hits.append(s)
    for m in pat_dir.finditer(src):
        hits.append(m.group(2))
    seen = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    print(f"{f}\t" + " ; ".join(seen[:5]))
