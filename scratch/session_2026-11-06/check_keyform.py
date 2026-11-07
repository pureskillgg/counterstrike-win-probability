import re, sys, glob, collections
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
exec(open("check_name_link.py", encoding="utf-8").read().split("# 2. player name tables")[0])
B = r"G:\pureskill\csds-cs2\tomes\combined_failed_22_25_20261027"
def form(k):
    if k is None: return "None"
    if re.fullmatch(r"csds/\d{4}/\d{2}/\d{2}/[0-9a-f\-]{36}/csds", k): return "csds_path"
    if re.fullmatch(r"[0-9a-f\-]{36}", k): return "uuid"
    return "other_len%d" % len(k)
tk_uuid = set()
for k in targets:
    m = re.search(r"([0-9a-f\-]{36})", k)
    if m: tk_uuid.add(m.group(1))
print("target key forms:", collections.Counter(form(k) for k in targets))
print("batch dir exists:", __import__("os").path.isdir(B))
fs = sorted(glob.glob(B + r"\tome\csds\basic_player_name.*\dataframe_*"))
print("player_name files in batch:", len(fs))
c = collections.Counter(); hit = 0
for f in fs:
    t = pq.read_table(f, columns=["match_key"])
    for mk in t.column("match_key").to_pylist():
        c[form(mk)] += 1
        m = re.search(r"([0-9a-f\-]{36})", mk or "")
        if m and m.group(1) in tk_uuid: hit += 1
print("name-table match_key forms:", dict(c))
print("name-table rows whose uuid matches a target:", hit)
