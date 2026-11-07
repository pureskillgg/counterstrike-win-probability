import json, sys, re, glob, os, collections
import pyarrow.parquet as pq
M = json.load(open(sys.argv[1], encoding="utf-8"))["assign"]
root = sys.argv[2]
files = sorted(glob.glob(os.path.join(root, "dataframe_*")))
cols = ["match_key","round","player_id","player_controller_id","player_id_pawn","name","clan_tag","steam_id","is_bot","player_id_fixed"]
t = pq.read_table(files[0], columns=cols)
for f in files[1:]:
    t = __import__("pyarrow").concat_tables([t, pq.read_table(f, columns=cols)])
n_all = t.num_rows
keys = t.column("match_key").to_pylist()
def date_of(k):
    m = re.match(r"csds/(\d{4})/(\d{2})/(\d{2})/", k or "")
    return (m.group(1), m.group(2), m.group(3)) if m else None
# Gate: keep only rows whose match is assigned train or val and is not dated 2026-10-26..31
keep = []
gate = collections.Counter()
for i, k in enumerate(keys):
    lab = M.get(k)
    d = date_of(k)
    if d and d[0] == "2026" and d[1] == "10" and d[2] in ("26","27","28","29","30","31"):
        gate["excluded_oct26_31"] += 1; continue
    if lab is None:
        gate["excluded_not_in_assign"] += 1; continue
    if lab not in ("train","val"):
        gate["excluded_label_" + lab] += 1; continue
    gate["kept_" + lab] += 1
    keep.append(i)
print("ROWS_TOTAL", n_all, "GATE", dict(gate))
sub = t.take(keep)
def col(name): return sub.column(name).to_pylist()
def cls_str(v):
    if v is None: return "NULL"
    if v == "": return "EMPTY"
    if re.fullmatch(r"[a-z]{8}", v): return "8_lower"
    if re.fullmatch(r"[A-Z]", v): return "one_upper_letter"
    if re.fullmatch(r"7656119\d{10}", v): return "STEAMID64_pattern"
    if v.startswith("STEAM_") or v.startswith("[U:1:"): return "STEAM_or_U_pattern"
    if re.fullmatch(r"\d+", v): return "all_digits_len%d" % len(v)
    return "other_len%d" % min(len(v), 20)
for c in ["name","clan_tag","steam_id"]:
    vals = col(c)
    cc = collections.Counter(cls_str(v) for v in vals)
    print("COL", c, "distinct", len(set(vals)), "classes", dict(cc))
for c in ["player_id","player_controller_id","player_id_pawn","player_id_fixed","round"]:
    vals = col(c)
    dl = collections.Counter(("NULL" if v is None else "len%d" % len(str(v))) for v in vals)
    sid64 = sum(1 for v in vals if v is not None and 76561190000000000 <= v <= 76561210000000000)
    print("NUM", c, "distinct", len(set(vals)), "digit_len", dict(dl), "steamid64_range_hits", sid64)
print("IS_BOT", dict(collections.Counter(col("is_bot"))))
# linkage structure within the kept subset (counts only)
mk = col("match_key"); pid = col("player_id"); st = col("steam_id"); fx = col("player_id_fixed")
per_match_players = collections.defaultdict(set); per_match_letters = collections.defaultdict(set)
pid_letter = collections.defaultdict(set)
for a,b,c,d in zip(mk,pid,st,fx):
    per_match_players[a].add(b); per_match_letters[a].add(c)
    pid_letter[(a,b)].add(c)
print("MATCHES_KEPT", len(per_match_players))
print("PLAYERS_PER_MATCH_hist", dict(collections.Counter(len(v) for v in per_match_players.values())))
print("LETTERS_PER_MATCH_hist", dict(collections.Counter(len(v) for v in per_match_letters.values())))
print("LETTERS_PER_PLAYER_KEY_hist", dict(collections.Counter(len(v) for v in pid_letter.values())))
print("PLAYER_KEYS_(match,player_id)", len(pid_letter))
print("PLAYER_ID_EQ_FIXED_rows", sum(1 for b,f in zip(pid,fx) if b==f), "of", len(pid))
