# verify A1-2: for the claimed 10 matches, read the RAW E: parquet (read-only) and recompute the starter split.
# Also: resolve the one unresolved round of cbb071c2 with the engine per-team counter (score_update new_score).
# E: is only read. Nothing is written under E:.
import os, glob, collections
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
RAW_ROOT = r"E:\csds_final\csds\2026\10"
CLAIM10 = [
    "csds/2026/10/01/7368f3ae-7353-4f68-8d04-7c97f22ea2b8/csds",
    "csds/2026/10/01/ec8a424b-067f-4209-a42c-36d5a65cd97f/csds",
    "csds/2026/10/02/5429f41b-7721-4232-9784-bd9a3e2045a5/csds",
    "csds/2026/10/02/5f014ad2-63f8-415d-9952-5984b2924721/csds",
    "csds/2026/10/02/83d3ede2-5226-4035-8f32-87a32f083665/csds",
    "csds/2026/10/02/b2e50666-9b0b-4dcd-b8cc-59703e6a3064/csds",
    "csds/2026/10/02/d69b442a-8c83-4700-8a25-a035082e0ba8/csds",
    "csds/2026/10/02/e60396ec-5ff6-4a18-a2c9-b7187b650485/csds",
    "csds/2026/10/03/2093742d-edc0-4041-961c-a261f4f26876/csds",
    "csds/2026/10/03/cbb071c2-83a9-4167-b4c1-6fed1e145da0/csds",
]


def tome_channel(ch, keys):
    pages = sorted(glob.glob(os.path.join(TOME, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))
    cols_pool = {"round_end": ["round", "winner_team_code"], "player_info": ["round", "player_id", "team_code"],
                 "score_update": ["round", "team_code", "old_score", "new_score"], "player_death": ["round", "tick", "player_team_code", "attacker_team_code"]}
    cols = cols_pool[ch] + ["match_key"]
    return pl.concat([pl.scan_parquet(p).select(cols) for p in pages], how="diagonal_relaxed").filter(pl.col("match_key").is_in(keys)).collect()


def raw_channel(key, ch):
    # key = csds/2026/10/DD/<uuid>/csds
    parts = key.split("/")
    day, uid = parts[3], parts[4]
    p = os.path.join(RAW_ROOT, day, uid, ch)
    return pl.read_parquet(p)


def split_from_pi(pi, rounds):
    """pi: dict round -> dict pid->team; rounds: dict round -> (winner). Returns (A_wins, B_wins, unresolved, A_side_by_round)."""
    r1 = min(pi.keys())
    A = {p for p, tc in pi[r1].items() if tc == 2}
    B = {p for p, tc in pi[r1].items() if tc == 3}
    aw = bw = unres = 0
    sides = {}
    for r in sorted(rounds):
        w = rounds[r]
        pr = pi.get(r, {})
        sa = [pr[p] for p in A if p in pr]
        sb = [pr[p] for p in B if p in pr]
        if not sa or not sb:
            unres += 1
            continue
        sA = collections.Counter(sa).most_common(1)[0][0]
        sB = collections.Counter(sb).most_common(1)[0][0]
        sides[r] = sA
        aw += int(sA == w)
        bw += int(sB == w)
    return aw, bw, unres, sides, A, B


hdr_t = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=["key", "t_starters_score_final", "ct_starters_score_final", "max_rounds"]).filter(pl.col("key").is_in(CLAIM10))
hdr = {k: (t, c, m) for k, t, c, m in hdr_t.select(["key", "t_starters_score_final", "ct_starters_score_final", "max_rounds"]).iter_rows()}
te = tome_channel("round_end", CLAIM10)
tpi = tome_channel("player_info", CLAIM10)
tsu = tome_channel("score_update", CLAIM10)
print("tome loaded for 10 claimed keys: header", len(hdr), "round_end rows", te.height, "player_info rows", tpi.height)

for key in CLAIM10:
    t, c, m = hdr[key]
    # RAW
    rre = raw_channel(key, "basic_round_end") if False else None
    try:
        rh = pl.read_parquet(os.path.join(RAW_ROOT, key.split("/")[3], key.split("/")[4], "header"))
        raw_t = rh["t_starters_score_final"][0]
        raw_c = rh["ct_starters_score_final"][0]
        raw_mr = rh["max_rounds"][0]
    except Exception as e:
        raw_t = raw_c = raw_mr = f"ERR {type(e).__name__}"
    try:
        rre = pl.read_parquet(os.path.join(RAW_ROOT, key.split("/")[3], key.split("/")[4], "round_end"))
        rpi = pl.read_parquet(os.path.join(RAW_ROOT, key.split("/")[3], key.split("/")[4], "player_info"))
        raw_rounds = {r: w for r, w in rre.select(["round", "winner_team_code"]).iter_rows()}
        raw_pi = collections.defaultdict(dict)
        for r, p, tc in rpi.select(["round", "player_id", "team_code"]).iter_rows():
            raw_pi[r][p] = tc
        raw_aw, raw_bw, raw_unres, raw_sides, _, _ = split_from_pi(raw_pi, raw_rounds)
    except Exception as e:
        raw_aw = raw_bw = raw_unres = f"ERR {type(e).__name__}: {str(e)[:80]}"
        raw_rounds = {}
    # TOME
    tr = te.filter(pl.col("match_key") == key).select(["round", "winner_team_code"]).iter_rows()
    tome_rounds = {r: w for r, w in tr}
    tp = tpi.filter(pl.col("match_key") == key).select(["round", "player_id", "team_code"]).iter_rows()
    tome_pi = collections.defaultdict(dict)
    for r, p, tc in tp:
        tome_pi[r][p] = tc
    t_aw, t_bw, t_unres, t_sides, _, _ = split_from_pi(tome_pi, tome_rounds)
    same_rounds = raw_rounds == tome_rounds
    print(f"{key.split('/')[3]} {key.split('/')[4][:8]} mr={m} hdr(t,ct)=({t},{c}) | RAW hdr=({raw_t},{raw_c}) mr={raw_mr} | "
          f"RAW roster A(starterT)={raw_aw} B(starterCT)={raw_bw} unres={raw_unres} | TOME roster A={t_aw} B={t_bw} unres={t_unres} | "
          f"raw==tome round_end: {same_rounds}")

# cbb071c2: resolve the unresolved round with the engine per-team counter
key = "csds/2026/10/03/cbb071c2-83a9-4167-b4c1-6fed1e145da0/csds"
rounds = {r: w for r, w in te.filter(pl.col("match_key") == key).select(["round", "winner_team_code"]).iter_rows()}
pi = collections.defaultdict(dict)
for r, p, tc in tpi.filter(pl.col("match_key") == key).select(["round", "player_id", "team_code"]).iter_rows():
    pi[r][p] = tc
r1 = min(pi.keys())
A = {p for p, tc in pi[r1].items() if tc == 2}
B = {p for p, tc in pi[r1].items() if tc == 3}
print("cbb071c2 starter-T players (A):", sorted(A), "| starter-CT players (B):", sorted(B))
for r in sorted(rounds):
    pr = pi.get(r, {})
    print("  round", r, "winner", rounds[r], "| A present", sorted(p for p in A if p in pr), "A sides", [pr[p] for p in A if p in pr],
          "| B present", sorted(p for p in B if p in pr), "B sides", [pr[p] for p in B if p in pr])
su = {}
for r, tc, old, new in tsu.filter(pl.col("match_key") == key).select(["round", "team_code", "old_score", "new_score"]).iter_rows():
    su[r] = (tc, old, new)
print("cbb071c2 score_update rows (round: team_code, old->new):", {r: (v[0], v[1], v[2]) for r, v in sorted(su.items())})
# per-entity counter using resolved rounds only; for each round find which entity's counter predicts the engine new_score
aw = bw = 0
for r in sorted(rounds):
    pr = pi.get(r, {})
    sa = [pr[p] for p in A if p in pr]
    sb = [pr[p] for p in B if p in pr]
    W = rounds[r]
    if sa and sb:
        sA = collections.Counter(sa).most_common(1)[0][0]
        winner_ent = "A" if sA == W else "B"
    else:
        # unresolved: use engine counter (engine new_score must equal the winner entity's count + 1)
        cand = []
        tc, old, new = su.get(r, (None, None, None))
        if new is not None:
            if new == aw + 1:
                cand.append("A")
            if new == bw + 1:
                cand.append("B")
        winner_ent = f"UNRESOLVED engine-candidates={cand}"
        print("  round", r, "unresolved; engine new_score", new, "old", old, "A so far", aw, "B so far", bw, "=>", cand, "| winner code", W)
        if len(cand) == 1:
            winner_ent = cand[0]
    if winner_ent == "A":
        aw += 1
    elif winner_ent == "B":
        bw += 1
print("cbb071c2 final (engine-resolved) A (starter-T) =", aw, "| B (starter-CT) =", bw, "| header (t, ct) = (5, 7)")
