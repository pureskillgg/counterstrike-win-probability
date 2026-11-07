"""D3-4 verifier: which source is right at round 13 (and the round-after-swap)?
(a) header final-score reconstruction: starters' final score from winner_team_code vs from clean freeze-delta sides, for
    candidate swap schedules P (subsets of {8,12,24,27,30,33}). Only matches with contiguous rounds 1..N and N == header total.
(b) bomb-event check: bomb_exploded -> T (2), bomb_defused -> CT (3) vs winner_team_code and vs freeze-delta side, by round.
ASSUMPTIONS: header t_starters_score_final = score of team that started on T; swap after p means rounds > p are on swapped sides;
bomb_exploded/bomb_defused in round_state are the round's bomb outcome.
"""
import glob, sys, collections, itertools
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(240)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def folder(ch):
    return TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
def read(ch, cols=None):
    fs = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")
H = read("header", ["key", "t_starters_score_final", "ct_starters_score_final", "map_name"]).with_columns(
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day")).filter(pl.col("day") >= 12)
keys = H["key"].to_list(); keyset = set(keys)
hdr = {k: (int(t), int(c), mp) for k, t, c, mp in H.select(["key", "t_starters_score_final", "ct_starters_score_final", "map_name"]).iter_rows()}
RE = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RE = RE.filter(pl.col("match_key").is_in(keys))
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
RS = RS.filter(pl.col("match_key").is_in(keys))
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").group_by(["match_key", "round"])
      .agg(pl.col("t_score").first(), pl.col("ct_score").first()))
BX = set(RS.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round"]).iter_rows())
BD = set(RS.filter(pl.col("event_type") == "bomb_defused").select(["match_key", "round"]).iter_rows())
del RS
re = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
fe = {(m, r): (t, c) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
mr = collections.defaultdict(set)
for (m, r) in re: mr[m].add(r)
def dside(m, r):
    if (m, r) in fe and (m, r + 1) in fe:
        a, b = fe[(m, r)], fe[(m, r + 1)]
        dt, dc = b[0] - a[0], b[1] - a[1]
        if (dt, dc) == (1, 0): return 2
        if (dt, dc) == (0, 1): return 3
    return None
# (a) header reconstruction
SWAPSETS = [tuple(c) for n in range(0, 4) for c in itertools.combinations([8, 12, 24, 27, 30, 33], n)]
def starters_T(P, w_of, N):
    tot = 0
    for r in range(1, N + 1):
        k = sum(1 for p in P if p < r)
        on_T = (k % 2 == 0)
        w = w_of(r)
        if w is None: return None
        if (on_T and w == 2) or ((not on_T) and w == 3): tot += 1
    return tot
clean = []
for m in keys:
    rs = mr.get(m, set())
    N = max(rs) if rs else 0
    if N and rs == set(range(1, N + 1)) and N == hdr[m][0] + hdr[m][1]:
        clean.append(m)
print("day12-25 matches:", len(keys), "| contiguous 1..N with N == header total:", len(clean))
hits_code = collections.Counter(); hits_delta = collections.Counter()
match_code = {}; match_delta = {}
n_any_code = n_any_delta = 0
for m in clean:
    N = max(mr[m]); ht = hdr[m][0]
    sc = [P for P in SWAPSETS if starters_T(P, lambda r: re[(m, r)], N) == ht]
    sd = [P for P in SWAPSETS if starters_T(P, lambda r: (dside(m, r) if dside(m, r) is not None else re[(m, r)]), N) == ht]
    for P in sc: hits_code[P] += 1
    for P in sd: hits_delta[P] += 1
    n_any_code += bool(sc); n_any_delta += bool(sd)
    match_code[m] = sc; match_delta[m] = sd
print("matches consistent with >=1 schedule: code-based", n_any_code, "| delta-based", n_any_delta, "of", len(clean))
print("top schedules consistent with header (code-based):", hits_code.most_common(8))
print("top schedules consistent with header (delta-based):", hits_delta.most_common(8))
# does the {12} schedule (MR12 halftime only) fit, by source?
p12 = (12,)
print("schedule (12,) fits header: code-based", hits_code[p12], "| delta-based", hits_delta[p12], "of", len(clean))
# round-13 disagreement subset: does the header side with code or delta?
dis13 = [m for m in clean if 13 in mr[m] and dside(m, 13) is not None and dside(m, 13) != re[(m, 13)]]
print("\nclean matches with round-13 code vs delta disagreement:", len(dis13))
fit_code = sum(1 for m in dis13 if (12,) in match_code[m] or any(P for P in match_code[m] if 12 in P and 13 not in P))
fit_delta = sum(1 for m in dis13 if (12,) in match_delta[m] or any(P for P in match_delta[m] if 12 in P and 13 not in P))
print("  of these, header fits a 12-swap schedule with CODE:", fit_code, "| with DELTA:", fit_delta)
print("  header fits some schedule with CODE:", sum(1 for m in dis13 if match_code[m]), "| with DELTA:", sum(1 for m in dis13 if match_delta[m]))
# (b) bomb events
print("\n(b) bomb-outcome check, all day12-25 rounds with a bomb event and a round_end row:")
rows = collections.Counter()
for m in keys:
    for r in mr.get(m, ()):
        bomb = 2 if (m, r) in BX else (3 if (m, r) in BD else None)
        if bomb is None or (m, r) in BX and (m, r) in BD: continue
        grp = "r13" if r == 13 else ("other")
        w = re[(m, r)]
        d = dside(m, r)
        rows[(grp, "bomb==code" if bomb == w else "bomb!=code")] += 1
        if d is not None:
            rows[(grp, "delta_clean", "bomb==delta" if bomb == d else "bomb!=delta")] += 1
for k, v in sorted(rows.items()): print("  ", k, v)
# round 13 bomb agreement by swap state
print("\nround-13 bomb rows by whether header fits (12,) with CODE:")
b13 = collections.Counter()
for m in clean:
    if 13 not in mr[m]: continue
    bomb = 2 if (m, 13) in BX else (3 if (m, 13) in BD else None)
    if bomb is None: continue
    b13[("bomb==code" if bomb == re[(m, 13)] else "bomb!=code", "hdr_fits_(12,)_code" if (12,) in match_code[m] else "hdr_not_fit_(12,)_code")] += 1
for k, v in sorted(b13.items()): print("  ", k, v)
