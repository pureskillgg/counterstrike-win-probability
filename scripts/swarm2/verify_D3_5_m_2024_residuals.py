"""D3-5 verifier, step M (2024 G: tome): residual checks.

1. start2 matches (round 1 absent from round_end): infer round-1 winner from the round_state freeze_end of round 2
   (fields start at (0, 0) at match start; a unit +1 on T field means T won round 1, CT field means CT won).
   Then rerun the header truth test (order-free) for winners and fields, engine halftime h, M <= 2h only.
2. Unclear (non-unit) freeze deltas at the affected transitions: what they are.
3. The 4 round-13 bomb-mismatch matches with no lead disagreement: their round coverage.
4. 1 Hz sample labels (data\\state_sample\\rounds.parquet): schema and comparison with the tome for days 01-07 if keyed.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


H = rd("header", ["key", "t_starters_score_final", "ct_starters_score_final"])
hdr = {k: (t, c) for k, t, c in H.iter_rows()}
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
sec = RS.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("h"))
hmap = {m: h for m, h in sec.iter_rows()}
BS = rd("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bomb = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
W = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    W[m][r] = w
LIVE = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    LIVE[m][r] = (t, c)
BOMB = {}
for m, r, evs in bomb.iter_rows():
    BOMB[(m, r)] = CT if "bomb_defused" in evs else T

print("== 2024 G: tome, residual checks ==")
# ---------- 1. start2 header test with inferred round-1 winner
res = collections.Counter()
ex = []
for m in W:
    if m not in hdr or hdr[m][0] is None:
        continue
    rs = sorted(W[m])
    if not (rs and rs[0] == 2 and rs == list(range(2, len(rs) + 2))):
        continue
    if 2 not in LIVE[m]:
        res["no_freeze_2"] += 1
        continue
    t2, c2 = LIVE[m][2]
    if (t2, c2) == (1, 0):
        w1 = T
    elif (t2, c2) == (0, 1):
        w1 = CT
    else:
        res["round1_unclear"] += 1
        continue
    winners = dict(W[m])
    winners[1] = w1
    M = max(winners)
    ht, hc = hdr[m]
    if M != ht + hc:
        res["start2_not_complete"] += 1
        continue
    h = hmap.get(m, 12)
    tag = f"h{h}" if m in hmap else "hNA"
    res[f"start2_complete_{tag}"] += 1
    if M in LIVE[m]:
        t_n, c_n = LIVE[m][M]
        wM = winners[M]
        tf, cf = t_n + (wM == T), c_n + (wM == CT)
        res[f"start2_field_ok_{tag}"] += int(sorted([tf, cf]) == sorted([ht, hc]))
    if M <= 2 * h and m in hmap:
        a = sum(1 for r in winners if winners[r] == (T if r <= h else CT))
        b = M - a
        res[f"start2_winner_tested_{tag}"] += 1
        ok = (a, b) in [(ht, hc), (hc, ht)]
        res[f"start2_winner_ok_{tag}"] += int(ok)
        if not ok and len(ex) < 3:
            ex.append((m[-45:], M, ht, hc, a, b, h))
print("\n[1] start2 header truth test (round 1 inferred from freeze 2):")
for k in sorted(res):
    print("   ", k, res[k])
print("    winner failures (match, M, hdrT, hdrC, A, B, h):", ex)

# ---------- 2. unclear deltas at the affected transitions (contiguous)
unc = collections.Counter()
samples = []
for m in W:
    rs = sorted(W[m])
    if rs != list(range(1, len(rs) + 1)):
        continue
    h = hmap.get(m, 12)
    if h not in (8, 12):
        continue
    r = h + 1 if False else (h if False else None)
    tr = h  # transition after round h is the swap transition; the pistol round is h+1 (delta r=h+1)
    for rr in (h + 1,):
        if rr in LIVE[m] and (rr + 1) in LIVE[m]:
            t1, c1 = LIVE[m][rr]
            t2, c2 = LIVE[m][rr + 1]
            dd = (t2 - t1, c2 - c1)
            if dd not in [(1, 0), (0, 1)]:
                unc[(h, "unclear_at_pistol_delta")] += 1
                if len(samples) < 4:
                    samples.append((m[-45:], rr, LIVE[m][rr], LIVE[m][rr + 1], W[m].get(rr)))
        else:
            unc[(h, "pistol_delta_missing_freeze")] += 1
            if rr not in LIVE[m]:
                unc[(h, "freeze_missing_at_pistol")] += 1
            if rr + 1 not in LIVE[m]:
                unc[(h, "freeze_missing_after_pistol")] += 1
print("\n[2] contiguous matches: unclear or missing freeze at the first post-switch delta (h+1):")
for k in sorted(unc):
    print("   ", k, unc[k])
print("    samples (match, round, freeze_r, freeze_r+1, winner):", samples)

# ---------- 3. the 4 matches with round-13 bomb mismatch and no lead disagreement
def swap_h(h):
    return lambda r: r == h or (r > 2 * h and (r - 2 * h - 3) % 6 == 0)


aff = []
for (m, r), b in BOMB.items():
    if r == 13 and m in W and W[m].get(13) is not None and W[m][13] != b:
        aff.append(m)
aff_contig = [m for m in aff if sorted(W[m]) == list(range(1, len(W[m]) + 1))]
print("\n[3] round-13 bomb mismatch matches (contiguous, W[13] != bomb):", len(aff_contig))
for m in aff_contig:
    rs = sorted(W[m])
    cmp_rounds = [r for r in rs if r in LIVE[m]]
    if m in hmap and hmap[m] == 12:
        pass
    later = [r for r in cmp_rounds if r >= 14]
    if not later:
        print("    no live rounds >= 14:", m[-45:], "N =", len(rs), "h =", hmap.get(m))
print("    N (rounds recorded) distribution across affected contiguous matches:",
      sorted(collections.Counter(len(W[m]) for m in aff_contig).items())[:20])

# ---------- 4. 1 Hz sample labels
p = ROOT / "data" / "state_sample" / "rounds.parquet"
print("\n[4] data/state_sample/rounds.parquet:")
try:
    sch = pl.read_parquet_schema(str(p))
    print("    columns:", list(sch.keys()))
    df = pl.read_parquet(str(p))
    print("    rows:", df.height, "| distinct match cols:", [c for c in df.columns if "match" in c.lower()])
    print(df.head(3))
except Exception as exc:  # noqa: BLE001
    print("    read failed:", type(exc).__name__, str(exc)[:200])
print("\n== end of residual section ==")
