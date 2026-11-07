"""D6-4 independent verifier, part H:
H1 16-round (f16) round 9 in build 9961: score winner vs bomb outcome vs round_end, for any player_count (bomb rows).
   Tests whether the f16 first-post-swap inversion is also visible against the bomb outcome, since the arbiter is unusable there.
H2 2026 bomb-vs-round_end with the 5v5 code for 2026 (player_count == 10), since the claim's == 20 filter selects 0 rows in 2026.
"""
import sys, glob, collections
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
B26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed").collect()


def label(s, r):
    if r not in s or r + 1 not in s:
        return None
    t0, c0 = s[r]
    t1, c1 = s[r + 1]
    no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        return None
    if no:
        return T if (t1 - t0, c1 - c0) == (1, 0) else CT
    if sw:
        return T if (t1 - c0, c1 - t0) == (0, 1) else CT
    return None


print("== H1. 16-round format (score swap after round 8), build 9961 and 9842: round 9 ==")
H = load(B24, "header", ["key", "build_num"])
bm = {k: b for k, b in H.iter_rows()}
RS = load(B24, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"])
fe = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique(["match_key", "round"], keep="first")
scm = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm[m][r] = (t, c)
f16 = set()
for m, s in scm.items():
    t0, c0 = s.get(8, (None, None))
    if 8 in s and 9 in s:
        t1, c1 = s[9]
        if (t1 - c0, c1 - t0) in [(1, 0), (0, 1)] and (t1 - t0, c1 - c0) not in [(1, 0), (0, 1)]:
            f16.add(m)
RE = load(B24, "round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS = load(B24, "bomb_state", ["match_key", "round", "event_type"])
outs = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
outs = outs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
bomb = {(m, r): s for m, r, s in outs.select(["match_key", "round", "bomb_side"]).iter_rows()}
c = collections.defaultdict(collections.Counter)
for (m, r), (w, pc) in {(m, r): (w, pc) for m, r, w, pc in RE.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows()}.items():
    if r != 9 or m not in f16:
        continue
    b = bm.get(m)
    sc = label(scm[m], 9)
    bs = bomb.get((m, r))
    key = f"build {b}"
    if sc is not None:
        c[key]["score=rE" if sc == w else "score!=rE"] += 1
    if bs is not None:
        c[key]["bomb=rE" if bs == w else "bomb!=rE"] += 1
        if sc is not None:
            c[key]["bomb=score" if bs == sc else "bomb!=score"] += 1
for k in sorted(c):
    print(f"  {k}: {dict(c[k])}")

print("\n== H2. 2026 bomb-outcome rounds with the 5v5 code (player_count == 10): bomb vs round_end ==")
RE26 = load(B26, "round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS26 = load(B26, "bomb_state", ["match_key", "round", "event_type"])
o26 = (BS26.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
       .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
o26 = o26.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
J26 = RE26.filter(pl.col("player_count") == 10).join(o26, on=["match_key", "round"], how="inner")
print(f"  2026 bomb rows (player_count==10): n={J26.height}; mismatch={int((J26['bomb_side'] != J26['winner_team_code']).sum())}")
r13 = J26.filter(pl.col("round") == 13)
print(f"  2026 round-13 bomb rows (player_count==10): n={r13.height}; mismatch={int((r13['bomb_side'] != r13['winner_team_code']).sum())}")
